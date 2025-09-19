# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 00:29:09 2025

@author: kmkho
"""

"""
Flask server for the phishing detection model
"""

import os
import json
import pickle
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from urllib.parse import urlparse
import tldextract
import re
from transformers import BertTokenizer
from torch.nn import functional as F
import logging
import url_features  # Import the URL feature extraction module

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='phishing_server.log'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define paths - MODIFIED: Handle case when __file__ is not defined (e.g., in interactive environments)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # If running in interactive environment like Jupyter notebook
    SCRIPT_DIR = os.getcwd()  # Use current working directory instead
    print(f"Running in interactive environment, using current directory: {SCRIPT_DIR}")

MODEL_DIR = os.path.join(SCRIPT_DIR, 'model_artifacts')
os.makedirs(MODEL_DIR, exist_ok=True)

logger.info(f"Looking for model files in: {MODEL_DIR}")

# Load model components
class HybridBERTModel(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_engineered_features=None):
        super(HybridBERTModel, self).__init__()
        
        from transformers import BertModel, BertConfig
        
        # Initialize BERT with a configuration that allows for gradient checkpointing
        self.bert_config = BertConfig.from_pretrained(bert_model_name)
        self.bert_config.gradient_checkpointing = True  # Memory optimization
        self.bert = BertModel.from_pretrained(bert_model_name, config=self.bert_config)

        # BERT output size
        bert_hidden_size = self.bert.config.hidden_size  # Usually 768
        
        # Attention mechanism for engineered features
        self.feature_attention = torch.nn.Sequential(
            torch.nn.Linear(num_engineered_features, num_engineered_features),
            torch.nn.Tanh(),
            torch.nn.Linear(num_engineered_features, 1, bias=False),
            torch.nn.Softmax(dim=1)
        )
        
        # Process engineered features separately before combining
        self.feature_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_engineered_features, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        
        # Process BERT embeddings separately
        self.bert_encoder = torch.nn.Sequential(
            torch.nn.Linear(bert_hidden_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        
        # Combined size after separate processing
        combined_size = 256 + 128  # BERT encoder output + feature encoder output
        
        # Fully connected layers with residual connections
        self.fc1 = torch.nn.Linear(combined_size, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        
        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        
        self.fc3 = torch.nn.Linear(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        
        # Output layer
        self.classifier = torch.nn.Linear(64, 2)  # 2 output classes
        
        # Activation functions
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()  # Sometimes performs better than ReLU
        
        # Dropout layers with different rates
        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.5)  # Higher dropout for later layers
        
        # Store intermediate outputs for interpretability
        self.bert_output = None
        self.feature_output = None
        self.combined_output = None
        self.feature_attention_weights = None
        self.bert_attentions = None
        self.bert_hidden_states = None

    def forward(self, input_ids, attention_mask, engineered_features):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=True,  # Get attention weights for interpretability
            output_hidden_states=True  # Get all hidden states
        )
        
        pooled_output = outputs.pooler_output  # [CLS] token embedding
        self.bert_attentions = outputs.attentions  # Store for interpretability
        self.bert_hidden_states = outputs.hidden_states  # Store for interpretability
        
        # Apply dropout to BERT output
        pooled_output = self.dropout1(pooled_output)
        
        # Process BERT output
        bert_encoded = self.bert_encoder(pooled_output)
        self.bert_output = bert_encoded  # Store for interpretability
        
        # Apply attention to engineered features
        feature_attention = self.feature_attention(engineered_features)
        self.feature_attention_weights = feature_attention  # Store for interpretability
        
        # Process engineered features
        feature_encoded = self.feature_encoder(engineered_features)
        self.feature_output = feature_encoded  # Store for interpretability
        
        # Concatenate BERT output with engineered features
        combined = torch.cat((bert_encoded, feature_encoded), dim=1)
        self.combined_output = combined  # Store for interpretability
        
        # First fully connected block with residual connection
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        
        # Second fully connected block
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        
        # Third fully connected block
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        
        # Output layer
        logits = self.classifier(x)
        
        return logits
        
    # Method to get attention weights for explainability
    def get_attention_weights(self):
        return {
            'bert_attentions': self.bert_attentions,
            'feature_attention': self.feature_attention_weights
        }

    # Method to get intermediate outputs for explainability
    def get_intermediate_outputs(self):
        return {
            'bert_output': self.bert_output,
            'feature_output': self.feature_output,
            'combined_output': self.combined_output
        }

def load_model_components():
    """Load all necessary components for prediction"""
    logger.info("Loading model components...")
    
    try:
        # Check if model files exist
        required_files = [
            os.path.join(MODEL_DIR, 'tokenizer'),
            os.path.join(MODEL_DIR, 'scaler.pkl'),
            os.path.join(MODEL_DIR, 'feature_names.json'),
            os.path.join(MODEL_DIR, 'model_config.json'),
            os.path.join(MODEL_DIR, 'best_phishing_model.pt')
        ]
        
        # Check each file
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"Missing required file: {file_path}")
        
        # Check for GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer')
        if os.path.exists(tokenizer_path):
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            logger.info("Tokenizer loaded successfully")
        else:
            logger.warning("Tokenizer not found, using default 'bert-base-uncased'")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")
        else:
            logger.error("Scaler file not found")
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
        # Load feature names
        feature_names_path = os.path.join(MODEL_DIR, 'feature_names.json')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Feature names loaded successfully: {len(feature_names)} features")
        else:
            logger.error("Feature names file not found")
            raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")
        
        # Load model configuration
        model_config_path = os.path.join(MODEL_DIR, 'model_config.json')
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
            logger.info("Model config loaded successfully")
        else:
            logger.warning("Model config file not found, using defaults")
            model_config = {}
        
        # Initialize model
        model = HybridBERTModel(
            bert_model_name='bert-base-uncased',
            num_engineered_features=len(feature_names)
        ).to(device)
        
        # Load model weights
        model_weights_path = os.path.join(MODEL_DIR, 'best_phishing_model.pt')
        if os.path.exists(model_weights_path):
            # Modified: Load with strict=False to handle missing keys
            model_state = torch.load(model_weights_path, map_location=device)
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
            
            # Log any missing or unexpected keys
            if missing_keys:
                logger.warning(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")
                
            logger.info("Model weights loaded successfully with compatible keys")
        else:
            logger.error("Model weights file not found")
            raise FileNotFoundError(f"Model weights file not found at {model_weights_path}")
        
        model.eval()
        
        logger.info("Model components loaded successfully.")
        return model, tokenizer, scaler, feature_names, device
    
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        raise

# Initialize model components
try:
    model, tokenizer, scaler, feature_names, device = load_model_components()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    # Create placeholders if model loading fails - will return error responses
    model, tokenizer, scaler, feature_names, device = None, None, None, [], None

# ADDED: Fallback prediction when model isn't loaded
def fallback_prediction(url):
    """Simple heuristics-based prediction when the model isn't available"""
    suspicious_keywords = ['login', 'secure', 'account', 'banking', 'verify', 'password', 'update']
    
    # Check for suspicious patterns in the URL
    risk_factors = []
    
    # Parse the URL
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Check for IP address instead of domain name
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
            risk_factors.append("URL uses IP address instead of domain name")
        
        # Check for suspicious keywords in domain
        for keyword in suspicious_keywords:
            if keyword in domain.lower():
                risk_factors.append(f"Domain contains suspicious keyword: '{keyword}'")
        
        # Check for excessive subdomains
        subdomain_count = domain.count('.')
        if subdomain_count > 3:
            risk_factors.append(f"Excessive subdomains ({subdomain_count})")
        
        # Check for URL length
        if len(url) > 100:
            risk_factors.append(f"Unusually long URL ({len(url)} characters)")
        
        # Check for many query parameters
        if len(parsed_url.query) > 50:
            risk_factors.append("Excessive query parameters")
            
        # Determine phishing probability based on risk factors
        is_phishing = len(risk_factors) > 0
        confidence = min(0.5 + (len(risk_factors) * 0.1), 0.9) if is_phishing else 0.7
        
        logger.info(f"Using fallback prediction for {url}: {is_phishing} (confidence: {confidence})")
        
        return {
            'url': url,
            'is_phishing': is_phishing,
            'phishing_probability': confidence if is_phishing else (1 - confidence),
            'confidence': confidence,
            'prediction': 'Possibly Phishing' if is_phishing else 'Probably Legitimate',
            'risk_factors': risk_factors,
            'risk_level': 'Medium' if is_phishing else 'Low',
            'note': 'Using basic heuristics (model not loaded)'
        }
        
    except Exception as e:
        logger.error(f"Error in fallback prediction: {e}")
        return {
            'url': url,
            'is_phishing': False,
            'confidence': 0.5,
            'prediction': 'Unknown',
            'risk_factors': ["Unable to analyze URL"],
            'note': 'Error in analysis'
        }

def preprocess_url(url):
    """Preprocess URL for tokenization"""
    # Replace special characters with spaces around them
    for char in ['/', '.', '-', '=', '?', '&', '_', ':', '@']:
        url = url.replace(char, f' {char} ')
    # Additional preprocessing
    url = url.replace('http', 'http ')
    url = url.replace('https', 'https ')
    url = url.replace('www', 'www ')
    return url

def predict_url(url):
    """Predict whether a URL is phishing or legitimate"""
    try:
        if not model or not tokenizer or not scaler:
            logger.warning("Model components not loaded, using fallback prediction")
            return fallback_prediction(url)
            
        logger.info(f"Predicting URL: {url}")
        
        # Preprocess URL for BERT
        processed_url = preprocess_url(url)
        
        # Tokenize URL
        encoded = tokenizer.encode_plus(
            processed_url,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Extract engineered features using the URL features module
        url_analysis = url_features.analyze_url(url)
        extracted_features = url_analysis['features']
        risk_factors = url_analysis['analysis']['risk_factors']
        
        # Create a feature vector based on the expected feature names
        feature_vector = []
        for feature in feature_names:
            # If the feature exists in our extracted features, use it
            # Otherwise use 0 as a default value
            feature_vector.append(extracted_features.get(feature, 0))
        
        # Convert to numpy array and reshape
        feature_array = np.array([feature_vector])
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, features_tensor)
            probs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, dim=1)
        
        # Get prediction results
        is_phishing = bool(pred.item())
        phishing_prob = float(probs[0, 1].item())
        confidence = float(probs[0, pred.item()].item())
        
        # Create result dictionary with risk factors
        result = {
            'url': url,
            'is_phishing': is_phishing,
            'phishing_probability': phishing_prob,
            'confidence': confidence,
            'prediction': 'Phishing' if is_phishing else 'Legitimate',
            'risk_factors': risk_factors,
            'risk_level': url_analysis['analysis']['risk_level']
        }
        
        logger.info(f"Prediction result: {result['prediction']} with confidence {confidence:.4f}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return fallback_prediction(url)
# Add these endpoints to your app.py file

@app.route('/api/debug-features', methods=['POST'])
def debug_features():
    """Debug endpoint to see extracted features"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing URL parameter'}), 400
            
        url = data['url']
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        logger.info(f"Debug features request for URL: {url}")
        
        # Get features
        url_analysis = url_features.analyze_url(url, debug_mode=True)
        extracted_features = url_analysis['features']
        
        # Create feature vector in the same order as feature_names
        feature_vector = []
        feature_values = {}
        missing_features = []
        for feature in feature_names:
            if feature in extracted_features:
                value = extracted_features.get(feature, 0)
                feature_vector.append(value)
                feature_values[feature] = value
            else:
                missing_features.append(feature)
                feature_vector.append(0)
                feature_values[feature] = 0
            
        # Scale features if scaler is available
        scaled_dict = {}
        if scaler:
            try:
                feature_array = np.array([feature_vector])
                scaled_features = scaler.transform(feature_array)
                scaled_dict = {feature_names[i]: float(scaled_features[0][i]) 
                              for i in range(len(feature_names))}
            except Exception as e:
                scaled_dict = {"error": f"Scaling error: {str(e)}"}
        else:
            scaled_dict = {"error": "Scaler not loaded"}
            
        # Check if any expected features are missing
        extra_features = [f for f in extracted_features if f not in feature_names 
                         and f not in ['URL', 'Domain', 'TLD', 'Title', 'FILENAME']]
        
        # Get BERT preprocessing results
        processed_url = preprocess_url(url)
        
        # Return detailed debug info
        return jsonify({
            'url': url,
            'model_loaded': model is not None,
            'feature_count': len(feature_names),
            'feature_names_order': feature_names,
            'extracted_features': feature_values,
            'missing_features': missing_features,
            'extra_features': extra_features,
            'scaled_features': scaled_dict,
            'bert_preprocessing': processed_url,
            'risk_analysis': url_analysis['analysis'],
            'meta_features': {
                'URL': extracted_features.get('URL', ''),
                'Domain': extracted_features.get('Domain', ''),
                'TLD': extracted_features.get('TLD', '')
            }
        })
        
    except Exception as e:
        logger.error(f"Error in debug-features endpoint: {e}")
        return jsonify({
            'error': str(e),
            'traceback': str(traceback.format_exc())
        }), 500

@app.route('/api/test-known-urls', methods=['GET'])
def test_known_urls():
    """Test predictions on known URLs"""
    try:
        # Add URLs from your training data that you know the correct classification for
        known_urls = [
            {"url": "https://www.google.com", "expected": False},
            {"url": "https://www.facebook.com", "expected": False},
            {"url": "https://login-secure-paypal.com/verify", "expected": True},
            {"url": "http://192.168.1.1/admin", "expected": True}
            # Add more URLs from your dataset
        ]
        
        results = []
        for item in known_urls:
            url = item["url"]
            expected = item["expected"]
            
            prediction = predict_url(url)
            matches_expected = prediction['is_phishing'] == expected
            
            results.append({
                'url': url,
                'expected': "Phishing" if expected else "Legitimate",
                'predicted': prediction['prediction'],
                'is_phishing': prediction['is_phishing'],
                'confidence': prediction['confidence'],
                'matches_expected': matches_expected,
                'risk_factors': prediction['risk_factors']
            })
        
        success_rate = sum(1 for r in results if r['matches_expected']) / len(results) if results else 0
            
        return jsonify({
            'known_url_tests': results,
            'success_rate': success_rate,
            'model_loaded': model is not None,
            'using_fallback': model is None
        })
        
    except Exception as e:
        logger.error(f"Error in test-known-urls endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare-predictions', methods=['POST'])
def compare_predictions():
    """Compare ML prediction with fallback prediction"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing URL parameter'}), 400
            
        url = data['url']
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        # Get fallback prediction
        fallback_result = fallback_prediction(url)
        
        # Try to get ML prediction if model is available
        ml_result = None
        if model and tokenizer and scaler:
            try:
                # This is a copy of the ML prediction code from predict_url but without fallback
                processed_url = preprocess_url(url)
                
                encoded = tokenizer.encode_plus(
                    processed_url,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                url_analysis = url_features.analyze_url(url)
                extracted_features = url_analysis['features']
                risk_factors = url_analysis['analysis']['risk_factors']
                
                feature_vector = []
                for feature in feature_names:
                    feature_vector.append(extracted_features.get(feature, 0))
                
                feature_array = np.array([feature_vector])
                scaled_features = scaler.transform(feature_array)
                features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask, features_tensor)
                    probs = F.softmax(outputs, dim=1)
                    _, pred = torch.max(outputs, dim=1)
                
                is_phishing = bool(pred.item())
                phishing_prob = float(probs[0, 1].item())
                confidence = float(probs[0, pred.item()].item())
                
                ml_result = {
                    'url': url,
                    'is_phishing': is_phishing,
                    'phishing_probability': phishing_prob,
                    'confidence': confidence,
                    'prediction': 'Phishing' if is_phishing else 'Legitimate',
                    'risk_factors': risk_factors,
                    'risk_level': url_analysis['analysis']['risk_level']
                }
            except Exception as e:
                ml_result = {'error': f"ML prediction failed: {str(e)}"}
        
        # Return comparison
        return jsonify({
            'url': url,
            'model_loaded': model is not None,
            'fallback_prediction': fallback_result,
            'ml_prediction': ml_result,
            'predictions_match': (ml_result and ml_result.get('is_phishing') == fallback_result.get('is_phishing')) 
                               if ml_result and 'error' not in ml_result else False
        })
        
    except Exception as e:
        logger.error(f"Error in compare-predictions endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Add these imports at the top of your file
import traceback
@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    return '''
    <html>
        <head>
            <title>Phishing Detection API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1 {
                    color: #333;
                }
                .container {
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    margin-top: 20px;
                }
                pre {
                    background-color: #eee;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
                code {
                    font-family: monospace;
                }
                .footer {
                    margin-top: 30px;
                    text-align: center;
                    color: #777;
                    font-size: 0.8em;
                }
            </style>
        </head>
        <body>
            <h1>Phishing Detection API</h1>
            <div class="container">
                <h2>API Endpoints</h2>
                <ul>
                    <li><code>/api/check-url</code> - POST endpoint to check a URL</li>
                    <li><code>/api/analyze-url</code> - POST endpoint to analyze URL features</li>
                    <li><code>/api/health</code> - GET endpoint to check server health</li>
                </ul>
                
                <h2>Example Request</h2>
                <pre><code>
curl -X POST http://localhost:5000/api/check-url \\
    -H "Content-Type: application/json" \\
    -d '{"url": "https://example.com"}'
                </code></pre>
                
                <h2>Example Response</h2>
                <pre><code>
{
    "url": "https://example.com",
    "is_phishing": false,
    "phishing_probability": 0.01,
    "confidence": 0.99,
    "prediction": "Legitimate",
    "risk_factors": [],
    "risk_level": "Low"
}
                </code></pre>
            </div>
            
            <div class="footer">
                <p>Powered by PyTorch & BERT | Phishing Detection API</p>
            </div>
        </body>
    </html>
    '''

@app.route('/setup', methods=['GET'])
def setup_help():
    """Setup help page"""
    # New page to help with model setup
    return '''
    <html>
        <head>
            <title>Phishing Detection API Setup</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1, h2, h3 {
                    color: #333;
                }
                .container {
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    margin-top: 20px;
                }
                code {
                    background-color: #eee;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-family: monospace;
                }
                pre {
                    background-color: #eee;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-family: monospace;
                }
                .warning {
                    color: #856404;
                    background-color: #fff3cd;
                    border: 1px solid #ffeeba;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                .note {
                    color: #004085;
                    background-color: #cce5ff;
                    border: 1px solid #b8daff;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <h1>Phishing Detection API Setup Guide</h1>
            
            <div class="warning">
                <h3>⚠️ Model Not Loaded</h3>
                <p>The API is currently running in fallback mode because the model files could not be loaded.</p>
            </div>
            
            <div class="container">
                <h2>Required Model Files</h2>
                <p>The following files need to be placed in the model_artifacts directory:</p>
                <ul>
                    <li><code>tokenizer/</code> - Directory containing BERT tokenizer files</li>
                    <li><code>scaler.pkl</code> - Pickle file with feature scaler</li>
                    <li><code>feature_names.json</code> - JSON file with feature names</li>
                    <li><code>model_config.json</code> - JSON file with model configuration</li>
                    <li><code>best_phishing_model.pt</code> - PyTorch model weights file</li>
                </ul>
                
                <h2>Model Directory Location</h2>
                <p>The server is looking for model files in:</p>
                <pre>''' + MODEL_DIR + '''</pre>
                
                <h2>How to Fix</h2>
                <ol>
                    <li>Make sure all required files are placed in the model directory</li>
                    <li>Check file permissions to ensure the server can read the files</li>
                    <li>Restart the server after placing the files</li>
                </ol>
                
                <div class="note">
                    <p><strong>Note:</strong> Until the model files are properly set up, the API will use basic heuristics for phishing detection with lower accuracy.</p>
                </div>
            </div>
            
            <div class="container">
                <h2>Server Status</h2>
                <p>Check the current server status:</p>
                <pre><a href="/api/health" target="_blank">/api/health</a></pre>
            </div>
        </body>
    </html>
    '''

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Demo page to test the API directly in browser"""
    result = None
    url = ''
    
    if request.method == 'POST':
        url = request.form.get('url', '')
        if url:
            # Basic URL validation
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            # Get prediction
            result = predict_url(url)
    
    # Create model notice message
    notice_div = '''
    <div class="notice">
        <strong>Notice:</strong> The ML model is not currently loaded. Using basic heuristics instead.
        <a href="/setup">Click here for setup instructions</a>
    </div>
    ''' if model is None else ''
    
    # Create result div if we have results
    result_div = ""
    if result:
        is_phishing = result.get('is_phishing', False)
        risk_factors = result.get('risk_factors', [])
        result_class = 'phishing' if is_phishing else 'legitimate'
        
        result_note = f'<p><em>Note: {result.get("note")}</em></p>' if 'note' in result else ''
        
        risk_factors_html = '<p>No significant risk factors detected.</p>' if not risk_factors else '<ul>' + ''.join([f'<li>{factor}</li>' for factor in risk_factors]) + '</ul>'
        
        result_div = f'''
        <div class="result {result_class}">
            <h3>Result: {result.get('prediction')}</h3>
            <p>URL: {result.get('url')}</p>
            <p>Confidence: {result.get('confidence') * 100:.2f}%</p>
            <p>Risk Level: {result.get('risk_level', 'Unknown')}</p>
            
            <div class="risk-factors">
                <h4>Risk Factors:</h4>
                {risk_factors_html}
            </div>
            
            {result_note}
        </div>
        '''

    # Create simple HTML demo page
    return f'''
    <html>
        <head>
            <title>Phishing Detection Demo</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }}
                h1 {{
                    color: #333;
                }}
                .container {{
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    margin-top: 20px;
                }}
                .form-group {{
                    margin-bottom: 15px;
                }}
                input[type="text"] {{
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                button {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #45a049;
                }}
                .result {{
                    margin-top: 20px;
                    padding: 15px;
                    border-radius: 5px;
                }}
                .phishing {{
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    color: #721c24;
                }}
                .legitimate {{
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                }}
                .risk-factors {{
                    margin-top: 15px;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    color: #777;
                    font-size: 0.8em;
                }}
                .notice {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeeba;
                    color: #856404;
                    padding: 10px;
                    margin-bottom: 15px;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <h1>Phishing URL Detection Demo</h1>
            
            {notice_div}
            
            <div class="container">
                <form method="POST" action="/demo">
                    <div class="form-group">
                        <label for="url">Enter URL to check:</label>
                        <input type="text" id="url" name="url" value="{url}" placeholder="https://example.com" required>
                    </div>
                    <button type="submit">Check URL</button>
                </form>
                
                {result_div}
            </div>
            
            <div class="footer">
                <p>Powered by PyTorch & BERT | Phishing Detection API</p>
            </div>
        </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))