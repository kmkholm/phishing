import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import random
import urllib.parse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from tqdm import tqdm
from collections import defaultdict
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import io
import base64
from PIL import Image
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#===============================================================================
# 1. Meta-Dataset for Few-Shot Learning
#===============================================================================
class URLMetaDataset:
    """
    Meta-dataset for few-shot learning with URLs.
    Manages episodes, tasks, and support/query sets.
    """
    def __init__(self, url_source, n_way=2, k_shot=5, query_size=15, cache_size=1000):
        self.n_way = n_way  # Number of classes (usually 2: phishing/legitimate)
        self.k_shot = k_shot  # Number of examples per class for support set
        self.query_size = query_size  # Number of examples per class in query set
        self.cache_size = cache_size
        self.url_source = url_source
        self.feature_cache = {}  # Cache for extracted features
        
        # Load phishing URLs from external source
        self.phishing_urls = self._load_phishing_urls()
        
        # Generate legitimate URLs (could be from real dataset)
        self.legitimate_domains = [
            'google.com', 'microsoft.com', 'amazon.com', 'apple.com',
            'github.com', 'youtube.com', 'wikipedia.org', 'linkedin.com',
            'twitter.com', 'instagram.com', 'reddit.com', 'netflix.com',
            'facebook.com', 'ebay.com', 'adobe.com', 'dropbox.com',
            'paypal.com', 'wordpress.org', 'zoom.us', 'walmart.com'
        ]
        self.legitimate_paths = [
            '', '/', '/index.html', '/about', '/contact', '/products',
            '/services', '/blog', '/login', '/account', '/help',
            '/support', '/news', '/search', '/category', '/cart',
            '/checkout', '/terms', '/privacy', '/faq'
        ]
        
        # Pre-generate some legitimate URLs
        self.legitimate_urls = self._generate_legitimate_urls(1000)
        
        # Create tasks by grouping URLs by domain/patterns
        self._create_task_pools()
    
    def _load_phishing_urls(self):
        """Load phishing URLs from external source"""
        try:
            # Try to load from GitHub repo
            response = requests.get(self.url_source)
            if response.status_code == 200:
                urls = response.text.strip().split('\n')
                return urls
            else:
                # Fallback to synthetic phishing URLs
                return self._generate_synthetic_phishing_urls(1000)
        except Exception as e:
            print(f"Error loading phishing URLs: {e}")
            return self._generate_synthetic_phishing_urls(1000)
    
    def _generate_synthetic_phishing_urls(self, count):
        """Generate synthetic phishing URLs for testing"""
        phishing_domains = [
            'paypa1.com', 'g00gle.com', 'amaz0n.com', 'faceb00k.com',
            'micr0s0ft.com', 'appl3.com', 'netfl1x.com', 'tw1tter.com',
            'inst4gram.com', 'paypal-secure.com', 'account-verify.net',
            'login-secure-server.com', 'verification-account.com',
            'banking-update.com', 'security-check.net', 'update-account.org'
        ]
        phishing_paths = [
            '/login', '/signin', '/account', '/verify', '/secure',
            '/update', '/password', '/reset', '/confirm', '/validation',
            '/security', '/authenticate', '/verification', '/authorize',
            '/access', '/wallet', '/billing', '/payment'
        ]
        
        urls = []
        for _ in range(count):
            domain = random.choice(phishing_domains)
            path = random.choice(phishing_paths)
            query = ''
            if random.random() > 0.5:
                query = f"?account={random.randint(10000, 99999)}&verify=true"
            urls.append(f"http://{domain}{path}{query}")
        return urls
    
    def _generate_legitimate_urls(self, count):
        """Generate legitimate URLs"""
        urls = []
        for _ in range(count):
            domain = random.choice(self.legitimate_domains)
            path = random.choice(self.legitimate_paths)
            query = ''
            if random.random() > 0.7:
                query = f"?q={random.choice(['search', 'product', 'info', 'help'])}"
            urls.append(f"https://{domain}{path}{query}")
        return urls

    def _create_task_pools(self):
        """Group URLs into tasks based on patterns/domains"""
        # Group phishing URLs by domain
        self.phishing_tasks = {}
        for url in self.phishing_urls:
            try:
                domain = urllib.parse.urlparse(url).netloc
                base_domain = '.'.join(domain.split('.')[-2:])
                if base_domain not in self.phishing_tasks:
                    self.phishing_tasks[base_domain] = []
                self.phishing_tasks[base_domain].append(url)
            except:
                continue
        
        # Keep only domains with enough examples
        min_examples = self.k_shot + self.query_size
        self.phishing_tasks = {k: v for k, v in self.phishing_tasks.items() if len(v) >= min_examples}
        
        # Create legitimate URL tasks
        self.legitimate_tasks = {}
        for domain in self.legitimate_domains:
            domain_urls = [url for url in self.legitimate_urls if domain in url]
            if len(domain_urls) >= min_examples:
                self.legitimate_tasks[domain] = domain_urls
            else:
                # Generate more URLs for this domain if needed
                additional = min_examples - len(domain_urls)
                for _ in range(additional):
                    path = random.choice(self.legitimate_paths)
                    query = ''
                    if random.random() > 0.7:
                        query = f"?q={random.choice(['search', 'product', 'info', 'help'])}"
                    domain_urls.append(f"https://{domain}{path}{query}")
                self.legitimate_tasks[domain] = domain_urls
    
    def extract_features(self, url):
        """Extract features from a URL"""
        # Check cache first
        if url in self.feature_cache:
            return self.feature_cache[url]
        
        # Parse the URL
        try:
            parsed_url = urllib.parse.urlparse(url)
        except:
            # Return default features for invalid URLs
            default_features = {
                "domain_length": 0, "has_www": 0, "has_subdomain": 0,
                "is_ip": 0, "path_length": 0, "path_depth": 0,
                "has_suspicious_path": 0, "has_query": 0, "query_length": 0,
                "query_param_count": 0, "has_https": 0, "url_special_chars": 0,
                "domain_special_chars": 0
            }
            return default_features
        
        # Extract features
        features = {
            # Domain-based features
            "domain_length": len(parsed_url.netloc),
            "has_www": 1 if parsed_url.netloc.startswith('www.') else 0,
            "has_subdomain": 1 if parsed_url.netloc.count('.') > 1 else 0,
            "is_ip": 1 if all(c.isdigit() or c == '.' for c in parsed_url.netloc) else 0,

            # Path-based features
            "path_length": len(parsed_url.path),
            "path_depth": parsed_url.path.count('/'),
            "has_suspicious_path": 1 if any(word in parsed_url.path.lower() 
                                            for word in ['login', 'signin', 'account', 'password', 
                                                        'secure', 'update', 'verify']) else 0,

            # Query-based features
            "has_query": 1 if len(parsed_url.query) > 0 else 0,
            "query_length": len(parsed_url.query),
            "query_param_count": parsed_url.query.count('&') + 1 if parsed_url.query else 0,

            # Security features
            "has_https": 1 if parsed_url.scheme == 'https' else 0,

            # Special characters
            "url_special_chars": sum(1 for c in url if not c.isalnum() and c not in '.-/:?=&'),
            "domain_special_chars": sum(1 for c in parsed_url.netloc if not c.isalnum() and c not in '.-')
        }
        
        # Cache the result if not full
        if len(self.feature_cache) < self.cache_size:
            self.feature_cache[url] = features
            
        return features
    
    def preprocess_url(self, url):
        """Preprocess URL to make it more suitable for tokenization"""
        # Replace special characters with spaces around them to help tokenization
        for char in ['/', '.', '-', '=', '?', '&', '_', ':', '@']:
            url = url.replace(char, f' {char} ')
        # Additional preprocessing specific to URLs
        url = url.replace('http', 'http ')
        url = url.replace('https', 'https ')
        url = url.replace('www', 'www ')
        return url
    
    def sample_episode(self):
        """
        Sample a few-shot episode with N-way K-shot format.
        For phishing detection, N=2 typically (phishing/legitimate)
        
        Returns:
            support_set: K examples per class for training
            query_set: Query examples for evaluation
            task_description: Information about the current task
        """
        # For phishing, we'll create tasks with different domains/patterns
        # Sample a phishing domain and a legitimate domain
        available_phishing = list(self.phishing_tasks.keys())
        available_legitimate = list(self.legitimate_tasks.keys())
        
        if not available_phishing or not available_legitimate:
            raise ValueError("Not enough data to create episodes")
            
        phishing_domain = random.choice(available_phishing)
        legitimate_domain = random.choice(available_legitimate)
        
        # Get all URLs for these domains
        phishing_urls = self.phishing_tasks[phishing_domain]
        legitimate_urls = self.legitimate_tasks[legitimate_domain]
        
        # Create support set (K examples per class)
        support_urls = {
            'phishing': random.sample(phishing_urls, self.k_shot),
            'legitimate': random.sample(legitimate_urls, self.k_shot)
        }
        
        # Remove support URLs from the pool to create query set
        remaining_phishing = [url for url in phishing_urls if url not in support_urls['phishing']]
        remaining_legitimate = [url for url in legitimate_urls if url not in support_urls['legitimate']]
        
        # Create query set (for evaluation)
        query_urls = {
            'phishing': random.sample(remaining_phishing, min(self.query_size, len(remaining_phishing))),
            'legitimate': random.sample(remaining_legitimate, min(self.query_size, len(remaining_legitimate)))
        }
        
        # Combine URLs with labels and extract features
        support_set = []
        for label, urls in support_urls.items():
            label_idx = 1 if label == 'phishing' else 0
            for url in urls:
                features = self.extract_features(url)
                support_set.append({
                    'url': url,
                    'label': label_idx,
                    'features': features
                })
        
        query_set = []
        for label, urls in query_urls.items():
            label_idx = 1 if label == 'phishing' else 0
            for url in urls:
                features = self.extract_features(url)
                query_set.append({
                    'url': url,
                    'label': label_idx,
                    'features': features
                })
        
        # Shuffle the sets
        random.shuffle(support_set)
        random.shuffle(query_set)
        
        task_description = {
            'phishing_domain': phishing_domain,
            'legitimate_domain': legitimate_domain,
            'n_way': self.n_way,
            'k_shot': self.k_shot
        }
        
        return support_set, query_set, task_description
        
    def get_feature_descriptions(self):
        """Get descriptions of features for XAI"""
        return {
            "domain_length": "Length of the domain name",
            "has_www": "Whether the domain starts with 'www'",
            "has_subdomain": "Whether the domain has subdomains",
            "is_ip": "Whether the domain is an IP address",
            "path_length": "Length of the URL path",
            "path_depth": "Depth of the URL path (number of / characters)",
            "has_suspicious_path": "Whether the path contains suspicious words like 'login', 'verify', etc.",
            "has_query": "Whether the URL contains query parameters",
            "query_length": "Length of the query string",
            "query_param_count": "Number of query parameters",
            "has_https": "Whether the URL uses HTTPS",
            "url_special_chars": "Number of special characters in the entire URL",
            "domain_special_chars": "Number of special characters in the domain"
        }
    
    def analyze_url_components(self, url):
        """Analyze URL components for XAI visualization"""
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Split URL into components
            components = {
                "scheme": parsed.scheme,
                "domain": parsed.netloc,
                "path": parsed.path,
                "query": parsed.query,
                "fragment": parsed.fragment
            }
            
            # Check for suspicious elements
            suspicious_words = [
                'login', 'signin', 'account', 'password', 'secure', 'verify', 
                'confirm', 'update', 'bank', 'paypal', 'billing', 'payment'
            ]
            
            suspiciousness = {}
            
            # Check domain
            domain_sus = 0
            domain = parsed.netloc.lower()
            if any(word in domain for word in suspicious_words):
                domain_sus += 0.5
            if domain.count('.') > 2:  # Multiple subdomains
                domain_sus += 0.3
            if any(c.isdigit() for c in domain):  # Digits in domain
                domain_sus += 0.3
            if '-' in domain:  # Hyphens in domain
                domain_sus += 0.2
            suspiciousness["domain"] = min(1.0, domain_sus)
            
            # Check path
            path_sus = 0
            path = parsed.path.lower()
            for word in suspicious_words:
                if word in path:
                    path_sus += 0.3
            if len(path) > 30:  # Long path
                path_sus += 0.2
            if path.count('/') > 3:  # Deep path
                path_sus += 0.2
            suspiciousness["path"] = min(1.0, path_sus)
            
            # Check query
            query_sus = 0
            query = parsed.query.lower()
            for word in suspicious_words:
                if word in query:
                    query_sus += 0.3
            if len(query) > 50:  # Long query
                query_sus += 0.2
            if query.count('&') > 3:  # Many parameters
                query_sus += 0.2
            suspiciousness["query"] = min(1.0, query_sus)
            
            # Check scheme
            suspiciousness["scheme"] = 0.7 if parsed.scheme != "https" else 0.0
            
            # Check fragment
            fragment_sus = 0
            fragment = parsed.fragment.lower()
            for word in suspicious_words:
                if word in fragment:
                    fragment_sus += 0.3
            suspiciousness["fragment"] = min(1.0, fragment_sus)
            
            return components, suspiciousness
            
        except Exception as e:
            print(f"Error analyzing URL: {e}")
            return {}, {}

#===============================================================================
# 2. URL Dataset for Tokenization and Feature Extraction
#===============================================================================
class URLDataset(Dataset):
    """Dataset for processing URL data with tokenization and feature extraction"""
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        url = item['url']
        label = item['label']
        features = item['features']
        
        # Preprocess and tokenize URL
        processed_url = self._preprocess_url(url)
        encoded = self.tokenizer.encode_plus(
            processed_url,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Convert features dictionary to tensor
        feature_names = sorted(features.keys())
        feature_values = [features[name] for name in feature_names]
        feature_tensor = torch.tensor(feature_values, dtype=torch.float32)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'features': feature_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'url': url
        }
    
    def _preprocess_url(self, url):
        """Preprocess URL for tokenization"""
        for char in ['/', '.', '-', '=', '?', '&', '_', ':', '@']:
            url = url.replace(char, f' {char} ')
        url = url.replace('http', 'http ')
        url = url.replace('https', 'https ')
        url = url.replace('www', 'www ')
        return url

#===============================================================================
# 3. ModernBERT Prototypical Network for Few-Shot Learning with XAI support
#===============================================================================
class ModernBERTPrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot learning of URLs using ModernBERT.
    Embeds URLs and compares distances to class prototypes.
    Includes XAI hooks for interpretability.
    """
    def __init__(self, model_id='answerdotai/ModernBERT-base', feature_dim=13):
        super(ModernBERTPrototypicalNetwork, self).__init__()
        
        # URL encoder using ModernBERT
        self.encoder = AutoModel.from_pretrained(model_id)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        
        # Feature encoder for engineered URL features
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Projection for combining transformer and feature encodings
        self.projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size + 128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Distance function (negative Euclidean distance)
        self.distance_function = lambda x, y: -torch.sum(torch.pow(x.unsqueeze(1) - y, 2), dim=2)
        
        # For attention visualization (XAI)
        self.save_attention = False
        self.attention_weights = None
        self.token_embeddings = None
        self.tokens = None
    
    def encode(self, input_ids, attention_mask, features, save_attention=False):
        """Encode URLs into an embedding space"""
        # Record tokens for attention visualization if needed
        if save_attention:
            self.tokens = input_ids
            
        # Encode URL text with ModernBERT
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=save_attention
        )
        
        url_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Save attention weights and token embeddings for visualization
        if save_attention:
            self.attention_weights = outputs.attentions
            self.token_embeddings = outputs.last_hidden_state
        
        # Encode engineered features
        feature_embedding = self.feature_encoder(features)
        
        # Combine embeddings
        combined = torch.cat([url_embedding, feature_embedding], dim=1)
        embedding = self.projection(combined)
        
        # Normalize embedding (important for distance calculations)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def forward(self, support_input_ids, support_attention_mask, support_features, 
                support_labels, query_input_ids, query_attention_mask, query_features):
        """
        Forward pass for prototypical networks:
        1. Encode support and query examples
        2. Compute class prototypes from support set
        3. Calculate distances between query examples and class prototypes
        """
        # Encode support examples
        support_embeddings = self.encode(
            support_input_ids, support_attention_mask, support_features,
            save_attention=self.save_attention
        )
        
        # Encode query examples
        query_embeddings = self.encode(
            query_input_ids, query_attention_mask, query_features,
            save_attention=self.save_attention
        )
        
        # Get unique classes
        classes = torch.unique(support_labels)
        n_classes = len(classes)
        
        # Compute class prototypes
        prototypes = torch.zeros(n_classes, support_embeddings.size(1)).to(support_embeddings.device)
        for i, c in enumerate(classes):
            mask = support_labels == c
            if mask.sum() > 0:  # Ensure there's at least one example
                prototypes[i] = support_embeddings[mask].mean(0)
        
        # Calculate distances between query examples and prototypes
        logits = self.distance_function(query_embeddings, prototypes)
        
        return logits
        
    def get_prototypes(self, support_input_ids, support_attention_mask, support_features, support_labels):
        """Get prototypes for each class - used in XAI"""
        # Encode support examples
        support_embeddings = self.encode(
            support_input_ids, support_attention_mask, support_features
        )
        
        # Get unique classes
        classes = torch.unique(support_labels)
        n_classes = len(classes)
        
        # Compute class prototypes
        prototypes = {}
        for i, c in enumerate(classes):
            mask = support_labels == c
            if mask.sum() > 0:  # Ensure there's at least one example
                prototypes[c.item()] = support_embeddings[mask].mean(0)
        
        return prototypes
    
    def get_attention_visualization(self, tokenizer, input_ids, layer_idx=-1):
        """
        Generate attention visualization for a single example
        Returns: HTML with attention heatmap
        """
        if self.attention_weights is None:
            return "No attention weights available. Run with save_attention=True first."
        
        # Get attention from the specified layer (default: last layer)
        layer = layer_idx if layer_idx >= 0 else len(self.attention_weights) - 1
        if layer >= len(self.attention_weights):
            return f"Layer {layer} not available. Max layer is {len(self.attention_weights) - 1}"
        
        # Get attention weights for this layer
        attn = self.attention_weights[layer].mean(1)[0]  # Average across heads
        
        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Create attention visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(attn.cpu().detach().numpy(), cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Attention weight", rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        
        # Turn off ticks and labels if too many tokens
        if len(tokens) > 30:
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title("Attention Heatmap")
        fig.tight_layout()
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Return as HTML img tag
        return f'<img src="data:image/png;base64,{img_str}" alt="Attention visualization" />'
    
    def get_token_importance(self, tokenizer, input_ids, input_embeddings=None):
        """
        Calculate token importance based on their contribution to the final embedding
        Returns: dictionary mapping tokens to importance scores
        """
        if input_embeddings is None and self.token_embeddings is None:
            return "No token embeddings available. Run with save_attention=True first."
        
        embeddings = input_embeddings if input_embeddings is not None else self.token_embeddings
        
        # Use the first example
        single_embedding = embeddings[0]  # [seq_len, hidden_size]
        cls_embedding = single_embedding[0]  # [hidden_size]
        
        # Calculate cosine similarity between CLS and each token
        similarities = []
        for token_idx in range(single_embedding.size(0)):
            token_embedding = single_embedding[token_idx]
            sim = F.cosine_similarity(cls_embedding.unsqueeze(0), token_embedding.unsqueeze(0)).item()
            similarities.append(sim)
        
        # Convert to importance scores (normalize)
        min_sim, max_sim = min(similarities), max(similarities)
        importances = [(s - min_sim) / (max_sim - min_sim) for s in similarities]
        
        # Map to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        token_importance = {token: importance for token, importance in zip(tokens, importances)}
        
        return token_importance

#===============================================================================
# 4. ModernBERT MAML-inspired Phishing Detector with XAI
#===============================================================================
class ModernBERTMAMLPhishingDetector(nn.Module):
    """
    Model-Agnostic Meta-Learning for URL phishing detection using ModernBERT.
    Implements the MAML algorithm for quick adaptation to new phishing patterns.
    Includes XAI hooks for interpretability.
    """
    def __init__(self, model_id='answerdotai/ModernBERT-base', feature_dim=13):
        super(ModernBERTMAMLPhishingDetector, self).__init__()
        
        # Base encoder (shared)
        self.encoder = AutoModel.from_pretrained(model_id)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )
        
        # Combined representation
        self.representation = nn.Sequential(
            nn.Linear(self.encoder_hidden_size + 128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head (this will be adapted in inner loop)
        self.classifier = nn.Linear(512, 2)  # 2 classes: legitimate/phishing
        
        # For XAI
        self.save_attention = False
        self.attention_weights = None
        self.feature_importances = None
        self.representation_activations = None
    
    def forward(self, input_ids, attention_mask, features, save_activations=False):
        """Forward pass through the network with optional activation recording for XAI"""
        # Encode URL text with ModernBERT
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=self.save_attention
        )
        
        url_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Save attention weights if needed
        if self.save_attention:
            self.attention_weights = outputs.attentions
        
        # Encode engineered features
        feature_embedding = self.feature_encoder(features)
        
        # Store feature importances
        if save_activations:
            # Store feature activations by tracking gradient flow
            self.feature_importances = {}
            for i in range(features.size(1)):
                # Create a copy of features with the i-th feature set to 0
                modified_features = features.clone()
                original_value = modified_features[0, i].item()
                modified_features[0, i] = 0
                
                # Get the modified embedding
                modified_feature_embedding = self.feature_encoder(modified_features)
                
                # Calculate difference in embedding norm
                diff = torch.norm(feature_embedding - modified_feature_embedding).item()
                self.feature_importances[i] = diff
        
        # Combine embeddings
        combined = torch.cat([url_embedding, feature_embedding], dim=1)
        representation = self.representation(combined)
        
        # Store representation for interpretation
        if save_activations:
            self.representation_activations = representation.detach()
        
        # Classification
        logits = self.classifier(representation)
        
        return logits
    
    def clone_model(self):
        """Create a clone of the model for inner loop updates"""
        clone = ModernBERTMAMLPhishingDetector(
            model_id=self.encoder.config._name_or_path,
            feature_dim=next(self.feature_encoder.parameters()).size(1)
        )
        
        # Copy parameters
        clone.load_state_dict(self.state_dict())
        
        return clone
    
    def adapt(self, input_ids, attention_mask, features, labels, 
              num_adaptation_steps=3, step_size=0.1):
        """
        Adapt the model to a new task using the support set (inner loop of MAML)
        """
        # Clone model for adaptation
        adapted_model = self.clone_model()
        adapted_model.to(input_ids.device)
        
        # Set to training mode
        adapted_model.train()
        
        # Create optimizer for adaptation
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=step_size)
        
        # Inner loop adaptation
        for _ in range(num_adaptation_steps):
            # Forward pass
            logits = adapted_model(input_ids, attention_mask, features)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Return the adapted model
        return adapted_model
    
    def generate_counterfactuals(self, input_ids, attention_mask, features, tokenizer, num_features=5):
        """
        Generate counterfactual examples by tweaking features
        Returns: list of counterfactuals with their predicted class
        """
        self.eval()
        counterfactuals = []
        
        # Get original prediction
        with torch.no_grad():
            logits = self(input_ids, attention_mask, features)
            probs = F.softmax(logits, dim=1)
            orig_pred = torch.argmax(probs, dim=1).item()
            orig_prob = probs[0, orig_pred].item()
        
        # Try modifying each feature to flip the prediction
        for i in range(features.size(1)):
            # Create a modified feature vector
            modified_features = features.clone()
            
            # Try two modifications:
            # 1. Set to zero
            modified_features[0, i] = 0
            
            # Check prediction
            with torch.no_grad():
                logits = self(input_ids, attention_mask, modified_features)
                probs = F.softmax(logits, dim=1)
                new_pred = torch.argmax(probs, dim=1).item()
                new_prob = probs[0, new_pred].item()
            
            # If prediction flipped or probability changed significantly
            if new_pred != orig_pred or abs(new_prob - orig_prob) > 0.2:
                counterfactuals.append({
                    'feature_idx': i,
                    'modification': 'set to 0',
                    'original_value': features[0, i].item(),
                    'new_value': 0.0,
                    'original_class': orig_pred,
                    'new_class': new_pred,
                    'probability_change': new_prob - orig_prob
                })
            
            # 2. Invert (if binary) or set to high value (if continuous)
            if features[0, i].item() <= 1.0:  # Likely binary
                modified_features[0, i] = 1.0 - features[0, i].item()
            else:  # Continuous
                modified_features[0, i] = features[0, i].item() * 1.5  # Increase by 50%
            
            # Check prediction
            with torch.no_grad():
                logits = self(input_ids, attention_mask, modified_features)
                probs = F.softmax(logits, dim=1)
                new_pred = torch.argmax(probs, dim=1).item()
                new_prob = probs[0, new_pred].item()
            
            # If prediction flipped or probability changed significantly
            if new_pred != orig_pred or abs(new_prob - orig_prob) > 0.2:
                counterfactuals.append({
                    'feature_idx': i,
                    'modification': 'inverted/increased',
                    'original_value': features[0, i].item(),
                    'new_value': modified_features[0, i].item(),
                    'original_class': orig_pred,
                    'new_class': new_pred,
                    'probability_change': new_prob - orig_prob
                })
        
        # Sort by absolute probability change and return top N
        counterfactuals.sort(key=lambda x: abs(x['probability_change']), reverse=True)
        return counterfactuals[:num_features]
    
    def get_layer_activations(self):
        """Get intermediate layer activations for interpretation"""
        if self.representation_activations is None:
            return None
        
        return self.representation_activations

#===============================================================================
# 5. Hybrid Few-Shot Model for Phishing Detection with ModernBERT and XAI
#===============================================================================
class HybridFewShotModernBERTPhishingDetector(nn.Module):
    """
    Hybrid model combining Prototypical Network and MAML approaches with
    ModernBERT for few-shot phishing detection.
    Includes comprehensive XAI support.
    """
    def __init__(self, model_id='answerdotai/ModernBERT-base', feature_dim=13):
        super(HybridFewShotModernBERTPhishingDetector, self).__init__()
        
        # Prototypical Network component
        self.proto_net = ModernBERTPrototypicalNetwork(
            model_id=model_id, 
            feature_dim=feature_dim
        )
        
        # MAML component
        self.maml_net = ModernBERTMAMLPhishingDetector(
            model_id=model_id,
            feature_dim=feature_dim
        )
        
        # Combination layer (meta-learner)
        self.combination = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # For XAI
        self.xai_mode = False
        self.model_contributions = None
    
    def forward_proto(self, support_input_ids, support_attention_mask, 
                      support_features, support_labels, 
                      query_input_ids, query_attention_mask, query_features):
        """Forward pass through prototypical network"""
        return self.proto_net(
            support_input_ids, support_attention_mask, support_features,
            support_labels, query_input_ids, query_attention_mask, query_features
        )
    
    def forward_maml(self, support_input_ids, support_attention_mask, 
                    support_features, support_labels,
                    query_input_ids, query_attention_mask, query_features,
                    num_adaptation_steps=3):
        """Forward pass through MAML network with adaptation"""
        # Adapt model using support set
        adapted_model = self.maml_net.adapt(
            support_input_ids, support_attention_mask, support_features, support_labels,
            num_adaptation_steps=num_adaptation_steps
        )
        
        # Evaluate on query set using adapted model
        adapted_model.eval()
        with torch.no_grad():
            logits = adapted_model(query_input_ids, query_attention_mask, query_features)
        
        return logits
    
    def forward(self, support_input_ids, support_attention_mask, support_features, 
                support_labels, query_input_ids, query_attention_mask, query_features):
        """
        Forward pass through hybrid model, combining prototypical networks and MAML
        """
        # Enable XAI mode if requested
        if self.xai_mode:
            self.proto_net.save_attention = True
            self.maml_net.save_attention = True
        
        # Get predictions from both models
        proto_logits = self.forward_proto(
            support_input_ids, support_attention_mask, support_features,
            support_labels, query_input_ids, query_attention_mask, query_features
        )
        
        maml_logits = self.forward_maml(
            support_input_ids, support_attention_mask, support_features,
            support_labels, query_input_ids, query_attention_mask, query_features
        )
        
        # Calculate model contributions for XAI
        if self.xai_mode:
            # Use softmax to get probabilities
            proto_probs = F.softmax(proto_logits, dim=1)
            maml_probs = F.softmax(maml_logits, dim=1)
            
            # Get differences in confidences
            proto_conf = proto_probs.max(dim=1)[0]
            maml_conf = maml_probs.max(dim=1)[0]
            
            # Store contributions
            self.model_contributions = {
                'proto': proto_conf.detach().cpu().numpy(),
                'maml': maml_conf.detach().cpu().numpy()
            }
            
            # Disable XAI mode after use
            self.proto_net.save_attention = False
            self.maml_net.save_attention = False
            self.xai_mode = False
        
        # Combine predictions (weighted average)
        combined_logits = (0.6 * proto_logits + 0.4 * maml_logits)
        
        return combined_logits, proto_logits, maml_logits
    
    def explain(self, tokenizer, meta_dataset, support_input_ids, support_attention_mask,
                support_features, support_labels, query_input_ids, query_attention_mask, 
                query_features, query_urls, query_idx=0):
        """
        Generate comprehensive explanation for a specific query example
        
        Args:
            tokenizer: ModernBERT tokenizer
            meta_dataset: URLMetaDataset instance for feature descriptions
            support_*: Support set tensors
            query_*: Query set tensors
            query_urls: List of query URLs 
            query_idx: Index of query example to explain
            
        Returns:
            Dictionary with comprehensive explanation
        """
        # Enable XAI mode
        self.xai_mode = True
        self.proto_net.save_attention = True
        self.maml_net.save_attention = True
        
        # Forward pass to collect data for explanation
        with torch.no_grad():
            combined_logits, proto_logits, maml_logits = self(
                support_input_ids, support_attention_mask, support_features, support_labels,
                query_input_ids[query_idx:query_idx+1], 
                query_attention_mask[query_idx:query_idx+1],
                query_features[query_idx:query_idx+1]
            )
        
        # Get predicted class and confidence
        combined_probs = F.softmax(combined_logits, dim=1)
        proto_probs = F.softmax(proto_logits, dim=1)
        maml_probs = F.softmax(maml_logits, dim=1)
        
        pred_class = torch.argmax(combined_probs, dim=1).item()
        pred_confidence = combined_probs[0, pred_class].item()
        
        # Get class names
        class_names = ["legitimate", "phishing"]
        
        # Initialize explanation
        explanation = {
            "prediction": {
                "class": class_names[pred_class],
                "confidence": pred_confidence,
                "probabilities": {
                    "legitimate": combined_probs[0, 0].item(),
                    "phishing": combined_probs[0, 1].item()
                }
            },
            "model_contributions": {
                "prototypical": {
                    "class": class_names[torch.argmax(proto_probs, dim=1).item()],
                    "confidence": proto_probs.max().item(),
                    "weight": 0.6
                },
                "maml": {
                    "class": class_names[torch.argmax(maml_probs, dim=1).item()],
                    "confidence": maml_probs.max().item(),
                    "weight": 0.4
                }
            },
            "url_analysis": {},
            "feature_importance": {},
            "counterfactuals": {},
            "similar_examples": {}
        }
        
        # 1. URL analysis
        url = query_urls[query_idx]
        components, suspiciousness = meta_dataset.analyze_url_components(url)
        
        explanation["url_analysis"] = {
            "components": components,
            "suspiciousness_scores": suspiciousness,
        }
        
        # 2. Feature importance
        # Get feature descriptions
        feature_descriptions = meta_dataset.get_feature_descriptions()
        
        # Calculate feature importance using perturbation analysis
        feature_importance = {}
        for i in range(query_features.size(1)):
            # Create a modified feature tensor with this feature zeroed
            modified_features = query_features[query_idx:query_idx+1].clone()
            modified_features[0, i] = 0
            
            # Get prediction with this feature zeroed
            with torch.no_grad():
                mod_combined, _, _ = self(
                    support_input_ids, support_attention_mask, support_features, support_labels,
                    query_input_ids[query_idx:query_idx+1], 
                    query_attention_mask[query_idx:query_idx+1],
                    modified_features
                )
            
            # Calculate probability change
            mod_probs = F.softmax(mod_combined, dim=1)
            orig_prob = combined_probs[0, pred_class].item()
            mod_prob = mod_probs[0, pred_class].item()
            
            # Importance is the change in probability
            importance = orig_prob - mod_prob
            
            # Get feature name
            feature_names = sorted(feature_descriptions.keys())
            feature_name = feature_names[i]
            
            feature_importance[feature_name] = {
                "importance": importance,
                "description": feature_descriptions[feature_name],
                "value": query_features[query_idx, i].item()
            }
        
        # Sort by absolute importance
        explanation["feature_importance"] = {
            k: v for k, v in sorted(
                feature_importance.items(), 
                key=lambda item: abs(item[1]["importance"]), 
                reverse=True
            )
        }
        
        # 3. Generate counterfactuals
        counterfactuals = self.maml_net.generate_counterfactuals(
            query_input_ids[query_idx:query_idx+1],
            query_attention_mask[query_idx:query_idx+1],
            query_features[query_idx:query_idx+1],
            tokenizer
        )
        
        # Convert feature indices to names
        feature_names = sorted(feature_descriptions.keys())
        for cf in counterfactuals:
            cf['feature_name'] = feature_names[cf['feature_idx']]
            cf['feature_description'] = feature_descriptions[feature_names[cf['feature_idx']]]
            cf['class_from'] = class_names[cf['original_class']]
            cf['class_to'] = class_names[cf['new_class']]
        
        explanation["counterfactuals"] = counterfactuals
        
        # 4. Find similar examples from support set
        # Get embeddings
        query_embedding = self.proto_net.encode(
            query_input_ids[query_idx:query_idx+1],
            query_attention_mask[query_idx:query_idx+1],
            query_features[query_idx:query_idx+1]
        )
        
        support_embeddings = self.proto_net.encode(
            support_input_ids, support_attention_mask, support_features
        )
        
        # Calculate similarities
        similarities = F.cosine_similarity(
            query_embedding, support_embeddings
        ).cpu().numpy()
        
        # Get top 3 most similar examples
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        similar_examples = []
        for idx in top_indices:
            similar_examples.append({
                "url": support_input_ids[idx].item(),  # Need to convert back to URL
                "similarity": similarities[idx],
                "class": class_names[support_labels[idx].item()]
            })
        
        explanation["similar_examples"] = similar_examples
        
        # 5. Token importance from attention
        token_importance = self.proto_net.get_token_importance(
            tokenizer, 
            query_input_ids[query_idx:query_idx+1]
        )
        
        explanation["token_importance"] = token_importance
        
        # 6. Generate attention visualization
        attention_viz = self.proto_net.get_attention_visualization(
            tokenizer,
            query_input_ids[query_idx:query_idx+1]
        )
        
        explanation["attention_visualization"] = attention_viz
        
        return explanation
    
    def generate_explanation_html(self, explanation):
        """Generate a human-readable HTML explanation"""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1000px; margin: 0 auto; }
                .section { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .section-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
                .prediction { font-size: 24px; font-weight: bold; margin-bottom: 20px; }
                .prediction.phishing { color: #d9534f; }
                .prediction.legitimate { color: #5cb85c; }
                .feature-bar { height: 15px; margin-bottom: 2px; border-radius: 3px; }
                .feature-row { margin-bottom: 10px; }
                .feature-name { font-weight: bold; }
                .feature-value { float: right; }
                .feature-description { color: #666; font-size: 12px; }
                .positive { background-color: #5cb85c; }
                .negative { background-color: #d9534f; }
                .url-component { margin-bottom: 10px; }
                .url-component-name { font-weight: bold; }
                .highlight-low { background-color: rgba(92, 184, 92, 0.3); }
                .highlight-med { background-color: rgba(240, 173, 78, 0.3); }
                .highlight-high { background-color: rgba(217, 83, 79, 0.3); }
                .model-contribution { margin-bottom: 10px; }
                .counterfactual { margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
                .similar-example { margin-bottom: 10px; }
                .token { display: inline-block; padding: 2px 4px; margin: 2px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
        """
        
        # Prediction Section
        pred_class = explanation["prediction"]["class"]
        pred_confidence = explanation["prediction"]["confidence"]
        html += f"""
                <div class="section">
                    <div class="section-title">Prediction</div>
                    <div class="prediction {pred_class}">
                        {pred_class.upper()} ({pred_confidence:.2f})
                    </div>
                    <div>
                        <div>Legitimate: {explanation["prediction"]["probabilities"]["legitimate"]:.2f}</div>
                        <div>Phishing: {explanation["prediction"]["probabilities"]["phishing"]:.2f}</div>
                    </div>
                </div>
        """
        
        # URL Analysis Section
        html += """
                <div class="section">
                    <div class="section-title">URL Analysis</div>
        """
        
        components = explanation["url_analysis"]["components"]
        suspiciousness = explanation["url_analysis"]["suspiciousness_scores"]
        
        for component, value in components.items():
            if not value:  # Skip empty components
                continue
                
            score = suspiciousness.get(component, 0)
            highlight_class = "highlight-low"
            if score > 0.3:
                highlight_class = "highlight-med"
            if score > 0.7:
                highlight_class = "highlight-high"
                
            html += f"""
                    <div class="url-component">
                        <div class="url-component-name">{component}:</div>
                        <div class="{highlight_class}">{value}</div>
                    </div>
            """
            
        html += """
                </div>
        """
        
        # Feature Importance Section
        html += """
                <div class="section">
                    <div class="section-title">Feature Importance</div>
        """
        
        for feature, details in explanation["feature_importance"].items():
            importance = details["importance"]
            description = details["description"]
            value = details["value"]
            
            # Normalize for bar width (0-100%)
            width = min(100, abs(importance * 200))  # Scale for visibility
            bar_class = "positive" if importance > 0 else "negative"
            
            html += f"""
                    <div class="feature-row">
                        <div class="feature-name">{feature} <span class="feature-value">{value:.2f}</span></div>
                        <div class="feature-bar {bar_class}" style="width: {width}%;"></div>
                        <div class="feature-description">{description}</div>
                    </div>
            """
            
        html += """
                </div>
        """
        
        # Model Contributions Section
        html += """
                <div class="section">
                    <div class="section-title">Model Contributions</div>
        """
        
        proto = explanation["model_contributions"]["prototypical"]
        maml = explanation["model_contributions"]["maml"]
        
        html += f"""
                    <div class="model-contribution">
                        <div>Prototypical Network (weight: {proto["weight"]}):</div>
                        <div>Prediction: {proto["class"]} (confidence: {proto["confidence"]:.2f})</div>
                    </div>
                    <div class="model-contribution">
                        <div>MAML Network (weight: {maml["weight"]}):</div>
                        <div>Prediction: {maml["class"]} (confidence: {maml["confidence"]:.2f})</div>
                    </div>
        """
        
        html += """
                </div>
        """
        
        # Counterfactuals Section
        html += """
                <div class="section">
                    <div class="section-title">What Would Change the Prediction?</div>
        """
        
        for cf in explanation["counterfactuals"]:
            html += f"""
                    <div class="counterfactual">
                        <div>If <strong>{cf["feature_name"]}</strong> ({cf["feature_description"]}) was changed from {cf["original_value"]:.2f} to {cf["new_value"]:.2f}:</div>
                        <div>The prediction would change from <strong>{cf["class_from"]}</strong> to <strong>{cf["class_to"]}</strong></div>
                        <div>Confidence change: {cf["probability_change"]:.2f}</div>
                    </div>
            """
            
        html += """
                </div>
        """
        
        # Token Importance Section
        if "token_importance" in explanation and isinstance(explanation["token_importance"], dict):
            html += """
                    <div class="section">
                        <div class="section-title">Important URL Components</div>
                        <div>
            """
            
            for token, importance in explanation["token_importance"].items():
                # Skip special tokens and low importance tokens
                if token.startswith('[') or token.startswith('<') or importance < 0.1:
                    continue
                    
                # Scale for color intensity (0.1-1.0)
                intensity = min(1.0, max(0.1, importance))
                red = int(255 * intensity)
                green = int(255 * (1 - intensity))
                
                html += f"""
                        <span class="token" style="background-color: rgba({red}, {green}, 0, 0.3);">{token}</span>
                """
                
            html += """
                        </div>
                    </div>
            """
        
        # Attention Visualization Section
        if "attention_visualization" in explanation and explanation["attention_visualization"].startswith('<img'):
            html += """
                    <div class="section">
                        <div class="section-title">Attention Visualization</div>
            """
            
            html += explanation["attention_visualization"]
            
            html += """
                    </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html

#===============================================================================
# 6. Training Functions for Few-Shot Learning
#===============================================================================
def train_epoch(model, meta_dataset, tokenizer, optimizer, device, 
                num_episodes=100, inner_steps=3):
    """Train model for one epoch using episodic few-shot learning"""
    model.train()
    total_loss = 0
    total_acc = 0
    
    progress_bar = tqdm(range(num_episodes), desc="Training Episodes")
    
    for _ in progress_bar:
        # Sample a few-shot episode
        support_set, query_set, task_info = meta_dataset.sample_episode()
        
        # Create datasets
        support_dataset = URLDataset(support_set, tokenizer)
        query_dataset = URLDataset(query_set, tokenizer)
        
        # Create dataloaders
        support_loader = DataLoader(support_dataset, batch_size=len(support_dataset), shuffle=True)
        query_loader = DataLoader(query_dataset, batch_size=len(query_dataset), shuffle=True)
        
        # Get all support data in one batch
        for support_batch in support_loader:
            support_input_ids = support_batch['input_ids'].to(device)
            support_attention_mask = support_batch['attention_mask'].to(device)
            support_features = support_batch['features'].to(device)
            support_labels = support_batch['label'].to(device)
            break
        
        # Get all query data in one batch
        for query_batch in query_loader:
            query_input_ids = query_batch['input_ids'].to(device)
            query_attention_mask = query_batch['attention_mask'].to(device)
            query_features = query_batch['features'].to(device)
            query_labels = query_batch['label'].to(device)
            break
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        combined_logits, proto_logits, maml_logits = model(
            support_input_ids, support_attention_mask, support_features, support_labels,
            query_input_ids, query_attention_mask, query_features
        )
        
        # Calculate losses
        combined_loss = F.cross_entropy(combined_logits, query_labels)
        proto_loss = F.cross_entropy(proto_logits, query_labels)
        maml_loss = F.cross_entropy(maml_logits, query_labels)
        
        # Total loss (with emphasis on combined model)
        loss = 0.6 * combined_loss + 0.2 * proto_loss + 0.2 * maml_loss
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Calculate accuracy
        _, preds = torch.max(combined_logits, dim=1)
        acc = (preds == query_labels).float().mean().item()
        
        # Update metrics
        total_loss += loss.item()
        total_acc += acc
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{acc:.4f}"
        })
    
    # Calculate epoch metrics
    avg_loss = total_loss / num_episodes
    avg_acc = total_acc / num_episodes
    
    return avg_loss, avg_acc

def evaluate(model, meta_dataset, tokenizer, device, num_episodes=30):
    """Evaluate model on few-shot episodes"""
    model.eval()
    total_loss = 0
    total_acc = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_episodes), desc="Evaluation Episodes"):
            # Sample a few-shot episode
            support_set, query_set, _ = meta_dataset.sample_episode()
            
            # Create datasets
            support_dataset = URLDataset(support_set, tokenizer)
            query_dataset = URLDataset(query_set, tokenizer)
            
            # Create dataloaders
            support_loader = DataLoader(support_dataset, batch_size=len(support_dataset))
            query_loader = DataLoader(query_dataset, batch_size=len(query_dataset))
            
            # Get all support data in one batch
            for support_batch in support_loader:
                support_input_ids = support_batch['input_ids'].to(device)
                support_attention_mask = support_batch['attention_mask'].to(device)
                support_features = support_batch['features'].to(device)
                support_labels = support_batch['label'].to(device)
                break
            
            # Get all query data in one batch
            for query_batch in query_loader:
                query_input_ids = query_batch['input_ids'].to(device)
                query_attention_mask = query_batch['attention_mask'].to(device)
                query_features = query_batch['features'].to(device)
                query_labels = query_batch['label'].to(device)
                break
            
            # Forward pass
            combined_logits, _, _ = model(
                support_input_ids, support_attention_mask, support_features, support_labels,
                query_input_ids, query_attention_mask, query_features
            )
            
            # Calculate loss
            loss = F.cross_entropy(combined_logits, query_labels)
            
            # Calculate predictions
            _, preds = torch.max(combined_logits, dim=1)
            
            # Update metrics
            total_loss += loss.item()
            acc = (preds == query_labels).float().mean().item()
            total_acc += acc
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / num_episodes
    avg_acc = total_acc / num_episodes
    
    # Additional metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, avg_acc, precision, recall, f1, all_preds, all_labels

#===============================================================================
# 7. Main Training Loop
#===============================================================================
def main():
    print("Initializing Few-Shot Phishing Detection with ModernBERT and XAI...")
    
    # Parameters
    n_way = 2  # Binary classification: phishing vs legitimate
    k_shot = 5  # 5 examples per class in support set
    query_size = 10  # 10 examples per class in query set
    num_episodes = 200  # Number of episodes per epoch
    num_epochs = 10
    
    # Initialize meta-dataset
    url_source = "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/refs/heads/master/phishing-links-ACTIVE.txt"
    meta_dataset = URLMetaDataset(
        url_source=url_source,
        n_way=n_way,
        k_shot=k_shot,
        query_size=query_size
    )
    
    # Initialize tokenizer - using ModernBERT
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Initialize model with ModernBERT
    model = HybridFewShotModernBERTPhishingDetector(
        model_id=model_id,
        feature_dim=13  # Number of engineered features
    )
    model = model.to(device)
    
    # Fixed: Properly separate parameter groups to avoid duplication
    # Create separate parameter groups for different learning rates
    encoder_params = set()
    # First collect all encoder parameters
    for name, param in model.proto_net.encoder.named_parameters():
        encoder_params.add(param)
    for name, param in model.maml_net.encoder.named_parameters():
        encoder_params.add(param)
        
    # Now create proper parameter groups
    other_params = [p for p in model.parameters() if p not in encoder_params]
    
    # Create optimizer with correct parameter groups
    optimizer = torch.optim.AdamW([
        {"params": list(encoder_params), "lr": 1e-5},  # Lower learning rate for ModernBERT parameters
        {"params": other_params, "lr": 3e-5}  # Higher learning rate for other parameters
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, meta_dataset, tokenizer, optimizer, device,
            num_episodes=num_episodes
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        val_loss, val_acc, precision, recall, f1, _, _ = evaluate(
            model, meta_dataset, tokenizer, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fewshot_modernbert_phishing_model.pt')
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
    
    print("Training complete!")
    
    # Final evaluation on new episodes
    model.load_state_dict(torch.load('best_fewshot_modernbert_phishing_model.pt'))
    print("\nPerforming final evaluation...")
    
    final_loss, final_acc, final_precision, final_recall, final_f1, y_pred, y_true = evaluate(
        model, meta_dataset, tokenizer, device, num_episodes=50
    )
    
    print("\nFinal Test Metrics:")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Few-Shot Phishing Detection with ModernBERT')
    plt.savefig('fewshot_modernbert_confusion_matrix.png')
    plt.close()
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fewshot_modernbert_training_curves.png')
    plt.close()
    
    print("Evaluation complete! Saved visualizations.")

#===============================================================================
# 8. XAI Testing and Demo
#===============================================================================
def test_with_xai(model_path, tokenizer, meta_dataset, device, urls, is_phishing=True):
    """
    Test model on a set of URLs with detailed XAI explanations
    
    Args:
        model_path: Path to saved model
        tokenizer: ModernBERT tokenizer
        meta_dataset: URLMetaDataset instance
        device: Device to run on
        urls: List of URLs to test
        is_phishing: Whether the URLs are phishing
        
    Returns:
        Dict with results and explanations
    """
    # Load model
    model = HybridFewShotModernBERTPhishingDetector(
        model_id="answerdotai/ModernBERT-base",
        feature_dim=13
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Prepare the few-shot examples
    support_set = []
    for url in urls[:5]:  # Use first 5 as support
        features = meta_dataset.extract_features(url)
        support_set.append({
            'url': url,
            'label': 1 if is_phishing else 0,
            'features': features
        })
    
    # Add some legitimate examples if testing phishing (or vice versa)
    if is_phishing:
        # Add some legitimate examples
        for _ in range(5):
            url = f"https://{random.choice(meta_dataset.legitimate_domains)}/page"
            features = meta_dataset.extract_features(url)
            support_set.append({
                'url': url,
                'label': 0,  # Legitimate
                'features': features
            })
    else:
        # Add some phishing examples
        phishing_urls = meta_dataset._generate_synthetic_phishing_urls(5)
        for url in phishing_urls:
            features = meta_dataset.extract_features(url)
            support_set.append({
                'url': url,
                'label': 1,  # Phishing
                'features': features
            })
    
    # Prepare query set (remaining URLs)
    query_set = []
    for url in urls[5:]:  # Use remainder as query
        features = meta_dataset.extract_features(url)
        query_set.append({
            'url': url,
            'label': 1 if is_phishing else 0,
            'features': features
        })
    
    # Create datasets
    support_dataset = URLDataset(support_set, tokenizer)
    query_dataset = URLDataset(query_set, tokenizer)
    
    # Create dataloaders
    support_loader = DataLoader(support_dataset, batch_size=len(support_dataset))
    query_loader = DataLoader(query_dataset, batch_size=len(query_dataset))
    
    # Get all support data
    for support_batch in support_loader:
        support_input_ids = support_batch['input_ids'].to(device)
        support_attention_mask = support_batch['attention_mask'].to(device)
        support_features = support_batch['features'].to(device)
        support_labels = support_batch['label'].to(device)
        break
    
    # Get all query data
    for query_batch in query_loader:
        query_input_ids = query_batch['input_ids'].to(device)
        query_attention_mask = query_batch['attention_mask'].to(device)
        query_features = query_batch['features'].to(device)
        query_labels = query_batch['label'].to(device)
        query_urls = query_batch['url']
        break
    
    # Forward pass
    with torch.no_grad():
        combined_logits, proto_logits, maml_logits = model(
            support_input_ids, support_attention_mask, support_features, support_labels,
            query_input_ids, query_attention_mask, query_features
        )
    
    # Get predictions
    _, preds = torch.max(combined_logits, dim=1)
    
    # Calculate accuracy
    acc = (preds == query_labels).float().mean().item()
    
    # Detailed XAI analysis
    results = {
        "accuracy": acc,
        "predictions": [],
        "explanations": []
    }
    
    # Generate explanations for each query URL
    for i in range(len(query_urls)):
        # Generate explanation
        explanation = model.explain(
            tokenizer,
            meta_dataset,
            support_input_ids,
            support_attention_mask,
            support_features,
            support_labels,
            query_input_ids,
            query_attention_mask,
            query_features,
            query_urls,
            i
        )
        
        # Generate HTML explanation
        html_explanation = model.generate_explanation_html(explanation)
        
        # Add to results
        results["explanations"].append({
            "raw": explanation,
            "html": html_explanation
        })
        
        # Add prediction info
        pred_label = "phishing" if preds[i].item() == 1 else "legitimate"
        true_label = "phishing" if query_labels[i].item() == 1 else "legitimate"
        
        results["predictions"].append({
            "url": query_urls[i],
            "predicted": pred_label,
            "true": true_label,
            "correct": pred_label == true_label
        })
    
    return results

def generate_xai_report(results, output_dir="xai_reports"):
    """Generate XAI reports from results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create index.html
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Few-Shot Phishing Detection XAI Reports</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
            .url-list { margin-top: 20px; }
            .url-item { margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .url-item a { text-decoration: none; color: #0066cc; }
            .correct { background-color: rgba(92, 184, 92, 0.2); }
            .incorrect { background-color: rgba(217, 83, 79, 0.2); }
            .summary { margin-bottom: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Few-Shot Phishing Detection XAI Reports</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Accuracy: {accuracy:.2f}</p>
                <p>Total URLs analyzed: {total}</p>
            </div>
            
            <div class="url-list">
                <h2>URL Reports</h2>
    """.format(
        accuracy=results["accuracy"],
        total=len(results["predictions"])
    )
    
    # Process each URL
    for i, pred in enumerate(results["predictions"]):
        url = pred["url"]
        predicted = pred["predicted"]
        true = pred["true"]
        correct = pred["correct"]
        
        filename = f"report_{i}.html"
        
        # Write individual report
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(results["explanations"][i]["html"])
        
        # Add to index
        status_class = "correct" if correct else "incorrect"
        index_html += f"""
                <div class="url-item {status_class}">
                    <div><strong>URL:</strong> {url}</div>
                    <div><strong>Predicted:</strong> {predicted.upper()}</div>
                    <div><strong>True:</strong> {true.upper()}</div>
                    <div><a href="{filename}" target="_blank">View XAI Report</a></div>
                </div>
        """
    
    index_html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write index.html
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write(index_html)
    
    print(f"XAI reports generated in {output_dir}/")
    return os.path.join(output_dir, "index.html")

# Run the main function if script is executed directly
if __name__ == "__main__":
    # Train the model
    main()
    
    # Example of testing with XAI for explainability
    print("\nTesting model with XAI explanations...")
    
    # Example new phishing pattern (finance-related phishing)
    new_phishing_urls = [
        "http://banking-secure-verify.com/account/login",
        "http://secure-banking-login.net/verify/details",
        "http://bank-account-verification.org/secure/login",
        "http://online-banking-secure.com/verify/account",
        "http://secure-bank-verification.net/login",
        "http://banking-secure-login.com/verify/identity",
        "http://secure-account-verify.org/banking/login",
        "http://bank-verification-secure.net/login/account",
        "http://login-secure-banking.com/verify/details",
        "http://verify-account-secure.org/banking/login"
    ]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create meta dataset
    meta_dataset = URLMetaDataset(url_source="")
    
    # Test with XAI
    xai_results = test_with_xai(
        model_path='best_fewshot_modernbert_phishing_model.pt',
        tokenizer=tokenizer,
        meta_dataset=meta_dataset,
        device=device,
        urls=new_phishing_urls,
        is_phishing=True
    )
    
    # Generate XAI reports
    report_index = generate_xai_report(xai_results)
    
    print(f"XAI testing complete! Reports available at {report_index}")
