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
        self.query_size = query_size  # Number of examples per class for query set
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
# 3. Prototypical Network for Few-Shot Learning
#===============================================================================
class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot learning of URLs.
    Embeds URLs and compares distances to class prototypes.
    """
    def __init__(self, model_id='sentence-transformers/all-MiniLM-L6-v2', feature_dim=13):
        super(PrototypicalNetwork, self).__init__()
        
        # URL encoder using transformer
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
    
    def encode(self, input_ids, attention_mask, features):
        """Encode URLs into an embedding space"""
        # Encode URL text
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        url_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
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
            support_input_ids, support_attention_mask, support_features
        )
        
        # Encode query examples
        query_embeddings = self.encode(
            query_input_ids, query_attention_mask, query_features
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

#===============================================================================
# 4. MAML-inspired URL Phishing Detector
#===============================================================================
class MAMLPhishingDetector(nn.Module):
    """
    Model-Agnostic Meta-Learning for URL phishing detection.
    Implements the MAML algorithm for quick adaptation to new phishing patterns.
    """
    def __init__(self, model_id='sentence-transformers/all-MiniLM-L6-v2', feature_dim=13):
        super(MAMLPhishingDetector, self).__init__()
        
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
    
    def forward(self, input_ids, attention_mask, features):
        """Forward pass through the network"""
        # Encode URL text
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        url_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Encode engineered features
        feature_embedding = self.feature_encoder(features)
        
        # Combine embeddings
        combined = torch.cat([url_embedding, feature_embedding], dim=1)
        representation = self.representation(combined)
        
        # Classification
        logits = self.classifier(representation)
        
        return logits
    
    def clone_model(self):
        """Create a clone of the model for inner loop updates"""
        clone = MAMLPhishingDetector(
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

#===============================================================================
# 5. Hybrid Few-Shot Model for Phishing Detection
#===============================================================================
class HybridFewShotPhishingDetector(nn.Module):
    """
    Hybrid model combining Prototypical Network and MAML approaches for
    few-shot phishing detection.
    """
    def __init__(self, proto_model_id='sentence-transformers/all-MiniLM-L6-v2', 
                 maml_model_id='sentence-transformers/all-MiniLM-L6-v2', 
                 feature_dim=13):
        super(HybridFewShotPhishingDetector, self).__init__()
        
        # Prototypical Network component
        self.proto_net = PrototypicalNetwork(
            model_id=proto_model_id, 
            feature_dim=feature_dim
        )
        
        # MAML component
        self.maml_net = MAMLPhishingDetector(
            model_id=maml_model_id,
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
        # Get predictions from both models
        proto_logits = self.forward_proto(
            support_input_ids, support_attention_mask, support_features,
            support_labels, query_input_ids, query_attention_mask, query_features
        )
        
        maml_logits = self.forward_maml(
            support_input_ids, support_attention_mask, support_features,
            support_labels, query_input_ids, query_attention_mask, query_features
        )
        
        # Combine predictions (simple averaging for now)
        # More sophisticated combination could be learned
        combined_logits = (proto_logits + maml_logits) / 2
        
        return combined_logits, proto_logits, maml_logits

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
    print("Initializing Few-Shot Phishing Detection...")
    
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
    
    # Initialize tokenizer
    model_id = "sentence-transformers/all-MiniLM-L6-v2"  # Efficient and good for text similarity
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Initialize model
    model = HybridFewShotPhishingDetector(
        proto_model_id=model_id,
        maml_model_id=model_id,
        feature_dim=13  # Number of engineered features
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
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
            torch.save(model.state_dict(), 'best_fewshot_phishing_model.pt')
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
    
    print("Training complete!")
    
    # Final evaluation on new episodes
    model.load_state_dict(torch.load('best_fewshot_phishing_model.pt'))
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
    plt.title('Confusion Matrix for Few-Shot Phishing Detection')
    plt.savefig('fewshot_confusion_matrix.png')
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
    plt.savefig('fewshot_training_curves.png')
    plt.close()
    
    print("Evaluation complete! Saved visualizations.")

#===============================================================================
# 8. Example Usage
#===============================================================================
def test_on_new_phishing_pattern(model_path, tokenizer, device, urls, is_phishing=True):
    """
    Test model on a new phishing pattern with very few examples
    """
    # Load model
    model = HybridFewShotPhishingDetector(
        proto_model_id="sentence-transformers/all-MiniLM-L6-v2",
        maml_model_id="sentence-transformers/all-MiniLM-L6-v2",
        feature_dim=13
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Create meta-dataset just for feature extraction
    meta_dataset = URLMetaDataset(url_source="")
    
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
        urls = query_batch['url']
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
    
    # Display results
    print(f"\nResults for new {'phishing' if is_phishing else 'legitimate'} pattern:")
    print(f"Accuracy: {acc:.4f}")
    
    # Show individual predictions
    print("\nIndividual predictions:")
    for i, (url, pred) in enumerate(zip(urls, preds)):
        pred_label = "Phishing" if pred.item() == 1 else "Legitimate"
        true_label = "Phishing" if query_labels[i].item() == 1 else "Legitimate"
        print(f"URL: {url}")
        print(f"Prediction: {pred_label}, True: {true_label}")
        print("---")

# Run the main function
if __name__ == "__main__":
    main()
    
    # Example of testing on a new phishing pattern with very few examples
    print("\nTesting model on a new phishing pattern with few examples...")
    
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
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Test on new pattern
    test_on_new_phishing_pattern(
        model_path='best_fewshot_phishing_model.pt',
        tokenizer=tokenizer,
        device=device,
        urls=new_phishing_urls,
        is_phishing=True
    )
