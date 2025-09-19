"""
URL feature extraction module for phishing detection
Customized to match your specific dataset structure with 56 columns
"""

import re
import math
import logging
from urllib.parse import urlparse
import tldextract
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_url(url):
    """
    Analyze a URL and extract features matching the 56-column dataset structure
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        dict: Dictionary with extracted features and risk analysis
    """
    try:
        # Make sure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        # Parse the URL
        parsed_url = urlparse(url)
        domain_info = tldextract.extract(url)
        
        # Extract domain
        domain = domain_info.domain + '.' + domain_info.suffix if domain_info.suffix else domain_info.domain
        
        # Initialize features dictionary
        features = {}
        risk_factors = []
        
        # URL and Domain features
        features['URL'] = url
        features['URLLength'] = len(url)
        if features['URLLength'] > 100:
            risk_factors.append(f"Unusually long URL ({features['URLLength']} characters)")
            
        features['Domain'] = domain
        features['DomainLength'] = len(domain)
        
        # Check if domain is an IP address
        features['IsDomainIP'] = 1 if is_ip_address(parsed_url.netloc) else 0
        if features['IsDomainIP'] == 1:
            risk_factors.append("URL uses IP address instead of domain name")
            
        # TLD features
        features['TLD'] = domain_info.suffix if domain_info.suffix else ""
        features['TLDLength'] = len(features['TLD'])
        
        # Subdomain features
        features['NoOfSubDomain'] = count_subdomains(domain_info)
        if features['NoOfSubDomain'] > 3:
            risk_factors.append(f"Excessive subdomains ({features['NoOfSubDomain']})")
            
        # URL metrics
        features['URLSimilarityIndex'] = calculate_url_similarity(url)
        features['CharContinuationRate'] = calculate_char_continuation(url)
        
        # Probability metrics
        features['TLDLegitimateProb'] = calculate_tld_legitimacy(features['TLD'])
        features['URLCharProb'] = calculate_char_probability(url)
        
        # Obfuscation detection
        obfuscation_info = detect_obfuscation(url)
        features['HasObfuscation'] = 1 if obfuscation_info['has_obfuscation'] else 0
        features['NoOfObfuscatedChar'] = obfuscation_info['obfuscated_count']
        features['ObfuscationRatio'] = obfuscation_info['obfuscation_ratio']
        if features['HasObfuscation'] == 1:
            risk_factors.append(f"URL contains obfuscation techniques")
            
        # Character analysis
        char_analysis = analyze_characters(url)
        features['NoOfLettersInURL'] = char_analysis['letter_count']
        features['LetterRatioInURL'] = char_analysis['letter_ratio']
        features['NoOfDegitsInURL'] = char_analysis['digit_count'] 
        features['DegitRatioInURL'] = char_analysis['digit_ratio']
        
        # Special characters
        features['NoOfEqualsInURL'] = url.count('=')
        features['NoOfQMarkInURL'] = url.count('?')
        features['NoOfAmpersandInURL'] = url.count('&')
        features['NoOfOtherSpecialCharsInURL'] = count_special_chars(url) - features['NoOfEqualsInURL'] - features['NoOfQMarkInURL'] - features['NoOfAmpersandInURL']
        features['SpacialCharRatioInURL'] = features['NoOfOtherSpecialCharsInURL'] / features['URLLength'] if features['URLLength'] > 0 else 0
        
        if features['SpacialCharRatioInURL'] > 0.3:
            risk_factors.append(f"High special character ratio ({features['SpacialCharRatioInURL']:.2f})")
            
        # Security features
        features['IsHTTPS'] = 1 if parsed_url.scheme == 'https' else 0
        if features['IsHTTPS'] == 0:
            risk_factors.append("Site does not use HTTPS encryption")
            
        # Title features (would need actual page content)
        features['HasTitle'] = 0
        features['Title'] = ""
        features['DomainTitleMatchScore'] = 0.0
        features['URLTitleMatchScore'] = 0.0
            
        # Content-based features that require page analysis
        # Set default values since we can't easily fetch and analyze page content here
        content_features = {
            'LineOfCode': 0,
            'LargestLineLength': 0,
            'HasFavicon': 0,
            'Robots': 0,
            'IsResponsive': 0,
            'NoOfURLRedirect': 0,
            'NoOfSelfRedirect': 0,
            'HasDescription': 0,
            'NoOfPopup': 0,
            'NoOfiFrame': 0,
            'HasExternalFormSubmit': 0,
            'HasSocialNet': 0,
            'HasSubmitButton': 0,
            'HasHiddenFields': 0,
            'HasPasswordField': 0,
            'Bank': 0,
            'Pay': 0,
            'Crypto': 0,
            'HasCopyrightInfo': 0,
            'NoOfImage': 0,
            'NoOfCSS': 0,
            'NoOfJS': 0,
            'NoOfSelfRef': 0,
            'NoOfEmptyRef': 0,
            'NoOfExternalRef': 0
        }
        
        # Add content features to our feature set
        features.update(content_features)
        
        # For FILENAME column (not used in prediction)
        features['FILENAME'] = ""
        
        # Check for finance-related keywords
        if contains_finance_keywords(url):
            features['Bank'] = 1
            features['Pay'] = 1
            risk_factors.append("URL contains finance-related keywords")
        
        # Determine risk level
        risk_level = 'Low'
        if len(risk_factors) >= 3:
            risk_level = 'High'
        elif len(risk_factors) >= 1:
            risk_level = 'Medium'
            
        return {
            'features': features,
            'analysis': {
                'risk_factors': risk_factors,
                'risk_level': risk_level
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing URL {url}: {e}")
        return {
            'features': {},
            'analysis': {
                'risk_factors': ["Error analyzing URL: " + str(e)],
                'risk_level': 'Unknown'
            }
        }

def is_ip_address(domain):
    """Check if domain is an IP address"""
    # IPv4
    ipv4_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
    # IPv6
    ipv6_pattern = r'^[0-9a-fA-F:]+$'
    return bool(re.match(ipv4_pattern, domain) or re.match(ipv6_pattern, domain))

def count_subdomains(domain_info):
    """Count the number of subdomains"""
    if not domain_info.subdomain:
        return 0
    return domain_info.subdomain.count('.') + 1

def analyze_characters(url):
    """Analyze character types in the URL"""
    letter_count = sum(c.isalpha() for c in url)
    digit_count = sum(c.isdigit() for c in url)
    
    letter_ratio = letter_count / len(url) if len(url) > 0 else 0
    digit_ratio = digit_count / len(url) if len(url) > 0 else 0
    
    return {
        'letter_count': letter_count,
        'digit_count': digit_count,
        'letter_ratio': letter_ratio,
        'digit_ratio': digit_ratio
    }

def count_special_chars(url):
    """Count special characters in URL"""
    return sum(not c.isalnum() for c in url)

def detect_obfuscation(url):
    """
    Detect URL obfuscation techniques
    Returns a dict with obfuscation metrics
    """
    # Patterns that might indicate obfuscation
    obfuscation_patterns = [
        r'%[0-9a-fA-F]{2}',  # URL encoding
        r'0x[0-9a-fA-F]+',  # Hexadecimal
        r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?!\d)',  # IP address
        r'\d{10,}',  # Long numbers
        r'(?:\.){2,}',  # Multiple consecutive dots
        r'\S+@\S+\.\S+',  # Email-like patterns
        r'bit\.ly|goo\.gl|t\.co|tinyurl'  # URL shorteners
    ]
    
    obfuscated_count = 0
    for pattern in obfuscation_patterns:
        matches = re.findall(pattern, url.lower())
        for match in matches:
            obfuscated_count += len(match)
    
    has_obfuscation = obfuscated_count > 0
    obfuscation_ratio = obfuscated_count / len(url) if len(url) > 0 else 0
    
    return {
        'has_obfuscation': has_obfuscation,
        'obfuscated_count': obfuscated_count,
        'obfuscation_ratio': obfuscation_ratio
    }

def calculate_url_similarity(url):
    """
    Calculate a similarity index based on character distribution
    Higher values indicate more repetitive patterns
    """
    if len(url) < 2:
        return 0
    
    # Count character frequencies
    char_counts = Counter(url.lower())
    
    # Calculate entropy (lower entropy = higher similarity)
    entropy = 0
    for char, count in char_counts.items():
        prob = count / len(url)
        entropy -= prob * math.log2(prob) if prob > 0 else 0
    
    # Convert entropy to similarity score (0-100)
    # Lower entropy means more similar/repetitive
    max_entropy = math.log2(min(len(char_counts), len(url)))
    similarity = 100 * (1 - (entropy / max_entropy)) if max_entropy > 0 else 0
    
    return int(similarity)

def calculate_char_continuation(url):
    """Calculate the rate of consecutive repeating characters"""
    if len(url) <= 1:
        return 0.0
    
    continuation_count = 0
    for i in range(1, len(url)):
        if url[i] == url[i-1]:
            continuation_count += 1
    
    return continuation_count / (len(url) - 1)

def calculate_tld_legitimacy(tld):
    """
    Calculate legitimacy probability of TLD
    Returns a value between 0 and 1
    """
    # Common legitimate TLDs have higher scores
    common_tlds = {
        'com': 0.95, 'org': 0.95, 'net': 0.95, 'edu': 0.98, 
        'gov': 0.99, 'mil': 0.99, 'io': 0.9, 'co': 0.85
    }
    
    # Country TLDs
    country_tlds = [
        'us', 'uk', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'ru', 'br', 
        'in', 'it', 'es', 'nl', 'ch', 'no', 'se', 'dk', 'fi'
    ]
    
    if not tld:
        return 0.2
    
    if tld in common_tlds:
        return common_tlds[tld]
    elif tld in country_tlds:
        return 0.9
    elif len(tld) <= 3:
        return 0.7
    
    # Suspicious long or uncommon TLDs
    return 0.5 if len(tld) <= 5 else 0.3

def calculate_char_probability(url):
    """
    Calculate character distribution probability
    Returns a value between 0 and 1
    """
    # Analyze character types
    char_analysis = analyze_characters(url)
    
    # Suspicious features
    if char_analysis['digit_ratio'] > 0.5:
        return 0.3  # Too many digits
    
    special_ratio = count_special_chars(url) / len(url) if len(url) > 0 else 0
    if special_ratio > 0.25:
        return 0.4  # Too many special characters
    
    # Length-based scoring
    if len(url) > 100:
        return 0.5  # Very long URLs are suspicious
    
    # Character repetition
    char_counts = Counter(url.lower())
    max_repeat = max(char_counts.values()) if char_counts else 0
    if max_repeat > len(url) * 0.25:
        return 0.6  # Suspicious repetition
    
    # Looks normal
    return 0.8

def contains_finance_keywords(url):
    """Check if URL contains finance-related keywords"""
    keywords = [
        'bank', 'account', 'secure', 'login', 'signin', 'verify', 
        'wallet', 'pay', 'card', 'credit', 'debit', 'transaction',
        'finance', 'banking', 'crypto', 'bitcoin', 'paypal'
    ]
    
    url_lower = url.lower()
    for keyword in keywords:
        if keyword in url_lower:
            return True
    
    return False

# Self-test code
if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "https://secure-login-paypal.verify-account.com/secure",
        "http://192.168.1.1/admin",
        "https://bit.ly/3xR2SEp"
    ]
    
    for url in test_urls:
        result = analyze_url(url)
        print(f"\nURL: {url}")
        print(f"Risk level: {result['analysis']['risk_level']}")
        print(f"Risk factors: {result['analysis']['risk_factors']}")
        for feature in sorted(result['features'].keys()):
            print(f"{feature}: {result['features'][feature]}")