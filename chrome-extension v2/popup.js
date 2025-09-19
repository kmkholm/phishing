document.addEventListener('DOMContentLoaded', function() {
  // DOM Elements
  const urlInput = document.getElementById('url-input');
  const checkUrlBtn = document.getElementById('check-url-btn');
  const checkCurrentBtn = document.getElementById('check-current-btn');
  const loader = document.getElementById('loader');
  const resultContainer = document.getElementById('result-container');
  const resultHeader = document.getElementById('result-header');
  const resultDetails = document.getElementById('result-details');
  const resultConfidence = document.getElementById('result-confidence');
  const riskFactors = document.getElementById('risk-factors');
  const historyList = document.getElementById('history-list');
  const clearHistoryBtn = document.getElementById('clear-history-btn');
  const serverUrlInput = document.getElementById('server-url');
  const saveSettingsBtn = document.getElementById('save-settings-btn');

  // Check if we received a URL to check from background script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'checkUrl' && message.url) {
      urlInput.value = message.url;
      checkUrl(message.url);
    }
    sendResponse({received: true});
    return true;
  });

  // Load settings
  chrome.storage.local.get('serverUrl', function(data) {
    if (data.serverUrl) {
      serverUrlInput.value = data.serverUrl;
    } else {
      serverUrlInput.value = 'http://localhost:5000';
    }
  });

  // Save settings
  saveSettingsBtn.addEventListener('click', function() {
    const serverUrl = serverUrlInput.value.trim();
    if (serverUrl) {
      chrome.storage.local.set({ 'serverUrl': serverUrl }, function() {
        showNotification('Settings saved!');
      });
    } else {
      showNotification('Please enter a valid server URL', 'error');
    }
  });

  // Check current URL
  checkCurrentBtn.addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const currentUrl = tabs[0].url;
      urlInput.value = currentUrl;
      checkUrl(currentUrl);
    });
  });

  // Check inputted URL
  checkUrlBtn.addEventListener('click', function() {
    const url = urlInput.value.trim();
    if (url) {
      checkUrl(url);
    } else {
      showNotification('Please enter a URL', 'error');
    }
  });

  // Allow Enter key to trigger URL check
  urlInput.addEventListener('keyup', function(event) {
    if (event.key === 'Enter') {
      const url = urlInput.value.trim();
      if (url) {
        checkUrl(url);
      } else {
        showNotification('Please enter a URL', 'error');
      }
    }
  });

  // Check URL function
  function checkUrl(url) {
    showLoader();
    
    // Validate and format URL
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'http://' + url;
    }
    
    chrome.storage.local.get('serverUrl', function(data) {
      const serverUrl = data.serverUrl || 'http://localhost:5000';
      
      fetch(`${serverUrl}/api/check-url`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Server responded with status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        hideLoader();
        displayResult(url, data);
        saveToHistory(url, data);
      })
      .catch(error => {
        hideLoader();
        displayError(error);
        console.error('Error:', error);
      });
    });
  }

  // Display result
  function displayResult(url, data) {
    resultContainer.style.display = 'block';
    
    // Clear previous risk factors
    riskFactors.innerHTML = '';
    
    if (data.is_phishing) {
      resultHeader.innerHTML = '<span class="phishing">⚠️ Phishing Detected!</span>';
      resultDetails.innerHTML = `The URL <strong>${url}</strong> appears to be a phishing website.`;
      
      // Add confidence meter
      const confidencePercent = (data.confidence * 100).toFixed(2);
      resultConfidence.innerHTML = `
        <div>Confidence: ${confidencePercent}%</div>
        <div class="confidence-meter">
          <div class="confidence-indicator confidence-high" style="width: ${confidencePercent}%"></div>
        </div>
      `;
      
      // Add risk factors if available
      if (data.risk_factors && data.risk_factors.length > 0) {
        let factorsHtml = '<strong>Risk factors detected:</strong><ul>';
        data.risk_factors.forEach(factor => {
          factorsHtml += `<li>${factor}</li>`;
        });
        factorsHtml += '</ul>';
        riskFactors.innerHTML = factorsHtml;
      }
    } else {
      const confidencePercent = (data.confidence * 100).toFixed(2);
      
      // Determine if it's safe or warning based on confidence
      if (confidencePercent >= 75) {
        resultHeader.innerHTML = '<span class="safe">✅ Safe Website</span>';
        resultDetails.innerHTML = `The URL <strong>${url}</strong> appears to be legitimate.`;
        resultConfidence.innerHTML = `
          <div>Confidence: ${confidencePercent}%</div>
          <div class="confidence-meter">
            <div class="confidence-indicator confidence-high" style="width: ${confidencePercent}%"></div>
          </div>
        `;
      } else {
        resultHeader.innerHTML = '<span class="warning">⚠️ Possibly Safe, Low Confidence</span>';
        resultDetails.innerHTML = `The URL <strong>${url}</strong> doesn't appear malicious, but proceed with caution.`;
        resultConfidence.innerHTML = `
          <div>Confidence: ${confidencePercent}%</div>
          <div class="confidence-meter">
            <div class="confidence-indicator confidence-medium" style="width: ${confidencePercent}%"></div>
          </div>
        `;
        
        if (data.risk_factors && data.risk_factors.length > 0) {
          let factorsHtml = '<strong>Some concerns detected:</strong><ul>';
          data.risk_factors.forEach(factor => {
            factorsHtml += `<li>${factor}</li>`;
          });
          factorsHtml += '</ul>';
          riskFactors.innerHTML = factorsHtml;
        }
      }
    }
    
    // Scroll to show results
    resultContainer.scrollIntoView({ behavior: 'smooth' });
  }

  // Display error
  function displayError(error) {
    resultContainer.style.display = 'block';
    resultHeader.innerHTML = '<span class="phishing">❌ Error</span>';
    resultDetails.textContent = 'Could not connect to the phishing detection server.';
    resultConfidence.textContent = 'Please check server settings and try again.';
    riskFactors.innerHTML = `<strong>Error details:</strong> ${error.message}`;
  }

  // Show notification
  function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.classList.add('show');
      
      setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
          document.body.removeChild(notification);
        }, 300);
      }, 2000);
    }, 10);
  }

  // Show/hide loader
  function showLoader() {
    loader.style.display = 'block';
    resultContainer.style.display = 'none';
  }

  function hideLoader() {
    loader.style.display = 'none';
  }

  // Save to history
  function saveToHistory(url, data) {
    chrome.storage.local.get('history', function(result) {
      const history = result.history || [];
      
      // Add to beginning of array, limit to 10 items
      history.unshift({
        url: url,
        isPhishing: data.is_phishing,
        confidence: data.confidence,
        timestamp: new Date().toISOString()
      });
      
      // Remove duplicates
      const uniqueHistory = history.filter((item, index, self) => 
        index === self.findIndex(t => t.url === item.url)
      );
      
      // Limit to 10 items
      if (uniqueHistory.length > 10) {
        uniqueHistory.length = 10;
      }
      
      chrome.storage.local.set({ 'history': uniqueHistory }, function() {
        displayHistory();
      });
    });
  }

  // Display history
  function displayHistory() {
    chrome.storage.local.get('history', function(result) {
      const history = result.history || [];
      historyList.innerHTML = '';
      
      if (history.length === 0) {
        historyList.innerHTML = '<div class="history-item">No history yet.</div>';
        return;
      }
      
      history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.setAttribute('data-url', item.url);
        
        const urlSpan = document.createElement('span');
        urlSpan.className = 'history-url';
        urlSpan.textContent = item.url;
        urlSpan.title = item.url;
        
        const resultSpan = document.createElement('span');
        resultSpan.className = 'history-result ' + (item.isPhishing ? 'phishing' : 'safe');
        resultSpan.textContent = item.isPhishing ? 'Phishing' : 'Safe';
        
        historyItem.appendChild(urlSpan);
        historyItem.appendChild(resultSpan);
        
        // Add click event to re-check the URL
        historyItem.addEventListener('click', function() {
          urlInput.value = item.url;
          checkUrl(item.url);
        });
        
        historyList.appendChild(historyItem);
      });
    });
  }

  // Clear history
  clearHistoryBtn.addEventListener('click', function() {
    chrome.storage.local.set({ 'history': [] }, function() {
      displayHistory();
      showNotification('History cleared');
    });
  });

  // Initial load of history
  displayHistory();
});