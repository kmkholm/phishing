// Initialize extension when installed
chrome.runtime.onInstalled.addListener(() => {
  // Set default settings
  chrome.storage.local.get('serverUrl', function(data) {
    if (!data.serverUrl) {
      chrome.storage.local.set({ 'serverUrl': 'http://localhost:5000' });
    }
  });
  
  // Create context menu items
  chrome.contextMenus.create({
    id: 'check-link',
    title: 'Check if link is phishing',
    contexts: ['link']
  });
  
  chrome.contextMenus.create({
    id: 'check-page',
    title: 'Check this page for phishing',
    contexts: ['page']
  });
  
  console.log('Phishing Detector extension installed and initialized!');
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'check-link') {
    const url = info.linkUrl;
    
    // Open the popup with the URL to check
    checkUrlInPopup(url);
    
  } else if (info.menuItemId === 'check-page') {
    // Get the current page URL
    checkUrlInPopup(tab.url);
  }
});

// Function to check URL in popup
function checkUrlInPopup(url) {
  // We need to open the popup programmatically
  chrome.action.openPopup().catch(() => {
    // If we can't open the popup directly (this is the case in Manifest V3),
    // we'll send a message that the popup will receive when it's opened
    chrome.storage.local.set({ 'pendingUrlCheck': url }, () => {
      // Show a badge to indicate there's a URL to check
      chrome.action.setBadgeText({ text: '!' });
      chrome.action.setBadgeBackgroundColor({ color: '#dc3545' });
      
      // Notify the user to click the extension icon
      chrome.notifications.create({
        type: 'basic',
        iconUrl: 'images/icon128.png',
        title: 'Phishing Detector',
        message: 'Click the extension icon to check the selected URL'
      });
    });
  });
}

// Listen for when the popup is opened
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'popupOpened') {
    // Check if we have a pending URL to check
    chrome.storage.local.get('pendingUrlCheck', (data) => {
      if (data.pendingUrlCheck) {
        // Send the URL to the popup
        chrome.runtime.sendMessage({
          action: 'checkUrl',
          url: data.pendingUrlCheck
        });
        
        // Clear the pending URL and badge
        chrome.storage.local.remove('pendingUrlCheck');
        chrome.action.setBadgeText({ text: '' });
      }
    });
    sendResponse({received: true});
  }
  return true;
});

// Add tab-specific features
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    // Option: Auto-check pages
    chrome.storage.local.get('autoCheck', (data) => {
      if (data.autoCheck) {
        // Send a message to check this URL automatically
        chrome.runtime.sendMessage({
          action: 'checkUrl',
          url: tab.url
        });
      }
    });
  }
});

// Add optional safety features
// Set up alarm for periodic background checks
chrome.alarms.create('periodicSafetyCheck', { periodInMinutes: 60 });

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'periodicSafetyCheck') {
    // Get all open tabs and check them in the background
    chrome.tabs.query({}, (tabs) => {
      // Option: Check if background scanning is enabled
      chrome.storage.local.get('backgroundScan', (data) => {
        if (data.backgroundScan) {
          tabs.forEach(tab => {
            if (tab.url && tab.url.startsWith('http')) {
              // Perform a background check
              checkUrlInBackground(tab.url, tab.id);
            }
          });
        }
      });
    });
  }
});

// Function to check URLs in the background without opening the popup
function checkUrlInBackground(url, tabId) {
  chrome.storage.local.get('serverUrl', (data) => {
    const serverUrl = data.serverUrl || 'http://localhost:5000';
    
    fetch(`${serverUrl}/api/check-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: url })
    })
    .then(response => response.json())
    .then(data => {
      if (data.is_phishing && data.confidence > 0.7) {
        // Alert the user about high-confidence phishing detection
        chrome.notifications.create({
          type: 'basic',
          iconUrl: 'images/icon128.png',
          title: '⚠️ Phishing Alert!',
          message: `The page you're visiting (${url}) has been detected as a phishing site with high confidence.`,
          priority: 2
        });
        
        // Option: Add a warning badge to the tab
        chrome.action.setBadgeText({ text: '⚠️', tabId: tabId });
        chrome.action.setBadgeBackgroundColor({ color: '#dc3545', tabId: tabId });
      }
    })
    .catch(error => {
      console.error('Background check error:', error);
    });
  });
}