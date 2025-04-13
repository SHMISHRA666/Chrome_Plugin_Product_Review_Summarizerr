// Background script (service worker) for Smart Purchase Advisor

// Initialize extension when installed
chrome.runtime.onInstalled.addListener(() => {
  console.log('Smart Purchase Advisor installed');
  
  // Initialize storage with default values
  chrome.storage.local.set({
    serverUrl: 'http://localhost:8080/api/detect-product',
    lastAnalysis: null,
    extensionState: {
      contentScriptLoaded: false,
      lastError: null
    }
  });
});

// Handle URL changes to ensure content script is loaded
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url && isAmazonProductPage(tab.url)) {
    console.log(`Tab updated: ${tab.url} (Amazon product page detected)`);
    
    // Check if we need to inject the content script
    chrome.tabs.sendMessage(tabId, { action: 'ping' }, response => {
      if (chrome.runtime.lastError) {
        console.log('Content script not yet loaded, will inject');
        
        // Try to inject the content script
        chrome.scripting.executeScript({
          target: { tabId: tabId },
          files: ['content.js']
        }).then(() => {
          console.log('Content script injected successfully');
          chrome.storage.local.set({
            extensionState: {
              contentScriptLoaded: true,
              lastError: null
            }
          });
        }).catch(error => {
          console.error('Error injecting content script:', error);
          chrome.storage.local.set({
            extensionState: {
              contentScriptLoaded: false,
              lastError: `Failed to inject content script: ${error.message}`
            }
          });
        });
      } else {
        console.log('Content script is already loaded');
        chrome.storage.local.set({
          extensionState: {
            contentScriptLoaded: true,
            lastError: null
          }
        });
      }
    });
  }
});

// Function to check if a URL is an Amazon product page
function isAmazonProductPage(url) {
  return /amazon\.(com|co\.uk|ca|de|fr|es|it|nl|in|co\.jp|com\.au).*\/(dp|gp\/product|product-reviews)\/[A-Z0-9]{10}/i.test(url);
}

// Listen for messages from popup or content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Log all incoming messages
  console.log('Background received message:', message, 'from:', sender.tab ? `Tab ${sender.tab.id}` : 'Extension');
  
  // Track content script loaded status
  if (message.action === 'contentScriptLoaded') {
    console.log('Content script loaded in tab:', sender.tab ? sender.tab.id : 'unknown');
    chrome.storage.local.set({
      extensionState: {
        contentScriptLoaded: true,
        lastError: null
      }
    });
    sendResponse({ status: 'acknowledged' });
    return true;
  }
  
  // Get API server URL from storage
  if (message.action === 'getApiUrl') {
    chrome.storage.local.get(['serverUrl'], (result) => {
      sendResponse({ 
        url: result.serverUrl || 'http://localhost:8080/api/detect-product' 
      });
    });
    return true; // Keeps the message channel open for async response
  }
  
  // Check if content script is loaded
  if (message.action === 'isContentScriptLoaded') {
    chrome.storage.local.get(['extensionState'], (result) => {
      sendResponse({
        isLoaded: result.extensionState ? result.extensionState.contentScriptLoaded : false,
        lastError: result.extensionState ? result.extensionState.lastError : null
      });
    });
    return true;
  }
  
  // Check if current page is an Amazon product page
  if (message.action === 'checkProductPage') {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs.length === 0) {
        sendResponse({ isProductPage: false, error: 'No active tab found' });
        return;
      }
      
      const url = tabs[0].url;
      // Check if it's an Amazon product page
      const isProductPage = isAmazonProductPage(url);
      
      sendResponse({ 
        isProductPage: isProductPage,
        url: url
      });
    });
    return true; // Keeps the message channel open for async response
  }
  
  // Simple ping to check if background script is responsive
  if (message.action === 'ping') {
    sendResponse({ status: 'alive' });
    return true;
  }
  
  // Return true for any unhandled messages to avoid connection errors
  sendResponse({ status: 'unknown_command' });
  return true;
}); 