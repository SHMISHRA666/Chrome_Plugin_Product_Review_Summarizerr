/**
 * Background script (service worker) for Smart Purchase Advisor
 * Handles extension initialization, content script injection, and cross-script communication
 */

/**
 * Initialize extension settings when installed or updated
 * Sets default values in chrome.storage.local for extension configuration
 */
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

/**
 * Detect URL changes to inject content script when navigating to Amazon product pages
 * Ensures the content script is loaded whenever a user visits a relevant product page
 */
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

/**
 * Determine if a URL is an Amazon product page based on URL pattern
 * Matches various Amazon domains and product page URL patterns
 * 
 * @param {string} url - The URL to check
 * @returns {boolean} True if the URL is an Amazon product page
 */
function isAmazonProductPage(url) {
  return /amazon\.(com|co\.uk|ca|de|fr|es|it|nl|in|co\.jp|com\.au).*\/(dp|gp\/product|product-reviews)\/[A-Z0-9]{10}/i.test(url);
}

/**
 * Message listener for handling communication between popup and content script
 * Processes various action requests and returns appropriate responses
 */
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
  
  /**
   * Process API results and prepare data for display
   * Validates and standardizes pros/cons arrays and adds warnings if needed
   */
  if (message.action === 'processApiResults') {
    console.log('Processing API results:', message.data);
    // Make sure pros and cons are arrays
    if (message.data) {
      // Handle case where pros or cons might be hidden 
      let needsRefresh = false;
      
      if (!Array.isArray(message.data.pros)) {
        message.data.pros = [];
      } else if (message.data.pros.length === 1 && 
                typeof message.data.pros[0] === 'string' && 
                message.data.pros[0].includes('hidden')) {
        // If we received hidden pros, flag for refresh
        needsRefresh = true;
      }
      
      if (!Array.isArray(message.data.cons)) {
        message.data.cons = [];
      } else if (message.data.cons.length === 1 && 
                typeof message.data.cons[0] === 'string' && 
                message.data.cons[0].includes('hidden')) {
        // If we received hidden cons, flag for refresh
        needsRefresh = true;
      }
      
      // If we need to refresh data, add a warning
      if (needsRefresh) {
        if (!Array.isArray(message.data.warnings)) {
          message.data.warnings = [];
        }
        message.data.warnings.push("Some details are not fully displayed. Click 'Analyze Reviews' again to see all details.");
      }
      
      sendResponse({ 
        status: 'success',
        processedData: message.data,
        needsRefresh: needsRefresh
      });
    } else {
      sendResponse({ 
        status: 'error',
        error: 'No data to process'
      });
    }
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