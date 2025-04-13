// Wait for DOM content to load
document.addEventListener('DOMContentLoaded', () => {
  // DOM elements
  const analyzeButton = document.getElementById('analyze-button');
  const statusMessage = document.getElementById('status-message');
  const loader = document.getElementById('loader');
  const resultsContainer = document.getElementById('results-container');
  const productTitle = document.getElementById('product-title');
  const confidenceScore = document.getElementById('confidence-score');
  const confidenceLevel = document.getElementById('confidence-level');
  const sentimentScore = document.getElementById('sentiment-score');
  const sentimentText = document.getElementById('sentiment-text');
  const prosList = document.getElementById('pros-list');
  const consList = document.getElementById('cons-list');
  const reviewCount = document.getElementById('review-count');
  const warningsContainer = document.getElementById('warnings-container');
  const warningsList = document.getElementById('warnings-list');
  
  // Server URL for API requests - make sure this matches your backend
  const apiUrl = 'http://localhost:8080/api/detect-product';
  
  // Check if current page is an Amazon product page and if content script is loaded
  function checkIfProductPage() {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs || tabs.length === 0) {
        disableAnalysis('Cannot detect current tab');
        return;
      }
      
      const currentUrl = tabs[0].url;
      console.log("Current URL:", currentUrl);
      
      // More comprehensive Amazon product page detection
      const isAmazonProduct = /amazon\.(com|co\.uk|ca|de|fr|es|it|nl|in|co\.jp|com\.au).*\/(dp|gp\/product|product-reviews)\/[A-Z0-9]{10}/i.test(currentUrl);
      
      if (isAmazonProduct) {
        console.log("Amazon product detected");
        // Check if content script is loaded
        ensureContentScriptLoaded(tabs[0].id, () => {
          enableAnalysis();
        }, () => {
          disableAnalysis('Content script could not be loaded. Try refreshing the page.');
        });
      } else {
        console.log("Not an Amazon product page");
        disableAnalysis('Navigate to an Amazon product page to analyze reviews');
      }
    });
  }
  
  // Make sure content script is loaded before proceeding
  function ensureContentScriptLoaded(tabId, onSuccess, onFailure) {
    // First try to ping the content script
    chrome.tabs.sendMessage(tabId, { action: 'ping' }, response => {
      if (chrome.runtime.lastError || !response) {
        console.log("Content script not detected, attempting to inject it");
        
        // Script isn't loaded, try to inject it
        chrome.scripting.executeScript({
          target: { tabId: tabId },
          files: ['content.js']
        }).then(() => {
          console.log("Content script injection successful");
          
          // Give it a moment to initialize
          setTimeout(() => {
            // Verify injection worked
            chrome.tabs.sendMessage(tabId, { action: 'ping' }, verifyResponse => {
              if (chrome.runtime.lastError || !verifyResponse) {
                console.error("Content script still not responding after injection");
                onFailure();
              } else {
                console.log("Content script now responding");
                onSuccess();
              }
            });
          }, 200);
        }).catch(error => {
          console.error("Failed to inject content script:", error);
          onFailure();
        });
      } else {
        console.log("Content script already loaded");
        onSuccess();
      }
    });
  }
  
  // Enable analysis button
  function enableAnalysis() {
    statusMessage.textContent = 'Ready to analyze Amazon product reviews';
    statusMessage.className = 'status-info';
    analyzeButton.disabled = false;
  }
  
  // Disable analysis button with message
  function disableAnalysis(message) {
    statusMessage.textContent = message;
    statusMessage.className = 'status-warning';
    analyzeButton.disabled = true;
    hideResults();
  }
  
  // Show loading state
  function showLoading() {
    analyzeButton.disabled = true;
    loader.classList.remove('hidden');
    statusMessage.textContent = 'Analyzing product reviews...';
    statusMessage.className = 'status-info';
    hideResults();
  }
  
  // Show error state
  function showError(message) {
    statusMessage.textContent = message;
    statusMessage.className = 'status-error';
    loader.classList.add('hidden');
    analyzeButton.disabled = false;
    hideResults();
  }
  
  // Hide results container
  function hideResults() {
    resultsContainer.classList.add('hidden');
  }
  
  // Start analysis process
  function startAnalysis() {
    showLoading();
    
    // Request product data from content script
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs || tabs.length === 0) {
        showError('Cannot detect current tab');
        return;
      }
      
      // First ensure the content script is loaded
      ensureContentScriptLoaded(tabs[0].id, () => {
        // Now that we know the content script is loaded, send the scrape request
        chrome.tabs.sendMessage(
          tabs[0].id,
          { action: 'scrapeProductData' },
          (response) => {
            // Handle any communication errors
            if (chrome.runtime.lastError) {
              console.error("Runtime error:", chrome.runtime.lastError);
              showError(`Error: ${chrome.runtime.lastError.message}`);
              return;
            }
            
            // Check for valid response
            if (!response || !response.productData) {
              showError('Failed to retrieve product data');
              return;
            }
            
            // If there's a warning in the response
            if (response.warning) {
              statusMessage.textContent = response.warning;
              statusMessage.className = 'status-warning';
            }
            
            // Send product data to API server
            sendToApiServer(response.productData);
          }
        );
      }, () => {
        showError('Could not communicate with the page. Try refreshing the page and try again.');
      });
    });
  }
  
  // Send product data to API server
  function sendToApiServer(productData) {
    console.log("Sending data to API:", productData);
    
    // Make sure we have the title and site for the API
    if (!productData.title || productData.title.trim() === '') {
      productData.title = document.title || 'Unknown Product';
    }
    
    // Make sure site is set
    if (!productData.site || productData.site.trim() === '') {
      productData.site = 'amazon.com';
    }
    
    fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(productData)
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log("Received API response:", data);
      
      // Save analysis data
      chrome.storage.local.set({ 
        lastAnalysis: {
          data: data,
          timestamp: Date.now(),
          productUrl: productData.url
        }
      });
      
      // Display results
      displayResults(data, productData.title);
    })
    .catch(error => {
      console.error('API Error:', error);
      showError(`Error connecting to server: ${error.message}. Make sure the backend API is running at ${apiUrl}`);
    });
  }
  
  // Display analysis results
  function displayResults(data, title) {
    // Check if data contains an error
    if (data.error) {
      showError(`Analysis error: ${data.error}`);
      return;
    }
    
    // Update loading state
    loader.classList.add('hidden');
    statusMessage.textContent = 'Analysis complete!';
    statusMessage.className = 'status-success';
    analyzeButton.disabled = false;
    
    // Update product title
    productTitle.textContent = title || data.title || 'Product';
    
    // Update confidence score
    const confScore = data.confidence_score || 0;
    confidenceScore.textContent = `${confScore}%`;
    confidenceScore.style.backgroundColor = getConfidenceColor(confScore);
    confidenceLevel.textContent = data.confidence_level || 'Unknown';
    
    // Update sentiment score and text
    const sentScore = data.sentiment_score !== undefined ? data.sentiment_score : 0;
    sentimentScore.textContent = sentScore.toFixed(2);
    sentimentScore.style.backgroundColor = getSentimentColor(sentScore);
    sentimentText.textContent = data.overall_sentiment || 'Neutral';
    
    // Update pros list
    prosList.innerHTML = '';
    if (data.pros && data.pros.length > 0) {
      data.pros.forEach(pro => {
        const li = document.createElement('li');
        li.textContent = pro;
        prosList.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.textContent = 'No pros identified';
      prosList.appendChild(li);
    }
    
    // Update cons list
    consList.innerHTML = '';
    if (data.cons && data.cons.length > 0) {
      data.cons.forEach(con => {
        const li = document.createElement('li');
        li.textContent = con;
        consList.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.textContent = 'No cons identified';
      consList.appendChild(li);
    }
    
    // Update review count
    reviewCount.textContent = data.review_count || 0;
    
    // Show warnings if any
    if (data.warnings && data.warnings.length > 0) {
      warningsList.innerHTML = '';
      data.warnings.forEach(warning => {
        const li = document.createElement('li');
        li.textContent = warning;
        warningsList.appendChild(li);
      });
      warningsContainer.classList.remove('hidden');
    } else {
      warningsContainer.classList.add('hidden');
    }
    
    // Show results
    resultsContainer.classList.remove('hidden');
  }
  
  // Get color for confidence score
  function getConfidenceColor(score) {
    if (score >= 80) return 'var(--success-color)';
    if (score >= 60) return 'var(--primary-color)';
    if (score >= 40) return 'var(--warning-color)';
    return 'var(--error-color)';
  }
  
  // Get color for sentiment score
  function getSentimentColor(score) {
    if (score >= 0.5) return 'var(--success-color)';
    if (score >= 0) return 'var(--primary-color)';
    if (score >= -0.5) return 'var(--warning-color)';
    return 'var(--error-color)';
  }
  
  // Event Listeners
  analyzeButton.addEventListener('click', startAnalysis);
  
  // Check if current page is valid when popup opens
  checkIfProductPage();
}); 