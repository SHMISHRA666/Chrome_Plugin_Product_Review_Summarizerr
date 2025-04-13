/**
 * Content script for Smart Purchase Advisor
 * This script runs on Amazon product pages to scrape product data and reviews
 * It communicates with the extension popup via Chrome message passing API
 */

// Send a message to let the extension know the content script is loaded
console.log('Smart Purchase Advisor content script loaded');
chrome.runtime.sendMessage({ action: 'contentScriptLoaded' });

/**
 * Listener for messages from the extension popup or background script
 * Handles different message actions: ping, scrapeProductData
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Content script received message:', message);
  
  // Respond to ping message to confirm content script is loaded
  if (message.action === 'ping') {
    console.log('Received ping, responding with pong');
    sendResponse({ status: 'pong' });
    return true;
  }
  
  /**
   * Main functionality: scrape product data and reviews when requested
   * Attempts to get basic product info and reviews using multiple strategies:
   * 1. Try to get reviews from current product page
   * 2. If that fails, check if we're on a review page and scrape from there
   * 3. If not on a review page, try to navigate to the reviews page
   */
  if (message.action === 'scrapeProductData') {
    console.log('Starting product data scraping...');
    
    try {
      // Get basic product info (title, price, etc.)
      const productData = getProductInfo();
      console.log('Product info:', productData);
      
      // Try to get reviews from the current page
      const pageReviews = getReviewsFromCurrentPage();
      console.log(`Found ${pageReviews.length} reviews on current page`);
      
      if (pageReviews.length > 0) {
        // If we have reviews on the current page, use them
        productData.reviews = pageReviews;
        productData.extracted_reviews = true; // Flag to indicate we have reviews
        sendResponse({ productData: productData });
      } else {
        // If no reviews on current page, check if we're on a reviews page
        const isReviewPage = window.location.href.includes('/reviews/') || 
                             window.location.href.includes('/product-reviews/');
        
        if (isReviewPage) {
          // We're already on a reviews page, scrape all reviews
          const allReviews = getAllReviewsFromReviewPage();
          productData.reviews = allReviews;
          productData.extracted_reviews = allReviews.length > 0;
          console.log(`Found ${allReviews.length} reviews on review page`);
          sendResponse({ productData: productData });
        } else {
          // Get the reviews page URL
          const reviewsUrl = getReviewsPageUrl();
          console.log('Reviews URL:', reviewsUrl);
          
          if (reviewsUrl) {
            // Fetch reviews from the reviews page
            fetchReviewsFromUrl(reviewsUrl)
              .then(reviews => {
                productData.reviews = reviews;
                productData.extracted_reviews = reviews.length > 0;
                console.log(`Fetched ${reviews.length} reviews from URL`);
                sendResponse({ productData: productData });
              })
              .catch(error => {
                console.error('Error fetching reviews:', error);
                // Return product data without reviews
                productData.reviews = [];
                productData.extracted_reviews = false;
                sendResponse({ 
                  productData: productData,
                  warning: 'Could not fetch reviews. Analysis may be limited.'
                });
              });
            
            // Return true to indicate we'll respond asynchronously
            return true;
          } else {
            // Couldn't find reviews URL, return what we have
            productData.reviews = [];
            productData.extracted_reviews = false;
            sendResponse({ 
              productData: productData,
              warning: 'No reviews found. Analysis may be limited.'
            });
          }
        }
      }
    } catch (error) {
      console.error('Error scraping product data:', error);
      sendResponse({ 
        error: `Error scraping product data: ${error.message}` 
      });
    }
    
    return true; // Keep the message channel open
  }
  
  // For any unhandled message, send a response to prevent connection errors
  sendResponse({ status: 'unknown_message' });
  return true;
});

/**
 * Extract basic product information from the page
 * Tries multiple selectors for each piece of information to handle Amazon's variable DOM structure
 * @returns {Object} Product data object with title, site, price, URL
 */
function getProductInfo() {
  const productData = {
    title: '',
    site: 'amazon.com',
    price: '',
    url: window.location.href,
    reviews: [],
    include_full_details: true  // Flag to tell server to include all details including pros/cons
  };
  
  try {
    // Extract product title - try multiple potential selectors
    const titleSelectors = [
      '#productTitle',
      '.product-title',
      'h1.a-size-large',
      '#title',
      '.a-size-extra-large'
    ];
    
    for (const selector of titleSelectors) {
      const titleElement = document.querySelector(selector);
      if (titleElement && titleElement.textContent.trim()) {
        productData.title = titleElement.textContent.trim();
        break;
      }
    }
    
    // Extract product price - try multiple potential selectors
    const priceSelectors = [
      '.a-price .a-offscreen',
      '#priceblock_ourprice',
      '#priceblock_dealprice',
      '.a-price',
      '.price',
      '#price'
    ];
    
    for (const selector of priceSelectors) {
      const priceElement = document.querySelector(selector);
      if (priceElement && priceElement.textContent.trim()) {
        productData.price = priceElement.textContent.trim().replace(/\s+/g, ' ');
        break;
      }
    }
    
    // If we still don't have a title or price, try to extract from JSON-LD
    try {
      const jsonldElements = document.querySelectorAll('script[type="application/ld+json"]');
      for (const element of jsonldElements) {
        try {
          const data = JSON.parse(element.textContent);
          if (data.name && !productData.title) {
            productData.title = data.name;
          }
          if (data.offers && data.offers.price && !productData.price) {
            productData.price = `$${data.offers.price}`;
          }
        } catch (e) {
          console.error('Error parsing JSON-LD:', e);
        }
      }
    } catch (e) {
      console.error('Error extracting from JSON-LD:', e);
    }
    
    // Extract ASIN from URL
    const asinMatch = window.location.pathname.match(/\/(dp|product|product-reviews)\/([A-Z0-9]{10})/i);
    if (asinMatch && asinMatch[2]) {
      productData.asin = asinMatch[2];
    }
    
    // Extract domain for the site field
    const domainMatch = window.location.hostname.match(/amazon\.([a-z.]+)/i);
    if (domainMatch && domainMatch[1]) {
      productData.site = 'amazon.' + domainMatch[1];
    }
  } catch (error) {
    console.error('Error in getProductInfo:', error);
  }
  
  return productData;
}

/**
 * Extract reviews from the current page
 * Uses multiple selectors to find review text elements
 * @returns {Array} Array of review text strings
 */
function getReviewsFromCurrentPage() {
  const reviews = [];
  
  try {
    // Try various selectors for review content
    const reviewSelectors = [
      '.review-text-content span',
      '[data-hook="review-body"] span',
      '.a-expander-content p',
      '.a-expander-content',
      '.review-data .review-text',
      'div[data-hook="review-collapsed"]',
      '.a-section .review-text',
      '.cr-review-text',
    ];
    
    // Try each selector one by one until we find reviews
    for (const selector of reviewSelectors) {
      const reviewElements = document.querySelectorAll(selector);
      
      if (reviewElements && reviewElements.length > 0) {
        reviewElements.forEach(el => {
          const reviewText = el.textContent.trim();
          if (reviewText && reviewText.length > 10) {  // Minimum review length
            reviews.push(reviewText);
          }
        });
        
        // If we found reviews, no need to try other selectors
        if (reviews.length > 0) {
          break;
        }
      }
    }
  } catch (error) {
    console.error('Error getting reviews from current page:', error);
  }
  
  return reviews;
}

/**
 * Extract all reviews from a dedicated reviews page
 * Uses Amazon-specific selectors for the review page layout
 * @returns {Array} Array of review text strings
 */
function getAllReviewsFromReviewPage() {
  const reviews = [];
  
  try {
    // Try selectors specific to Amazon review pages
    const reviewElements = document.querySelectorAll('.review-text-content, [data-hook="review-body"]');
    
    reviewElements.forEach(el => {
      const reviewText = el.textContent.trim();
      if (reviewText && reviewText.length > 10) {
        reviews.push(reviewText);
      }
    });
    
    // If we don't find reviews with primary selectors, try alternative selectors
    if (reviews.length === 0) {
      const altReviewElements = document.querySelectorAll('.a-expander-content');
      altReviewElements.forEach(el => {
        const reviewText = el.textContent.trim();
        if (reviewText && reviewText.length > 10) {
          reviews.push(reviewText);
        }
      });
    }
  } catch (error) {
    console.error('Error getting reviews from review page:', error);
  }
  
  return reviews;
}

/**
 * Find the URL to the product's review page
 * Looks for links containing "See all reviews" or similar text
 * @returns {string|null} URL to reviews page or null if not found
 */
function getReviewsPageUrl() {
  try {
    // Try to find "See all reviews" link - check multiple potential selectors
    const reviewLinkSelectors = [
      'a[data-hook="see-all-reviews-link-foot"]',
      'a[data-hook="see-all-reviews-link"]',
      'a.a-link-emphasis:contains("See all reviews")',
      'a:contains("See all reviews")',
      'a:contains("customer reviews")',
      'a[href*="/product-reviews/"]',
      'a[href*="/reviews/"]'
    ];
    
    // Special selector handling for jQuery-like contains selector
    for (const selector of reviewLinkSelectors) {
      let elements;
      
      if (selector.includes(':contains(')) {
        // Handle contains text selector manually since we're not using jQuery
        const baseSelector = selector.split(':contains(')[0];
        const searchText = selector.match(/:contains\("(.+?)"\)/)[1];
        elements = Array.from(document.querySelectorAll(baseSelector))
          .filter(el => el.textContent.includes(searchText));
      } else {
        elements = document.querySelectorAll(selector);
      }
      
      if (elements && elements.length > 0) {
        // Get the first matching element's href
        const href = elements[0].getAttribute('href');
        if (href) {
          // Convert relative URL to absolute if needed
          if (href.startsWith('/')) {
            return `${window.location.origin}${href}`;
          }
          return href;
        }
      }
    }
    
    // Try to construct review URL from ASIN if we couldn't find a link
    const asinMatch = window.location.pathname.match(/\/(dp|product)\/([A-Z0-9]{10})/i);
    if (asinMatch && asinMatch[2]) {
      const asin = asinMatch[2];
      return `${window.location.origin}/product-reviews/${asin}`;
    }
  } catch (error) {
    console.error('Error finding reviews URL:', error);
  }
  
  return null;
}

/**
 * Fetch reviews from a specific URL
 * Makes a request to the reviews page and extracts review content
 * @param {string} url - URL to fetch reviews from
 * @returns {Promise<Array>} Promise resolving to array of review text strings
 */
async function fetchReviewsFromUrl(url) {
  try {
    // Create a hidden iframe to load the reviews page
    return new Promise((resolve, reject) => {
      // Set a timeout to avoid hanging if something goes wrong
      const timeoutId = setTimeout(() => {
        reject(new Error('Timeout while fetching reviews'));
      }, 15000);
      
      // Create an iframe to load the reviews page
      const iframe = document.createElement('iframe');
      iframe.style.display = 'none';
      
      iframe.onload = () => {
        try {
          clearTimeout(timeoutId);
          
          // Access the iframe document and extract reviews
          const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
          const reviews = [];
          
          // Try different review selectors within the iframe
          const reviewElements = iframeDoc.querySelectorAll(
            '.review-text-content, [data-hook="review-body"], .a-expander-content'
          );
          
          reviewElements.forEach(el => {
            const reviewText = el.textContent.trim();
            if (reviewText && reviewText.length > 10) {
              reviews.push(reviewText);
            }
          });
          
          // Remove the iframe after extracting reviews
          document.body.removeChild(iframe);
          resolve(reviews);
        } catch (error) {
          clearTimeout(timeoutId);
          document.body.removeChild(iframe);
          reject(error);
        }
      };
      
      iframe.onerror = (error) => {
        clearTimeout(timeoutId);
        document.body.removeChild(iframe);
        reject(error);
      };
      
      // Set the iframe source and add it to the page
      iframe.src = url;
      document.body.appendChild(iframe);
    });
  } catch (error) {
    console.error('Error in fetchReviewsFromUrl:', error);
    return [];
  }
} 