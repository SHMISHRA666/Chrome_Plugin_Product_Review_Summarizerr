// Content script for Smart Purchase Advisor
// This script runs on Amazon product pages to scrape product data and reviews

// Send a message to let the extension know the content script is loaded
console.log('Smart Purchase Advisor content script loaded');
chrome.runtime.sendMessage({ action: 'contentScriptLoaded' });

// Listen for messages from the extension popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Content script received message:', message);
  
  // Respond to ping message to confirm content script is loaded
  if (message.action === 'ping') {
    console.log('Received ping, responding with pong');
    sendResponse({ status: 'pong' });
    return true;
  }
  
  if (message.action === 'scrapeProductData') {
    console.log('Starting product data scraping...');
    
    try {
      // Get basic product info
      const productData = getProductInfo();
      console.log('Product info:', productData);
      
      // Try to get reviews from the current page
      const pageReviews = getReviewsFromCurrentPage();
      console.log(`Found ${pageReviews.length} reviews on current page`);
      
      if (pageReviews.length > 0) {
        // If we have reviews on the current page, use them
        productData.reviews = pageReviews;
        sendResponse({ productData: productData });
      } else {
        // If no reviews on current page, check if we're on a reviews page
        const isReviewPage = window.location.href.includes('/reviews/') || 
                             window.location.href.includes('/product-reviews/');
        
        if (isReviewPage) {
          // We're already on a reviews page, scrape all reviews
          const allReviews = getAllReviewsFromReviewPage();
          productData.reviews = allReviews;
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
                console.log(`Fetched ${reviews.length} reviews from URL`);
                sendResponse({ productData: productData });
              })
              .catch(error => {
                console.error('Error fetching reviews:', error);
                // Return product data without reviews
                productData.reviews = [];
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

// Function to extract basic product information
function getProductInfo() {
  const productData = {
    title: '',
    site: 'amazon.com',
    price: '',
    url: window.location.href,
    reviews: []
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

// Function to get reviews from the current product page
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
      'div[data-hook="review"] .a-spacing-small'
    ];
    
    // Try each selector and collect reviews
    for (const selector of reviewSelectors) {
      const elements = document.querySelectorAll(selector);
      
      elements.forEach(element => {
        const text = element.textContent.trim();
        if (text && text.length > 15 && !reviews.includes(text)) {
          reviews.push(text);
        }
      });
      
      // If we found reviews with this selector, stop trying others
      if (reviews.length > 0) {
        break;
      }
    }
  } catch (error) {
    console.error('Error in getReviewsFromCurrentPage:', error);
  }
  
  return reviews;
}

// Function to get all reviews from a review page
function getAllReviewsFromReviewPage() {
  const reviews = [];
  
  try {
    // Try various selectors for review content on review pages
    const reviewSelectors = [
      '[data-hook="review-body"] span',
      '.a-row.a-spacing-small.review-data span.review-text-content span',
      '.review-text-content span',
      '.a-expander-content p',
      '.cr-review-text',
      'div[data-hook="review"]'
    ];
    
    // Try each selector and collect reviews
    for (const selector of reviewSelectors) {
      const elements = document.querySelectorAll(selector);
      
      elements.forEach(element => {
        const text = element.textContent.trim();
        if (text && text.length > 15 && !reviews.includes(text)) {
          reviews.push(text);
        }
      });
      
      // If we found reviews with this selector, stop trying others
      if (reviews.length > 0) {
        break;
      }
    }
  } catch (error) {
    console.error('Error in getAllReviewsFromReviewPage:', error);
  }
  
  return reviews;
}

// Function to get the URL of the reviews page
function getReviewsPageUrl() {
  try {
    // Look for the "See all reviews" link
    const reviewsLinkSelectors = [
      'a[data-hook="see-all-reviews-link-foot"]',
      'a[href*="customerReviews"]',
      'a.a-link-emphasis[href*="reviews"]',
      'a[href*="/product-reviews/"]',
      'a[href*="/reviews/"]',
      'a[href*="showAllReviews"]',
      'a#ratings-summary'
    ];
    
    for (const selector of reviewsLinkSelectors) {
      const element = document.querySelector(selector);
      if (element && element.href) {
        return element.href;
      }
    }
    
    // If we couldn't find a direct link, try to construct one from the ASIN
    const asinMatch = window.location.pathname.match(/\/(dp|product)\/([A-Z0-9]{10})/i);
    if (asinMatch && asinMatch[2]) {
      const asin = asinMatch[2];
      return `https://www.amazon.com/product-reviews/${asin}`;
    }
  } catch (error) {
    console.error('Error in getReviewsPageUrl:', error);
  }
  
  return null;
}

// Function to fetch and parse reviews from a review page URL
async function fetchReviewsFromUrl(url) {
  try {
    console.log(`Fetching reviews from: ${url}`);
    
    // Fetch the reviews page
    const response = await fetch(url, {
      credentials: 'same-origin',
      headers: {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch reviews page: ${response.status}`);
    }
    
    const htmlText = await response.text();
    
    // Create a DOM parser to extract reviews from the HTML
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlText, 'text/html');
    
    const reviews = [];
    
    // Try various selectors for review content
    const reviewSelectors = [
      '[data-hook="review-body"] span',
      '.a-row.a-spacing-small.review-data span.review-text-content span',
      '.review-text-content span',
      '.a-expander-content p',
      '.cr-review-text',
      'div[data-hook="review"] .a-spacing-small'
    ];
    
    // Try each selector and collect reviews
    for (const selector of reviewSelectors) {
      const elements = doc.querySelectorAll(selector);
      
      elements.forEach(element => {
        const text = element.textContent.trim();
        if (text && text.length > 15 && !reviews.includes(text)) {
          reviews.push(text);
        }
      });
      
      // If we found reviews with this selector, stop trying others
      if (reviews.length > 0) {
        break;
      }
    }
    
    // If we have fewer than 5 reviews and there's a next page, fetch more
    if (reviews.length < 5) {
      // Look for "Next page" link
      const nextPageLink = doc.querySelector('li.a-last a');
      
      if (nextPageLink && nextPageLink.href) {
        try {
          // Make sure we have an absolute URL
          const nextPageUrl = new URL(nextPageLink.href, url).href;
          
          // Don't try to fetch if it's the same URL (to avoid infinite loops)
          if (nextPageUrl !== url) {
            const moreReviews = await fetchReviewsFromUrl(nextPageUrl);
            
            // Combine reviews but limit to 50 to prevent excessive requests
            return [...reviews, ...moreReviews].slice(0, 50);
          }
        } catch (error) {
          console.error('Error fetching additional reviews:', error);
        }
      }
    }
    
    return reviews;
  } catch (error) {
    console.error('Error in fetchReviewsFromUrl:', error);
    return [];
  }
} 