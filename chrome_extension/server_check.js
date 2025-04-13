/**
 * Server Connection Test Script for Smart Purchase Advisor
 * 
 * This script tests if the Python backend API server is running and accessible.
 * It sends a test request to the server and displays the result to the user,
 * including troubleshooting tips if the connection fails.
 * 
 * The extension requires this server to be running to analyze product reviews.
 */

document.addEventListener('DOMContentLoaded', function() {
  const SERVER_URL = 'http://localhost:8080/api/detect-product';
  const statusElement = document.getElementById('server-status');
  const checkButton = document.getElementById('check-server');
  const troubleshootingSection = document.getElementById('troubleshooting');
  
  /**
   * Test the server connection by sending a minimal test request
   * Displays the result with appropriate styling and shows troubleshooting
   * guidance if the connection fails
   */
  function checkServer() {
    statusElement.innerHTML = 'Checking server connection...';
    statusElement.className = 'status-info';
    
    // Send a minimal test request to the server
    fetch(SERVER_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        productTitle: "Test Product",
        site: "amazon",
        reviews: ["This is a test review for checking server connection"]
      })
    })
    .then(response => {
      if (response.ok) {
        return response.json().then(data => {
          // Server responded successfully with JSON
          statusElement.innerHTML = `✅ Server is running and responding correctly`;
          statusElement.className = 'status-success';
          troubleshootingSection.style.display = 'none';
          console.log('Server response:', data);
          return data;
        }).catch(err => {
          // Response was received but JSON parsing failed
          statusElement.innerHTML = `⚠️ Server responded but returned invalid JSON`;
          statusElement.className = 'status-warning';
          troubleshootingSection.style.display = 'block';
          console.error('JSON parsing error:', err);
        });
      } else {
        // Server responded with an error status
        statusElement.innerHTML = `❌ Server responded with status: ${response.status} ${response.statusText}`;
        statusElement.className = 'status-error';
        troubleshootingSection.style.display = 'block';
        console.error('Server error:', response.status, response.statusText);
      }
    })
    .catch(error => {
      // Network error - server might not be running
      statusElement.innerHTML = `❌ Could not connect to server: ${error.message}`;
      statusElement.className = 'status-error';
      troubleshootingSection.style.display = 'block';
      console.error('Connection error:', error);
    });
  }
  
  // Add click event to the check button
  checkButton.addEventListener('click', checkServer);
  
  // Check the server on page load
  checkServer();
}); 