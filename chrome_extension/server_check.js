// Script to check if the backend server is running
// Use this to verify the API connection before using the extension

document.addEventListener('DOMContentLoaded', function() {
  const SERVER_URL = 'http://localhost:8080/api/detect-product';
  const statusElement = document.getElementById('server-status');
  const checkButton = document.getElementById('check-server');
  const troubleshootingSection = document.getElementById('troubleshooting');
  
  // Function to check if the server is running
  function checkServer() {
    statusElement.innerHTML = 'Checking server connection...';
    statusElement.className = 'status-info';
    
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
        statusElement.innerHTML = `❌ Server responded with status: ${response.status} ${response.statusText}`;
        statusElement.className = 'status-error';
        troubleshootingSection.style.display = 'block';
        console.error('Server error:', response.status, response.statusText);
      }
    })
    .catch(error => {
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