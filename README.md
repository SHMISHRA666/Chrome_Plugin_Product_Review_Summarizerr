# Review Summarizer

A Chrome extension that analyzes and summarizes product reviews from popular e-commerce platforms.

## Overview

Review Summarizer analyzes product reviews on popular e-commerce sites to provide:

- Sentiment analysis of reviews (positive, negative, neutral)
- Key points extraction from reviews
- Pros and cons summarization
- Identification of common issues and highlights
- Overall review score interpretation

The extension summarizes in the plugin box to supported e-commerce sites that shows detailed review analysis to help users make informed purchase decisions.

**Watch the demo video:** [Review Summarizer in action](https://youtu.be/Z-BzPWn6Gm8)

## Architecture

The Review Summarizer consists of three main components:

1. **Chrome Extension**: Detects products on e-commerce sites and displays review summaries
2. **MCP Client**: Python application that processes review data and communicates with the LLM
3. **MCP Server**: Service that provides tools for review analysis and summarization

### Flow Diagram

```
Chrome Plugin → MCP Client → LLM → MCP Server → Results → Chrome Plugin
```

1. Chrome Plugin detects a product on an e-commerce site
2. MCP Client sends product reviews to LLM for analysis
3. LLM generates a tool invocation plan for review processing
4. MCP Server executes tools (sentiment analysis, key point extraction, etc.)
5. Results are structured and sent back to the Chrome Plugin
6. Chrome Plugin displays the review summary in the sidebar

## Project Structure

```
├── chrome_extension/        # Chrome extension files
│   ├── css/                 # Styles
│   │   └── sidebar.css      # Sidebar styling
│   │
│   ├── images/              # Icons and images
│   ├── js/                  # Extension scripts
│   │   ├── background.js    # Background service worker
│   │   ├── content.js       # Content script for product detection
│   │   └── popup.js         # Popup script
│   ├── manifest.json        # Extension manifest
│   └── popup.html           # Popup HTML
│
├── mcp_client.py            # Main MCP client that communicates with LLM
├── mcp_client_withalltools.py  # Alternative client with all tools included
├── mcp_client_mock_extra.py    # Mock client for testing purposes
├── mcp_server.py            # MCP server with tools for review analysis
├── mcp_server_mock.py       # Mock server for testing purposes
├── cot_tools_example.py     # Example implementation for MCP tools
├── cot_main_example.py      # Example implementation for LLM calling
├── main.py                  # Entry point script
├── pyproject.toml           # Project configuration
├── requirements.txt         # Project dependencies
└── .env                     # Environment variables (API keys)
```

## Features

- **Product Detection**: Automatically detects products on supported e-commerce sites
- **Sentiment Analysis**: Categorizes reviews into positive, negative, and neutral
- **Key Point Extraction**: Identifies important points mentioned across reviews
- **Pros and Cons Summary**: Summarizes strengths and weaknesses from reviews
- **Common Issues Identification**: Highlights frequently mentioned problems
- **Review Reliability Assessment**: Evaluates the trustworthiness of reviews
- **Self-checks and Fallbacks**: Handles failures gracefully with fallback mechanisms

## Supported E-commerce Sites

- Amazon.in / Amazon.com

## Setup and Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Chrome browser

### Backend Setup

1. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. Start the MCP client server:
   ```bash
   python mcp_client.py
   ```

### Chrome Extension Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select the `chrome_extension` directory
4. The extension should now appear in your Chrome toolbar

## Usage

1. Visit a supported e-commerce site (e.g., Amazon, Flipkart)
2. Browse to a product page
3. The extension will automatically detect the product and display review analysis in a sidebar
4. Use the extension popup to toggle features or manually trigger analysis

## Development

### Adding New E-commerce Sites

To add support for a new e-commerce site:

1. Add the site to the `SUPPORTED_SITES` array in `background.js`
2. Add selectors for the site in the `SITE_SELECTORS` object in `content.js`
3. Add the site to the host permissions in `manifest.json`

### Adding New Analysis Tools

To add a new analysis tool:

1. Create a new tool function in `mcp_server.py` using the `@mcp.tool()` decorator
2. Update the `execute_tool_plan` function in `mcp_client.py` to handle the new tool
3. Update the LLM prompt to include the new tool in `craft_initial_prompt`

## Limitations
1. This is not the best version of the tool
2. Currently only works with Amazon product pages
3. Limited to processing reviews visible on the current page
4. Navigation to the review page can retrieve more reviews, but not all at once
5. Web scraping functionality could be improved and optimized
6. Review analysis is limited by the quality and quantity of available reviews