import os
import json
import asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from rich.console import Console
from rich.panel import Panel
import logging
from aiohttp import web
from aiohttp_cors import setup as setup_cors, ResourceOptions

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

# Load environment variables and setup Gemini
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

async def generate_with_timeout(client, prompt, timeout=20):
    """Generate content with a timeout"""
    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    # Convert chat-like format to plain text if needed
                    contents=prompt
                    )
                ),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        logger.error("LLM request timed out")
        return None
    except Exception as e:
        logger.error(f"Error in LLM request: {e}")
        return None

class SmartPurchaseAdvisorClient:
    def __init__(self):
        self.session = None
        self.tool_results = {}
        self.product_info = None
        self.category = None
        self.current_site = None

    async def process_product(self, product_data):
        """Process a product detected by the Chrome plugin"""
        try:
            self.product_info = product_data
            self.current_site = product_data.get("site", "Unknown")
            
            # 1. First, classify the product (still useful for context)
            classification_result = await self.classify_product(product_data["title"])
            self.category = classification_result
            
            # 2. Create the prompt for the LLM
            prompt = await self.craft_initial_prompt(product_data, self.category)
            
            # 3. Send the prompt to the LLM
            logger.info(f"Sending prompt to LLM for product review analysis: {product_data['title']}")
            tool_plan = await self.get_tool_invocation_plan(prompt)
            
            # Only log a summary of the tool plan, not the full content with reviews
            tool_plan_summary = {"tool_calls": []}
            if "tool_calls" in tool_plan:
                for tool_call in tool_plan["tool_calls"]:
                    tool_name = tool_call.get("tool_name", tool_call.get("name", "unknown"))
                    tool_plan_summary["tool_calls"].append({"tool_name": tool_name})
            
            print("Received tool plan from LLM:")
            print(f"Plan with {len(tool_plan.get('tool_calls', []))} tool calls: {[t.get('tool_name', t.get('name', 'unknown')) for t in tool_plan.get('tool_calls', [])]}")
            
            # 4. Execute the tool plan (run review analysis)
            results = await self.execute_tool_plan(tool_plan)

            # 5. Self-check for failures
            self_check = await self.check_tool_results(results)
            
            # Ensure self_check is in a usable format even if there was an error
            if not self_check or (isinstance(self_check, dict) and "error" in self_check):
                logger.warning("Self-check failed, using default values")
                self_check = {
                    "reliability_score": 0,
                    "reliability_level": "Very Low",
                    "issues": ["Self-check tool failed"],
                    "warnings": ["Results may not be reliable"],
                    "insights": []
                }
            
            # 6. Final reasoning with the LLM (generate sentiment and confidence score)
            final_response = await self.perform_final_reasoning(results, self_check)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing product: {e}")
            return {"error": str(e)}

    async def classify_product(self, title):
        """Classify the product using the classify_product tool"""
        logger.info(f"Classifying product: {title}")
        result = await self.session.call_tool("classify_product", arguments={"title": title})
        category = result.content[0].text
        logger.info(f"Product category detected: {category}")
        return category

    async def craft_initial_prompt(self, product_data, category):
        """Craft the initial prompt for the LLM"""
        # Define the example JSON outside the f-string to avoid nesting issues
        example_json = '''
        {
          "tool_calls": [
            {
              "tool_name": "review_summary_tool",
              "parameters": {
                "product": "Samsung Galaxy S23 Ultra",
                "site": "amazon.com",
                "num_reviews": 100000
              }
            },
            {
              "tool_name": "calculate_confidence_score",
              "parameters": {
                "sentiment_data": {"sentiment_score": 0.75, "review_count": 10, "pros": ["Great camera", "Fast performance"], "cons": ["Battery life", "Price"]}
              }
            },
            {
              "tool_name": "self_check_tool_results",
              "parameters": {
                "tools_results": [{"review_summary_tool": {"result": "..."}}, {"calculate_confidence_score": {"result": "..."}}]
              }
            }
          ]
        }
        '''
        
        prompt = f"""
        You are a Product Review Analyzer. Your task is to analyze product reviews and provide 
        a sentiment analysis with a confidence score to help shoppers make informed decisions.
        
        You will create a tool invocation plan to:
        1. Classify the product category
        2. Summarize reviews using sentiment analysis
        3. Calculate a confidence score based on the review sentiment
        4. Provide detailed reasoning and consistency checks
        
        You have access to these tools:
        - classify_product(title: str) - Classifies product category based on title using semantic similarity
        - review_summary_tool(product: str, site: str = None, reviews: list = None, num_reviews: int = 100000) - Analyzes product reviews and returns sentiment analysis
        - calculate_confidence_score(sentiment_data: dict) - Calculates a confidence score based on sentiment data
        - self_check_tool_results(tools_results: list) - Self-check sentinel reliability and highlight potential issues
        - show_reasoning(product_data: dict) - Show detailed explanation of sentiment analysis and confidence score calculation
        - calculate(expression: str) - Calculate sentiment metrics or confidence score components
        - verify(expression: str, expected: float) - Verify sentiment metrics or confidence score calculations
        - review_consistency_check(reviews_data: dict) - Check consistency of review sentiments and identify potential biases
        
        Typical workflow:
        1. First use review_summary_tool to get sentiment analysis of the product reviews
        2. Then pass those results to calculate_confidence_score to get a confidence score
        3. Use self_check_tool_results to validate the reliability of your analysis
        4. Use show_reasoning, calculate, verify and review_consistency_check for more detailed analysis
        
        Example tool invocation plan:
        ```json
{example_json}
        ```
        
        The confidence score should be calculated based on:
        - The overall sentiment polarity (positive, neutral, negative)
        - The consistency of reviews (are they all similar or varied?)
        - The quantity of reviews analyzed
        - The presence of specific, detailed pros and cons
        
        For each step, you will specify the tool to use and the input parameters.
        You must verify each tool's success and provide fallbacks if needed.
        
        Your response must be in JSON format with a structured "tool_calls" array that includes the EXACT function names as shown above.
        
        TASK: Create a tool invocation plan to analyze reviews and calculate confidence.
        """
        
        # Create JSON product data separately to avoid nested f-string
        product_json = json.dumps({
            "title": product_data["title"],
            "site": product_data.get("site", "Unknown"),
            "category": category,
            "price": product_data.get("price", "Unknown"),
            "url": product_data.get("url", "Unknown")
        })
        
        # Append the product JSON to the prompt
        prompt += f"PRODUCT: {product_json}"
        
        return prompt

    async def get_tool_invocation_plan(self, prompt):
        """Get the tool invocation plan from the LLM"""
        response = await generate_with_timeout(client, prompt)
        
        if not response or not response.text:
            logger.error("Failed to get response from LLM")
            return {"error": "Failed to get response from LLM"}
        
        try:
            # Extract JSON from potential markdown code blocks
            text = response.text
            if "```json" in text and "```" in text:
                # Extract content between ```json and the last ```
                json_text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                # Extract content between first ``` and last ```
                json_text = text.split("```", 1)[1].split("```", 1)[0].strip()
            else:
                json_text = text
                
            result = json.loads(json_text)
            logger.info("Successfully received tool invocation plan from LLM")
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {response.text}")
            return {"error": "Failed to parse LLM response"}

    async def execute_tool_plan(self, tool_plan):
        """Execute the tool invocation plan"""
        if "error" in tool_plan:
            return {"error": tool_plan["error"]}
        
        results = {}
        
        try:
            if "tool_calls" in tool_plan:
                for tool_call in tool_plan["tool_calls"]:
                    # Handle different formats of tool calls
                    tool_name = None
                    tool_input = {}
                    
                    # Extract tool name (could be under 'tool', 'tool_name', or 'function')
                    if "tool" in tool_call:
                        tool_name = tool_call["tool"]
                    elif "tool_name" in tool_call:
                        tool_name = tool_call["tool_name"]
                    elif "function" in tool_call and "name" in tool_call["function"]:
                        tool_name = tool_call["function"]["name"]
                    elif "name" in tool_call:
                        tool_name = tool_call["name"]
                    
                    # Extract tool input (could be under 'input', 'parameters', or 'arguments')
                    if "input" in tool_call:
                        tool_input = tool_call["input"]
                    elif "parameters" in tool_call:
                        tool_input = tool_call["parameters"]
                    elif "arguments" in tool_call:
                        tool_input = tool_call["arguments"]
                    elif "function" in tool_call and "arguments" in tool_call["function"]:
                        tool_input = tool_call["function"]["arguments"]
                    
                    logger.info(f"Executing tool: {tool_name}")
                    
                    # 1. Classify Product
                    if tool_name in ["classify_product"]:
                        title = tool_input.get("title", self.product_info["title"])
                        result = await self.session.call_tool("classify_product", arguments={
                            "title": title
                        })
                        results["classify_product"] = result.content[0].text
                        print(f"Classify Product Results: {results['classify_product']}")
                    
                    # 2. Review Summary Tool
                    elif tool_name in ["review_summary", "review_summary_tool"]:
                        # Map input parameter names
                        product = tool_input.get("product", tool_input.get("product_title", self.product_info["title"]))
                        site = tool_input.get("site", self.current_site)
                        num_reviews = tool_input.get("num_reviews", 1000)
                        
                        # Get reviews from product_data if available
                        reviews = self.product_info.get("reviews", [])
                        print(f"Reviews: {reviews}")
                        
                        result = await self.session.call_tool("review_summary_tool", arguments={
                            "product": product,
                            "site": site,
                            "reviews": reviews,
                            "num_reviews": num_reviews
                        })
                        results["review_summary_tool"] = json.loads(result.content[0].text)
                        print(f"Review Summary Tool Results: {results['review_summary_tool']}")
                    
                    # 3. Calculate Confidence Score
                    elif tool_name in ["calculate_confidence_score"]:
                        sentiment_data = tool_input.get("sentiment_data", {})
                        result = await self.session.call_tool("calculate_confidence_score", arguments={
                            "sentiment_data": sentiment_data
                        })
                        results["calculate_confidence_score"] = json.loads(result.content[0].text)
                        print(f"Calculate Confidence Score Results: {results['calculate_confidence_score']}")
                        
                    # 4. Self Check Tool Results
                    elif tool_name in ["self_check_tool_results"]:
                        tools_results = tool_input.get("tools_results", results)
                        result = await self.session.call_tool("self_check_tool_results", arguments={
                            "tools_results": tools_results
                        })
                        results["self_check_tool_results"] = json.loads(result.content[0].text)
                        print(f"Self Check Tool Results: {results['self_check_tool_results']}")

                    # 5. Show Reasoning
                    elif tool_name in ["show_reasoning"]:
                        product_data = tool_input.get("product_data", tool_input.get("steps", {}))
                        result = await self.session.call_tool("show_reasoning", arguments={
                            "product_data": product_data
                        })
                        results["show_reasoning"] = result.content[0].text
                        print(f"Show Reasoning Results: {results['show_reasoning']}")
                    
                    # 6. Calculate
                    elif tool_name in ["calculate"]:
                        expression = tool_input.get("expression", "")
                        result = await self.session.call_tool("calculate", arguments={
                            "expression": expression
                        })
                        results["calculate"] = result.content[0].text
                        print(f"Calculate Results: {results['calculate']}")

                    # 7. Verify
                    elif tool_name in ["verify"]:
                        expression = tool_input.get("expression", "")
                        expected = tool_input.get("expected", 0)
                        result = await self.session.call_tool("verify", arguments={
                            "expression": expression,
                            "expected": float(expected)
                        })
                        results["verify"] = result.content[0].text
                        print(f"Verify Results: {results['verify']}")

                    # 8. Review Consistency Check
                    elif tool_name in ["review_consistency_check"]:
                        reviews_data = tool_input.get("reviews_data", {})
                        result = await self.session.call_tool("review_consistency_check", arguments={
                            "reviews_data": reviews_data
                        })
                        try:
                            results["review_consistency_check"] = json.loads(result.content[0].text)
                        except:
                            results["review_consistency_check"] = result.content[0].text
                        print(f"Review Consistency Check Results: {results['review_consistency_check']}")   
                    
                    # Backward compatibility for "check_consistency"
                    elif tool_name in ["check_consistency"]:
                        steps = tool_input.get("steps", [])
                        result = await self.session.call_tool("check_consistency", arguments={"steps": steps})
                        try:
                            results["check_consistency"] = eval(result.content[0].text)
                        except:
                            results["check_consistency"] = result.content[0].text
                        print(f"Check Consistency Results: {results['check_consistency']}")
                    else:
                        logger.warning(f"Unknown tool name: {tool_name}")
                        
            self.tool_results = results
            return results
            
        except Exception as e:
            logger.error(f"Error executing tool plan: {e}")
            return {"error": str(e)}

    async def check_tool_results(self, results):
        """Check tool results for failures and suggest fallbacks"""
        try:
            result = await self.session.call_tool("self_check_tool_results", arguments={"tools_results": results})
            check_results = json.loads(result.content[0].text)
            
            # Create a summary of the results for logging
            check_summary = {
                "reliability_score": check_results.get("reliability_score", 0),
                "reliability_level": check_results.get("reliability_level", "Unknown"),
                "issues_count": len(check_results.get("issues", [])),
                "warnings_count": len(check_results.get("warnings", [])),
                "insights_count": len(check_results.get("insights", []))
            }
            
            logger.info(f"Tool self-check results: {check_summary}")
            return check_results
        except Exception as e:
            logger.error(f"Error checking tool results: {e}")
            return {"error": str(e), "reliability_level": "Low", "reliability_score": 0}
    
    async def perform_final_reasoning(self, results, self_check):
        """Perform final reasoning with the LLM using tool results"""
        # Create examples outside of f-string to avoid nesting issues
        examples = """
EXAMPLE OUTPUTS FROM TOOLS:

1. classify_product:
   Input: {"title": "Samsung Galaxy S23 Ultra"}
   Output: "smartphone"

2. review_summary_tool:
   Input: {"product": "Samsung Galaxy S23 Ultra", "site": "amazon.com", "num_reviews": 100000}
   Output: {
     "overall_sentiment": "Positive",
     "sentiment_score": 0.75,
     "pros": ["Great camera", "Fast performance", "Beautiful display"],
     "cons": ["Battery life could be better", "Expensive"],
     "review_count": 10,
     "source": "amazon.com"
   }

3. calculate_confidence_score:
   Input: {"sentiment_data": {"sentiment_score": 0.75, "review_count": 10, "pros": ["Great camera"], "cons": ["Battery life"]}}
   Output: {
     "confidence_score": 85,
     "explanation": "Confidence score of 85% calculated based on...",
     "confidence_level": "High Confidence: Reviews indicate this is likely a good product",
     "components": {
       "sentiment_component": 62.5,
       "review_count_component": 15.0,
       "specificity_component": 5.0,
       "balance_component": 2.5
     }
   }

4. self_check_tool_results:
   Input: {"tools_results": [{"review_summary_tool": {...}}, {"calculate_confidence_score": {...}}]}
   Output: {
     "reliability_score": 80,
     "reliability_level": "High",
     "review_count": 10,
     "sentiment_score": 0.75,
     "issues": [],
     "warnings": ["Limited sample size (10 reviews) may affect confidence"],
     "insights": ["Good sample size (10 reviews) for analysis"]
   }

5. show_reasoning:
   Input: {"product_data": {"product_name": "Samsung Galaxy S23 Ultra", "review_count": 10, "sentiment_score": 0.75}}
   Output: "Detailed explanation of sentiment analysis and confidence calculation..."

6. calculate:
   Input: {"expression": "(0.75 + 1) * 50"}
   Output: "87.5"

7. verify:
   Input: {"expression": "(0.75 + 1) * 50", "expected": 87.5}
   Output: "true"

8. review_consistency_check:
   Input: {"reviews_data": {"reviews": ["Great phone!", "Love the camera"], "sentiments": [0.8, 0.9]}}
   Output: {
     "review_count": 10,
     "avg_sentiment": 0.75,
     "std_deviation": 0.3,
     "positive_ratio": 0.7,
     "negative_ratio": 0.2,
     "neutral_ratio": 0.1,
     "bias_level": "Medium",
     "consistency_level": "High",
     "insights": ["Sample size is adequate for reliable sentiment analysis"]
   }
"""

        prompt = f"""
        You are a Product Review Analyzer. You have analyzed the reviews for a product 
        and now need to provide a structured summary with sentiment analysis and confidence score.
        
        The following tools were used in the analysis:
        - classify_product(title: str) - Classifies product category based on title using semantic similarity
        - review_summary_tool(product: str, site: str = None, reviews: list = None, num_reviews: int = 100000) - Analyzes product reviews and returns sentiment analysis
        - calculate_confidence_score(sentiment_data: dict) - Calculates a confidence score based on sentiment data
        - self_check_tool_results(tools_results: list) - Self-check sentinel reliability and highlight potential issues
        - show_reasoning(product_data: dict) - Show detailed explanation of sentiment analysis and confidence score calculation
        - calculate(expression: str) - Calculate sentiment metrics or confidence score components
        - verify(expression: str, expected: float) - Verify sentiment metrics or confidence score calculations
        - review_consistency_check(reviews_data: dict) - Check consistency of review sentiments and identify potential biases
        
{examples}

        Create a concise response including:
        1. Review sentiment summary with pros and cons
        2. Confidence score (on a scale of 0-100%) with explanation
        3. Key factors that influenced the confidence score
        4. Confidence level interpretation
        
        The confidence score considers:
        - Overall sentiment polarity (positive reviews increase confidence)
        - Consistency of sentiments across reviews
        - Quantity and quality of specific pros and cons mentioned
        - Number of reviews analyzed
        - Balance between pros and cons (having both is more reliable)
        
        If confidence score components are provided, include them to show
        how the score was calculated.
        
        Output must be in JSON format suitable for display in a Chrome extension sidebar with these fields:
        - title: product title
        - overall_sentiment: overall sentiment assessment (positive, negative, or neutral)
        - sentiment_score: numerical sentiment score
        - confidence_score: numerical confidence score (0-100)
        - confidence_level: text interpretation of the confidence score
        - pros: array of key pros from reviews
        - cons: array of key cons from reviews
        - confidence_explanation: explanation of how confidence was calculated
        - confidence_components: breakdown of score components if available (sentiment_component, review_count_component, specificity_component, balance_component)
        - review_count: number of reviews analyzed
        - reliability_score: score from self-check (0-100)
        - reliability_level: level from self-check (Low, Medium, High)
        - issues: array of critical issues found during self-check
        - warnings: array of warnings found during self-check
        - insights: array of insights found during self-check
        
        EXAMPLE FINAL OUTPUT:
        {{
          "title": "Samsung Galaxy S23 Ultra",
          "overall_sentiment": "Positive",
          "sentiment_score": 0.75,
          "confidence_score": 85,
          "confidence_level": "High Confidence",
          "pros": ["Great camera", "Fast performance", "Beautiful display"],
          "cons": ["Battery life could be better", "Expensive"],
          "confidence_explanation": "Confidence score of 85% calculated based on sentiment (62.5 points), review count (15.0 points), specificity (5.0 points), and balance (2.5 points)",
          "confidence_components": {{
            "sentiment_component": 62.5,
            "review_count_component": 15.0,
            "specificity_component": 5.0,
            "balance_component": 2.5
          }},
          "review_count": 10,
          "reliability_score": 80,
          "reliability_level": "High",
          "issues": [],
          "warnings": ["Limited sample size (10 reviews) may affect confidence"],
          "insights": ["Good sample size (10 reviews) for analysis"]
        }}
        
        TASK: Generate final sentiment analysis and confidence assessment
        """
        
        # Create redacted copies of data structures for logging
        product_info_redacted = self.product_info.copy()
        if "reviews" in product_info_redacted:
            product_info_redacted["reviews"] = f"[{len(product_info_redacted.get('reviews', []))} reviews hidden]"
        
        # Filter out detailed review content from results
        results_redacted = {}
        for tool_name, result in results.items():
            if tool_name == "review_summary_tool" and isinstance(result, dict):
                result_copy = result.copy()
                if "pros" in result_copy:
                    result_copy["pros"] = f"[{len(result_copy.get('pros', []))} pros hidden]"
                if "cons" in result_copy:
                    result_copy["cons"] = f"[{len(result_copy.get('cons', []))} cons hidden]"
                results_redacted[tool_name] = result_copy
            else:
                results_redacted[tool_name] = result
        
        # Add product info and results to prompt without logging full content
        prompt += f"""
        PRODUCT INFO: {json.dumps(product_info_redacted)}
        CATEGORY: {self.category}
        TOOL RESULTS: {json.dumps(results_redacted)}
        SELF CHECK: {json.dumps(self_check)}
        """
        
        response = await generate_with_timeout(client, prompt)
        
        if not response or not response.text:
            logger.error("Failed to get final response from LLM")
            return {"error": "Failed to get final response from LLM"}
        
        try:
            # Extract JSON from potential markdown code blocks
            text = response.text
            if "```json" in text and "```" in text:
                # Extract content between ```json and the last ```
                json_text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                # Extract content between first ``` and last ```
                json_text = text.split("```", 1)[1].split("```", 1)[0].strip()
            else:
                json_text = text
                
            result = json.loads(json_text)
            logger.info("Successfully received final analysis from LLM")
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse final LLM response as JSON: {response.text}")
            
            # Try to create a response directly from the tool results
            confidence_result = results.get("calculate_confidence_score", {})
            review_result = results.get("review_summary_tool", {})
            
            # Try to provide some fallback response
            return {
                "title": self.product_info["title"],
                "overall_sentiment": review_result.get("overall_sentiment", "Unknown"),
                "sentiment_score": review_result.get("sentiment_score", 0),
                "confidence_score": confidence_result.get("confidence_score", 0),
                "confidence_level": confidence_result.get("confidence_level", "Unknown confidence"),
                "pros": review_result.get("pros", ["No pros found"]),
                "cons": review_result.get("cons", ["No cons found"]),
                "confidence_explanation": confidence_result.get("explanation", "Could not calculate confidence score"),
                "confidence_components": confidence_result.get("components", {}),
                "review_count": review_result.get("review_count", 0),
                "reliability_score": 0,
                "reliability_level": "Unknown",
                "issues": [],
                "warnings": [],
                "insights": [],
                "error": "Failed to generate structured analysis"
            }

# API routes for Chrome extension communication
async def handle_product_detection(request):
    """Handle product detection from the Chrome extension"""
    try:
        data = await request.json()
        
        # Create a copy of data with redacted review content for logging
        log_data = data.copy()
        if "reviews" in log_data:
            review_count = len(log_data["reviews"]) if log_data["reviews"] else 0
            log_data["reviews"] = f"[{review_count} reviews - content hidden]"
        
        logger.info(f"Received product detection: {log_data}")
        
        client = request.app['client']
        result = await client.process_product(data)
        
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Error handling product detection: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)

# Simple health check endpoint
async def health_check(request):
    return web.json_response({"status": "ok"})

async def start_server_with_mcp():
    """Start the API server with MCP integration"""
    try:
        console.print(Panel("Smart Purchase Advisor API Server", border_style="cyan"))
        
        # Create the MCP client
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            encoding_error_handler="replace"
        )
        
        # Start MCP client
        logger.info("Starting MCP client...")
        app = web.Application()
        
        # Setup routes
        app.router.add_post('/api/detect-product', handle_product_detection)
        app.router.add_get('/', health_check)
        
        # Setup CORS
        cors = setup_cors(app, defaults={
            "*": ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods=["POST", "GET", "OPTIONS"]
            )
        })
        
        # Apply CORS to all routes
        for route in list(app.router.routes()):
            cors.add(route)
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Create client instance and set session
                client = SmartPurchaseAdvisorClient()
                client.session = session
                
                # Store client in app state
                app['client'] = client
                
                # Start the web server
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, 'localhost', 8080)
                await site.start()
                
                console.print(Panel(f"Server started at http://localhost:8080", border_style="green"))
                
                # Keep the server running until interrupted
                try:
                    # Run forever
                    while True:
                        await asyncio.sleep(3600)  # Sleep for an hour
                except asyncio.CancelledError:
                    pass
                finally:
                    await runner.cleanup()
    
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")

async def main():
    """Test function for direct execution"""
    try:
        console.print(Panel("Smart Purchase Advisor", border_style="cyan"))

        server_params = StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            encoding_error_handler="replace"
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Create client instance and set session
                client = SmartPurchaseAdvisorClient()
                client.session = session
                
                # Example product data for testing
                example_product = {
                    "title": "Samsung Galaxy S23 Ultra",
                    "site": "amazon.com",
                    "price": "$1199.99",
                    "url": "https://www.amazon.com/samsung-galaxy-s23-ultra"
                }
                
                console.print(Panel(f"Processing product: {example_product['title']}", border_style="green"))
                
                
                # Process the product (normal flow)
                result = await client.process_product(example_product)
                
                # Display the result
                console.print(Panel(json.dumps(result, indent=2), border_style="cyan", title="Analysis Result"))

    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Purchase Advisor")
    parser.add_argument("--server", action="store_true", help="Start in server mode for Chrome extension")
    parser.add_argument("--test", action="store_true", help="Run a test analysis with example product")
    
    args = parser.parse_args()
    
    if args.server:
        # Start server for Chrome extension
        print("Starting server for Chrome extension")
        asyncio.run(start_server_with_mcp())
    elif args.test:
        # Run test with example product
        print("Running test with example product")
        asyncio.run(main())
    else:
        # Default to server mode
        print("Starting server in default mode")
        asyncio.run(start_server_with_mcp()) 