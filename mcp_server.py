from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import requests
from bs4 import BeautifulSoup
import json
import re
from textblob import TextBlob
import datetime
import asyncio
import aiohttp
import math
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import random
import traceback

console = Console()
mcp = FastMCP("SmartPurchaseAdvisor")

@mcp.tool()
def classify_product(title: str) -> TextContent:
    """Classify product category based on title using semantic similarity"""
    console.print("[blue]FUNCTION CALL:[/blue] classify_product()")
    console.print(f"[blue]Product Title:[/blue] {title}")
    
    # Define categories with representative examples
    categories = {
        "smartphone": "mobile phone electronic device smartphone iphone android samsung pixel",
        "laptop": "computer portable laptop notebook macbook thinkpad chromebook",
        "headphones": "audio headphones earphones earbuds wireless bluetooth airpods",
        "television": "tv television display screen led lcd oled smart tv",
        "camera": "camera photography dslr mirrorless digital canon nikon sony",
        "clothing": "clothing apparel fashion shirt t-shirt jeans pants dress jacket",
        "shoes": "footwear shoes sneakers boots sandals athletic casual dress"
    }
    
    # Load pre-trained sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Encode the product title
    title_embedding = model.encode(title.lower())
    
    # Encode category descriptions and calculate similarities
    similarities = {}
    for category, description in categories.items():
        category_embedding = model.encode(description)
        similarity = 1 - cosine(title_embedding, category_embedding)
        similarities[category] = similarity
    
    # Find the most similar category
    best_category = max(similarities.items(), key=lambda x: x[1])
    confidence = best_category[1]
    
    # Set a threshold for unknown categories
    if confidence < 0.4:
        console.print(f"[yellow]Low confidence ({confidence:.2f}), defaulting to 'other'[/yellow]")
        return TextContent(type="text", text="other")
    
    console.print(f"[green]Detected Category:[/green] {best_category[0]} [cyan](confidence: {confidence:.2f})[/cyan]")
    return TextContent(type="text", text=best_category[0])


@mcp.tool()
async def review_summary_tool(product: str, site: str = None, reviews: list = None, num_reviews: int = 100000) -> TextContent:
    """Summarize reviews using sentiment analysis"""
    console.print("[blue]FUNCTION CALL:[/blue] review_summary_tool()")
    console.print(f"[blue]Product:[/blue] {product} | [blue]Site:[/blue] {site} | [blue]Reviews received:[/blue] {len(reviews) if reviews else 0}")
    
    try:
        # Process actual reviews if provided
        if reviews and len(reviews) > 0:
            # Remove detailed review logging
            # console.print(f"Processing {len(reviews)} actual reviews from {site}")
            
            # Calculate sentiment for each review using TextBlob
            sentiments = []
            for review_text in reviews:
                blob = TextBlob(review_text)
                # TextBlob sentiment is between -1 (negative) and 1 (positive)
                sentiment_value = blob.sentiment.polarity
                sentiments.append(sentiment_value)
                
            # Calculate overall sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_label = "Positive" if avg_sentiment > 0.3 else "Negative" if avg_sentiment < -0.3 else "Neutral"
            
            # Extract common pros and cons
            pros = []
            cons = []
            
            for i, review in enumerate(reviews):
                if sentiments[i] > 0.3:
                    # This is a positive review
                    blob = TextBlob(review)
                    for sentence in blob.sentences:
                        if sentence.sentiment.polarity > 0.3:
                            pros.append(str(sentence))
                elif sentiments[i] < -0.3:
                    # This is a negative review
                    blob = TextBlob(review)
                    for sentence in blob.sentences:
                        if sentence.sentiment.polarity < -0.3:
                            cons.append(str(sentence))
            
            # Remove duplicates and limit to top items
            pros = list(set(pros))[:5]  # Top 5 pros
            cons = list(set(cons))[:5]  # Top 5 cons
            
            # Construct the response
            result = {
                "overall_sentiment": sentiment_label,
                "sentiment_score": round(avg_sentiment, 2),
                "pros": pros,
                "cons": cons,
                "review_count": len(reviews),
                "source": site or "multiple sources"
            }
            
        else:
            # No reviews provided
            console.print("[red]No reviews provided to analyze[/red]")
            return TextContent(
                type="text",
                text=json.dumps({
                    "error": "No reviews found for this product",
                    "overall_sentiment": "Unknown",
                    "sentiment_score": 0,
                    "pros": [],
                    "cons": [],
                    "review_count": 0,
                    "source": site or "unknown"
                })
            )
        
        # Remove detailed review printout
        console.print(f"[green]Review Summary for {product}:[/green] {result['overall_sentiment']} ({result['sentiment_score']}), {len(pros)} pros, {len(cons)} cons")
        
        # Remove printing individual pros and cons
        # console.print(f"[cyan]Overall Sentiment:[/cyan] {result['overall_sentiment']} ({result['sentiment_score']})")
        # console.print("[cyan]Pros:[/cyan]")
        # for pro in result["pros"]:
        #     console.print(f"- {pro}")
        # console.print("[cyan]Cons:[/cyan]")
        # for con in result["cons"]:
        #     console.print(f"- {con}")
        
        return TextContent(
            type="text",
            text=json.dumps(result)
        )
    except Exception as e:
        error_msg = f"Error in review summary: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        return TextContent(
            type="text",
            text=json.dumps({"error": error_msg})
        )

@mcp.tool()
def self_check_tool_results(tools_results: list) -> TextContent:
    """Self-check sentinel reliability and highlight potential issues in review analysis"""
    console.print("[blue]FUNCTION CALL:[/blue] self_check_tool_results()")
    
    try:
        # Check if tools_results is empty or invalid
        if not tools_results or not isinstance(tools_results, list):
            return TextContent(
                type="text",
                text="Error: Invalid or empty tools_results provided"
            )
        
        # Initialize collection for issues, warnings, and insights
        issues = []
        warnings = []
        insights = []
        
        # Extract critical information from tools_results
        review_count = 0
        avg_sentiment = None
        pros_count = 0
        cons_count = 0
        sentiment_score = None
        confidence_score = None
        
        # Analyze each tool result
        for i, result in enumerate(tools_results):
            # Skip if result is None or empty
            if not result:
                warnings.append(f"Tool {i+1} returned empty result")
                continue
                
            # Try to parse JSON if result is a string
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    # If it's not JSON, extract key metrics using regex
                    sentiment_match = re.search(r"sentiment[\s_:]+([+-]?\d+\.?\d*)", result, re.IGNORECASE)
                    if sentiment_match:
                        sentiment_score = float(sentiment_match.group(1))
                    
                    review_match = re.search(r"reviews?[\s_:]+(\d+)", result, re.IGNORECASE)
                    if review_match:
                        review_count = max(review_count, int(review_match.group(1)))
                    
                    confidence_match = re.search(r"confidence[\s_:]+([+-]?\d+\.?\d*)", result, re.IGNORECASE)
                    if confidence_match:
                        confidence_score = float(confidence_match.group(1))
                    
                    continue
            
            # Extract key metrics from parsed result
            if isinstance(result, dict):
                # Review count
                if "review_count" in result:
                    review_count = max(review_count, result["review_count"])
                elif "reviews" in result and isinstance(result["reviews"], list):
                    review_count = max(review_count, len(result["reviews"]))
                
                # Sentiment
                if "avg_sentiment" in result:
                    avg_sentiment = result["avg_sentiment"]
                elif "sentiment_score" in result:
                    sentiment_score = result["sentiment_score"]
                elif "sentiment" in result and isinstance(result["sentiment"], (int, float)):
                    sentiment_score = result["sentiment"]
                
                # Pros & Cons
                if "pros" in result and isinstance(result["pros"], list):
                    pros_count = len(result["pros"])
                if "cons" in result and isinstance(result["cons"], list):
                    cons_count = len(result["cons"])
                
                # Confidence
                if "confidence_score" in result:
                    confidence_score = result["confidence_score"]
        
        # Check for critical issues in review analysis
        
        # 1. Review count reliability
        if review_count == 0:
            issues.append("No reviews found for analysis")
        elif review_count < 3:
            warnings.append(f"Very small sample size ({review_count} reviews) reduces reliability")
        elif review_count < 10:
            warnings.append(f"Limited sample size ({review_count} reviews) may affect confidence")
        else:
            insights.append(f"Good sample size ({review_count} reviews) for analysis")
            
        # 2. Check sentiment reliability
        if sentiment_score is not None:
            if sentiment_score > 0.8:
                warnings.append(f"Extremely positive sentiment ({sentiment_score:.2f}) may indicate biased reviews")
            elif sentiment_score < -0.8:
                warnings.append(f"Extremely negative sentiment ({sentiment_score:.2f}) may indicate biased reviews")
            elif abs(sentiment_score) < 0.1:
                insights.append(f"Neutral sentiment ({sentiment_score:.2f}) indicates mixed opinions")
        elif avg_sentiment is not None:
            if avg_sentiment > 0.8:
                warnings.append(f"Extremely positive sentiment ({avg_sentiment:.2f}) may indicate biased reviews")
            elif avg_sentiment < -0.8:
                warnings.append(f"Extremely negative sentiment ({avg_sentiment:.2f}) may indicate biased reviews")
            elif abs(avg_sentiment) < 0.1:
                insights.append(f"Neutral sentiment ({avg_sentiment:.2f}) indicates mixed opinions")
        else:
            warnings.append("No sentiment score found in analysis")
            
        # 3. Check pros/cons balance
        if pros_count > 0 or cons_count > 0:
            if pros_count == 0:
                warnings.append("No pros identified, which is unusual even for negative products")
            elif cons_count == 0:
                warnings.append("No cons identified, which is unusual even for positive products")
            elif pros_count >= 3 * cons_count:
                warnings.append(f"Disproportionate pros ({pros_count}) compared to cons ({cons_count})")
            elif cons_count >= 3 * pros_count:
                warnings.append(f"Disproportionate cons ({cons_count}) compared to pros ({pros_count})")
            else:
                insights.append(f"Balanced pros ({pros_count}) and cons ({cons_count})")
                
        # 4. Check confidence score
        if confidence_score is not None:
            if confidence_score < 30:
                warnings.append(f"Very low confidence score ({confidence_score}) indicates unreliable analysis")
            elif confidence_score < 50:
                warnings.append(f"Low confidence score ({confidence_score}) suggests limited reliability")
            elif confidence_score > 90:
                insights.append(f"High confidence score ({confidence_score}) indicates reliable analysis")
        else:
            warnings.append("No confidence score found in analysis")
        
        # Create a summary report
        console.print("\n[bold cyan]Self-Check Analysis Summary[/bold cyan]")
        
        if issues:
            console.print(Panel(
                "\n".join(f"[red]• {issue}[/red]" for issue in issues),
                title="Critical Issues",
                border_style="red"
            ))
        
        if warnings:
            console.print(Panel(
                "\n".join(f"[yellow]• {warning}[/yellow]" for warning in warnings),
                title="Warnings",
                border_style="yellow"
            ))
        
        if insights:
            console.print(Panel(
                "\n".join(f"[blue]• {insight}[/blue]" for insight in insights),
                title="Analysis Insights",
                border_style="blue"
            ))
        
        # Calculate reliability score
        total_checks = 4  # Number of check categories
        issue_penalty = 25 * len(issues)
        warning_penalty = 10 * len(warnings)
        insight_bonus = 5 * len(insights)
        
        base_score = 70  # Starting score
        reliability_score = min(100, max(0, base_score - issue_penalty - warning_penalty + insight_bonus))
        
        reliability_level = "Low"
        if reliability_score >= 80:
            reliability_level = "High"
        elif reliability_score >= 60:
            reliability_level = "Medium"
        
        console.print(Panel(
            f"[bold]Reliability Score: {reliability_score:.0f}/100[/bold]\n" +
            f"Reliability Level: {reliability_level}\n" +
            f"Critical Issues: {len(issues)}\n" +
            f"Warnings: {len(warnings)}\n" +
            f"Insights: {len(insights)}",
            title="Reliability Summary",
            border_style="green" if reliability_score > 80 else "yellow" if reliability_score > 60 else "red"
        ))
        
        return TextContent(
            type="text",
            text=json.dumps({
                "reliability_score": reliability_score,
                "reliability_level": reliability_level,
                "review_count": review_count,
                "sentiment_score": sentiment_score if sentiment_score is not None else avg_sentiment,
                "issues": issues,
                "warnings": warnings,
                "insights": insights
            })
        )
    except Exception as e:
        console.print(f"[red]Error in self-check: {str(e)}[/red]")
        traceback.print_exc()
        return TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )

# Adding tools from cot_tools.py

@mcp.tool()
def show_reasoning(product_data: dict) -> TextContent:
    """
    Provides detailed explanation of sentiment analysis and confidence score calculation
    for a given product based on review data.
    """
    console.print("[blue]FUNCTION CALL:[/blue] show_reasoning()")
    
    try:
        # Check if product_data contains necessary information
        if not product_data:
            return TextContent(
                type="text",
                text="Error: No product data provided"
            )
        
        # Extract key data points
        product_name = product_data.get("product_name", "Unknown Product")
        review_count = product_data.get("review_count", 0)
        avg_sentiment = product_data.get("sentiment_score", None)
        confidence_score = product_data.get("confidence_score", None)
        reviews = product_data.get("reviews", [])
        pros = product_data.get("pros", [])
        cons = product_data.get("cons", [])
        
        # Create the reasoning table but don't print details
        reasoning_table = Table(title=f"Analysis Explanation for {product_name}")
        reasoning_table.add_column("Component", style="cyan")
        reasoning_table.add_column("Details", style="white")
        
        # 1. Data Overview Section
        data_overview = []
        data_overview.append(f"Product: {product_name}")
        data_overview.append(f"Reviews analyzed: {review_count}")
        if avg_sentiment is not None:
            sentiment_label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
            data_overview.append(f"Overall sentiment: {sentiment_label} ({avg_sentiment:.2f})")
        if confidence_score is not None:
            confidence_label = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 40 else "Low"
            data_overview.append(f"Confidence level: {confidence_label} ({confidence_score:.0f}/100)")
        reasoning_table.add_row("Data Overview", "\n".join(data_overview))
        
        # 2. Sentiment Analysis Explanation
        if avg_sentiment is not None:
            if avg_sentiment > 0.6:
                sentiment_explanation = "The product has strongly positive sentiment based on user reviews."
            elif avg_sentiment > 0.2:
                sentiment_explanation = "The product has moderately positive sentiment based on user reviews."
            elif avg_sentiment >= -0.2:
                sentiment_explanation = "The product has neutral sentiment with mixed user feedback."
            elif avg_sentiment >= -0.6:
                sentiment_explanation = "The product has moderately negative sentiment based on user reviews."
            else:
                sentiment_explanation = "The product has strongly negative sentiment based on user reviews."
            
            reasoning_table.add_row("Sentiment Analysis", sentiment_explanation)
        
        # 3. Confidence Score Explanation
        if confidence_score is not None:
            confidence_explanation = f"Confidence score: {confidence_score}/100 ({confidence_score >= 70 and 'High' or confidence_score >= 40 and 'Medium' or 'Low'})"
            reasoning_table.add_row("Confidence Score", confidence_explanation)
        
        # Skip detailed review analysis section to avoid printing review content
        
        # 5. Pros and Cons Analysis - just show counts
        if pros or cons:
            pros_cons_analysis = f"Pros: {len(pros)}, Cons: {len(cons)}"
            reasoning_table.add_row("Pros & Cons Summary", pros_cons_analysis)
        
        # 6. Final Summary - simplified
        if avg_sentiment is not None and confidence_score is not None:
            if confidence_score >= 70:
                if avg_sentiment > 0.2:
                    final_summary = "Recommendation: Product is recommended with high confidence."
                elif avg_sentiment < -0.2:
                    final_summary = "Recommendation: Product should be avoided with high confidence."
                else:
                    final_summary = "Recommendation: Consider alternatives with high confidence."
            else:
                final_summary = "Recommendation: Additional research recommended."
            
            reasoning_table.add_row("Final Assessment", final_summary)
        
        # Print a concise summary instead of the full reasoning table
        console.print(f"[cyan]Analysis for {product_name}:[/cyan] {review_count} reviews, Sentiment: {avg_sentiment:.2f}, Confidence: {confidence_score:.0f}/100")
        
        # Return the full table as string for client display, but don't print it to console
        return TextContent(
            type="text",
            text=str(reasoning_table)
        )
    except Exception as e:
        error_msg = f"Error in reasoning explanation: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        traceback.print_exc()
        return TextContent(
            type="text",
            text=error_msg
        )

@mcp.tool()
def calculate(expression: str) -> TextContent:
    """Calculate sentiment metrics or confidence score components"""
    console.print("[blue]FUNCTION CALL:[/blue] calculate()")
    console.print(f"[blue]Expression:[/blue] {expression}")
    try:
        result = eval(expression)
        console.print(f"[green]Result:[/green] {result}")
        return TextContent(
            type="text",
            text=str(result)
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        return TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )

@mcp.tool()
def verify(expression: str, expected: float) -> TextContent:
    """Verify sentiment metrics or confidence score calculations"""
    console.print("[blue]FUNCTION CALL:[/blue] verify()")
    console.print(f"[blue]Verifying:[/blue] {expression} = {expected}")
    try:
        actual = float(eval(expression))
        is_correct = abs(actual - float(expected)) < 1e-10
        
        if is_correct:
            console.print(f"[green]✓ Correct! {expression} = {expected}[/green]")
        else:
            console.print(f"[red]✗ Incorrect! {expression} should be {actual}, got {expected}[/red]")
            
        return TextContent(
            type="text",
            text=str(is_correct)
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        return TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )

@mcp.tool()
def review_consistency_check(reviews_data: dict) -> TextContent:
    """Check consistency of review sentiments and identify potential biases"""
    console.print("[blue]FUNCTION CALL:[/blue] review_consistency_check()")
    
    try:
        # Extract review data
        all_reviews = reviews_data.get("reviews", [])
        sentiments = reviews_data.get("sentiments", [])
        
        if not all_reviews or not sentiments or len(all_reviews) != len(sentiments):
            return TextContent(
                type="text",
                text=json.dumps({
                    "error": "Invalid review data provided for consistency check"
                })
            )
        
        # Create a table for analysis, but don't print it directly
        table = Table(
            title="Review Sentiment Consistency Analysis",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Interpretation", style="yellow")

        # Calculate consistency metrics
        num_reviews = len(all_reviews)
        avg_sentiment = sum(sentiments) / num_reviews
        
        # Calculate variance (spread) of sentiments
        variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / num_reviews
        std_deviation = math.sqrt(variance)
        
        # Count reviews by sentiment category
        positive_count = sum(1 for s in sentiments if s > 0.3)
        negative_count = sum(1 for s in sentiments if s < -0.3)
        neutral_count = num_reviews - positive_count - negative_count
        
        # Calculate ratios
        positive_ratio = positive_count / num_reviews
        negative_ratio = negative_count / num_reviews
        neutral_ratio = neutral_count / num_reviews
        
        # Check for potential review bias
        bias_level = "Low"
        bias_explanation = "Reviews appear balanced"
        
        if positive_ratio > 0.9:
            bias_level = "High"
            bias_explanation = "Reviews are overwhelmingly positive, which might indicate bias"
        elif negative_ratio > 0.9:
            bias_level = "High"
            bias_explanation = "Reviews are overwhelmingly negative, which might indicate bias"
        elif positive_ratio > 0.8:
            bias_level = "Medium"
            bias_explanation = "Reviews skew heavily positive"
        elif negative_ratio > 0.8:
            bias_level = "Medium"
            bias_explanation = "Reviews skew heavily negative"
        
        # Check for review consistency
        consistency_level = "High"
        if std_deviation > 0.6:
            consistency_level = "Low"
            consistency_explanation = "Sentiments vary widely across reviews"
        elif std_deviation > 0.4:
            consistency_level = "Medium"
            consistency_explanation = "Some variation in review sentiments"
        else:
            consistency_explanation = "Reviews are highly consistent in sentiment"
        
        # Add rows to table but we won't print it directly
        table.add_row("Number of Reviews", str(num_reviews), "Sample size assessment")
        table.add_row("Average Sentiment", f"{avg_sentiment:.2f}", f"{'Positive' if avg_sentiment > 0.3 else 'Negative' if avg_sentiment < -0.3 else 'Neutral'}")
        table.add_row("Sentiment Deviation", f"{std_deviation:.2f}", f"{consistency_level} consistency")
        table.add_row("Positive Reviews", f"{positive_count} ({positive_ratio:.0%})", f"{'Dominant' if positive_ratio > 0.5 else 'Secondary'} sentiment")
        table.add_row("Negative Reviews", f"{negative_count} ({negative_ratio:.0%})", f"{'Dominant' if negative_ratio > 0.5 else 'Secondary'} sentiment")
        table.add_row("Neutral Reviews", f"{neutral_count} ({neutral_ratio:.0%})", "Baseline sentiment")
        table.add_row("Bias Assessment", bias_level, bias_explanation)
        table.add_row("Consistency Level", consistency_level, consistency_explanation)
        
        # Replace detailed table output with a simple summary
        # console.print("\n[bold cyan]Review Consistency Analysis[/bold cyan]")
        # console.print(table)
        console.print(f"[cyan]Review Consistency Analysis:[/cyan] {num_reviews} reviews, Avg: {avg_sentiment:.2f}, Bias: {bias_level}, Consistency: {consistency_level}")
        
        # Additional insights
        insights = []
        
        if num_reviews < 5:
            insights.append("Sample size is very small for reliable sentiment analysis")
        elif num_reviews < 10:
            insights.append("Limited sample size may affect confidence in analysis")
        
        if abs(avg_sentiment) < 0.1:
            insights.append("Overall sentiment is very neutral, suggesting mixed opinions")
        
        if neutral_ratio > 0.5:
            insights.append("High proportion of neutral reviews suggests indecision or lukewarm reception")
        
        if std_deviation < 0.2 and num_reviews > 10:
            insights.append("Unusually consistent sentiments might suggest artificial reviews")
        
        # Skip printing insights panel
        # if insights:
        #     console.print(Panel(
        #         "\n".join(f"[blue]• {insight}[/blue]" for insight in insights),
        #         title="Analysis Insights",
        #         border_style="blue"
        #     ))
        
        return TextContent(
            type="text",
            text=json.dumps({
                "review_count": num_reviews,
                "avg_sentiment": round(avg_sentiment, 2),
                "std_deviation": round(std_deviation, 2),
                "positive_ratio": round(positive_ratio, 2),
                "negative_ratio": round(negative_ratio, 2),
                "neutral_ratio": round(neutral_ratio, 2),
                "bias_level": bias_level,
                "consistency_level": consistency_level,
                "insights": insights
            })
        )
    except Exception as e:
        console.print(f"[red]Error in review consistency check: {str(e)}[/red]")
        return TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )

@mcp.tool()
def calculate_confidence_score(sentiment_data: dict) -> TextContent:
    """Calculate a confidence score based on sentiment analysis of reviews"""
    console.print("[blue]FUNCTION CALL:[/blue] calculate_confidence_score()")
    # Only print the count of pros and cons, not their contents
    console.print(f"[blue]Sentiment Data:[/blue] score: {sentiment_data.get('sentiment_score', 0)}, reviews: {sentiment_data.get('review_count', 0)}, pros: {len(sentiment_data.get('pros', []))}, cons: {len(sentiment_data.get('cons', []))}")
    
    try:
        # Extract key metrics from sentiment data
        sentiment_score = float(sentiment_data.get("sentiment_score", 0))
        review_count = int(sentiment_data.get("review_count", 0))
        pros = sentiment_data.get("pros", [])
        cons = sentiment_data.get("cons", [])
        
        # Base score from sentiment (convert -1 to 1 scale to 0-100)
        base_score = (sentiment_score + 1) * 50
        
        # Adjust for number of reviews (more reviews = more reliable)
        # This follows a logarithmic curve to reflect diminishing returns
        if review_count > 0:
            review_factor = min(math.log10(review_count + 1) / math.log10(11), 1.0)  # max bonus at 10+ reviews
        else:
            review_factor = 0
        
        # Adjust for specificity of pros and cons
        pros_specificity = 0
        cons_specificity = 0
        
        if pros:
            # Calculate average length and complexity of pros
            avg_pro_length = sum(len(pro) for pro in pros) / len(pros)
            pros_specificity = min(avg_pro_length / 100, 1.0) * 0.7 + min(len(pros) / 5, 1.0) * 0.3
        
        if cons:
            # Calculate average length and complexity of cons
            avg_con_length = sum(len(con) for con in cons) / len(cons)
            cons_specificity = min(avg_con_length / 100, 1.0) * 0.7 + min(len(cons) / 5, 1.0) * 0.3
        
        # Balance of pros and cons (having both is good for confidence)
        balance_factor = 0
        if pros and cons:
            # Ideal balance is having both pros and cons - reflects honest review
            pros_cons_ratio = len(pros) / (len(pros) + len(cons))
            # Highest score when ratio is around 0.6 (slightly more pros than cons)
            balance_factor = 1.0 - abs(0.6 - pros_cons_ratio) * 1.25
            balance_factor = max(0, min(1.0, balance_factor))
        
        # Calculate specificity score (average of pros and cons)
        if pros or cons:
            if pros and cons:
                specificity_score = (pros_specificity + cons_specificity) / 2
            elif pros:
                specificity_score = pros_specificity * 0.8  # Penalty for only having pros
            else:
                specificity_score = cons_specificity * 0.7  # Bigger penalty for only having cons
        else:
            specificity_score = 0
        
        # Calculate final confidence score with weighted components
        # Weights: 50% sentiment, 30% review count, 10% specificity, 10% balance
        confidence_score = (
            0.5 * base_score +
            0.3 * (review_factor * 100) +
            0.1 * (specificity_score * 100) + 
            0.1 * (balance_factor * 100)
        )
        
        # Ensure score is between 0-100
        confidence_score = max(0, min(100, confidence_score))
        
        # Round to nearest integer
        confidence_score = round(confidence_score)
        
        # Generate detailed explanation
        sentiment_component = f"• {round(0.5 * base_score, 1)} points from sentiment score ({sentiment_score:.2f})"
        review_count_component = f"• {round(0.3 * (review_factor * 100), 1)} points from review count ({review_count} reviews)"
        specificity_component = f"• {round(0.1 * (specificity_score * 100), 1)} points from review specificity"
        balance_component = f"• {round(0.1 * (balance_factor * 100), 1)} points from pros/cons balance"
        
        explanation = (
            f"Confidence score of {confidence_score}% calculated based on:\n"
            f"{sentiment_component}\n"
            f"{review_count_component}\n"
            f"{specificity_component}\n"
            f"{balance_component}"
        )
        
        # Add interpretations
        confidence_level = ""
        if confidence_score >= 80:
            confidence_level = "Very High Confidence: Reviews strongly suggest this is a reliable product"
        elif confidence_score >= 65:
            confidence_level = "High Confidence: Reviews indicate this is likely a good product"
        elif confidence_score >= 50:
            confidence_level = "Moderate Confidence: Reviews show mixed but generally positive signals"
        elif confidence_score >= 35:
            confidence_level = "Low Confidence: Reviews raise some concerns about this product"
        else:
            confidence_level = "Very Low Confidence: Reviews suggest significant issues with this product"
        
        # Just print a summary instead of detailed explanation
        console.print(f"[green]Confidence Score:[/green] {confidence_score}%, Level: {confidence_level.split(':')[0]}")
        # console.print(f"[green]Explanation:[/green] {explanation}")
        # console.print(f"[green]Confidence Level:[/green] {confidence_level}")
        
        return TextContent(
            type="text",
            text=json.dumps({
                "confidence_score": confidence_score,
                "explanation": explanation,
                "confidence_level": confidence_level,
                "components": {
                    "sentiment_component": round(0.5 * base_score, 1),
                    "review_count_component": round(0.3 * (review_factor * 100), 1),
                    "specificity_component": round(0.1 * (specificity_score * 100), 1),
                    "balance_component": round(0.1 * (balance_factor * 100), 1)
                }
            })
        )
    except Exception as e:
        error_msg = f"Error calculating confidence score: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        return TextContent(
            type="text",
            text=json.dumps({"error": error_msg})
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run()
    else:
        mcp.run(transport="stdio") 