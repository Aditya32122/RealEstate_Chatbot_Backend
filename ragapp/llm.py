from .embeddings import generate_embedding
from .qdrant_client import search_vectors
import google.generativeai as genai
import os
import json
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def retrieve_context(query: str, top_k: int = 10):
    """Retrieve similar vectors from Qdrant"""
    query_vector = generate_embedding(query)
    results = search_vectors(query_vector, top_k=top_k)
    return [point.payload for point in results]

def llama_answer(query: str, context_rows: list):
    """Generate chart-ready JSON response using Gemini"""
    
    # Detect query intent
    query_lower = query.lower()
    is_comparison = any(word in query_lower for word in ['compare', 'vs', 'versus', 'between', 'difference', 'across'])
    is_trend = any(word in query_lower for word in ['trend', 'over time', 'yearly', 'year', 'growth', 'change', 'last', 'years', 'over'])
    has_total = any(word in query_lower for word in ['total', 'sum', 'aggregate', 'highest', 'top', 'which', 'best', 'worst'])
    
    logger.info(f"Query: {query}")
    logger.info(f"Detected - Comparison: {is_comparison}, Trend: {is_trend}, Total: {has_total}")
    
    # Format context data
    context_text = "\n\n".join([
        f"Record {i+1}:\n" + "\n".join([f"  {k}: {v}" for k, v in row.items() if v is not None])
        for i, row in enumerate(context_rows[:15])
    ])
    
    # Determine chart type with better logic
    if is_trend:
        # Time-based queries always use line charts
        chart_hint = "line"
        format_example = """
{
  "summary": "Trend analysis showing changes over time",
  "chart": {
    "type": "line",
    "data": [
      {"year": 2020, "Wakad": 9116, "Aundh": 8889},
      {"year": 2021, "Wakad": 9289, "Aundh": 9366},
      {"year": 2022, "Wakad": 9734, "Aundh": 9443}
    ]
  },
  "table": [...]
}
"""
    elif is_comparison and not is_trend:
        # Categorical comparison without time
        chart_hint = "bar"
        format_example = """
{
  "summary": "Comparison across different categories",
  "chart": {
    "type": "bar",
    "data": [
      {"location": "Wakad", "average_price": 9500},
      {"location": "Aundh", "average_price": 10000},
      {"location": "Baner", "average_price": 8500}
    ]
  },
  "table": [...]
}
"""
    elif has_total:
        # Aggregate/Total queries
        chart_hint = "bar"
        format_example = """
{
  "summary": "Categorical totals or rankings",
  "chart": {
    "type": "bar",
    "data": [
      {"location": "Wakad", "total_sales": 50000},
      {"location": "Aundh", "total_sales": 45000}
    ]
  },
  "table": [...]
}
"""
    else:
        # Default to line for analysis
        chart_hint = "line"
        format_example = """
{
  "summary": "Data analysis",
  "chart": {
    "type": "line",
    "data": [
      {"year": 2020, "value": 9116},
      {"year": 2021, "value": 9289}
    ]
  },
  "table": [...]
}
"""
    
    prompt = f"""You are a real estate data analyst. Analyze the data and return a structured JSON response.

Available Data:
{context_text}

User Query: "{query}"

Query Classification:
- Is Comparison: {is_comparison}
- Is Trend/Time-based: {is_trend}
- Has Total/Aggregate: {has_total}
- Recommended Chart: {chart_hint.upper()}

Expected Output Format:
{format_example}

CRITICAL RULES FOR CHART DATA:

1. **Chart Type Selection (MUST FOLLOW):**
   - "line" → Use when query mentions: trends, over time, years, growth, change, last X years
   - "bar" → Use when query mentions: compare (without time), total, highest, top, which, best

2. **Data Structure for LINE charts:**
   ✅ CORRECT FORMAT:
   [
     {{"year": 2020, "Wakad": 5000, "Aundh": 6000}},
     {{"year": 2021, "Wakad": 5500, "Aundh": 6500}},
     {{"year": 2022, "Wakad": 6000, "Aundh": 7000}}
   ]
   - First key = time dimension (year/date/month)
   - Other keys = location/metric names
   - Values MUST be numbers
   
   ❌ WRONG FORMATS:
   - [{{"label": "Wakad", "metric": 5000}}] ← NO labels
   - [{{"Wakad": {{"2020": 5000}}}}] ← NO nested objects
   - [{{"year": "2020", "value": "5000"}}] ← NO strings for numbers

3. **Data Structure for BAR charts:**
   ✅ CORRECT FORMAT:
   [
     {{"location": "Wakad", "average_price": 5000, "sales": 100}},
     {{"location": "Aundh", "average_price": 6000, "sales": 120}},
     {{"location": "Baner", "average_price": 5500, "sales": 90}}
   ]
   - First key = category (location/type/name)
   - Other keys = metrics (prices, sales, counts)
   - Values MUST be numbers

4. **Consistency Requirements:**
   - ALL rows must have IDENTICAL keys
   - ALL numeric values must be actual numbers (not strings)
   - Use simple, descriptive key names (no spaces, use underscores)

5. **Examples by Query Type:**

   Query: "Compare Wakad and Aundh trends over last 3 years"
   → Type: line
   → Data: [{{"year": 2022, "Wakad": 5000, "Aundh": 6000}}, {{"year": 2023, ...}}]

   Query: "Which location has highest price"
   → Type: bar
   → Data: [{{"location": "Wakad", "price": 5000}}, {{"location": "Aundh", ...}}]

   Query: "Show price trends in Wakad"
   → Type: line
   → Data: [{{"year": 2020, "price": 5000}}, {{"year": 2021, ...}}]

   Query: "Compare Wakad vs Aundh"
   → Type: bar
   → Data: [{{"location": "Wakad", "avg_price": 5000}}, {{"location": "Aundh", ...}}]

6. **Table Data:**
   - Include 5-10 most relevant rows
   - Use clear column names
   - Include all important metrics

7. **Summary:**
   - 2-3 concise sentences
   - Include specific numbers
   - Highlight key insights

Return ONLY the JSON object. NO markdown, NO code blocks, NO explanations."""

    model = genai.GenerativeModel('gemini-2.0-flash')
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Lower for more consistent structure
                top_p=0.8,
            )
        )
        
        result = response.text.strip()
        
        # Clean markdown formatting
        if result.startswith("```json"):
            result = result.replace("```json", "").replace("```", "").strip()
        elif result.startswith("```"):
            result = result.replace("```", "").strip()
        
        # Parse JSON
        parsed_result = json.loads(result)
        
        # Log the result for debugging
        logger.info(f"LLM Response - Chart Type: {parsed_result.get('chart', {}).get('type')}")
        logger.info(f"Chart Data Sample: {parsed_result.get('chart', {}).get('data', [])[:2]}")
        
        # Validate and fix structure
        if "summary" not in parsed_result:
            parsed_result["summary"] = "Analysis complete."
        
        if "chart" not in parsed_result or not parsed_result["chart"]:
            parsed_result["chart"] = {"type": chart_hint, "data": []}
        
        # Ensure chart type is valid
        if parsed_result["chart"].get("type") not in ["bar", "line"]:
            logger.warning(f"Invalid chart type: {parsed_result['chart'].get('type')}, using {chart_hint}")
            parsed_result["chart"]["type"] = chart_hint
        
        if "table" not in parsed_result or not parsed_result["table"]:
            parsed_result["table"] = context_rows[:10]
        
        # Validate chart data structure
        chart_data = parsed_result.get("chart", {}).get("data", [])
        if chart_data:
            first_row = chart_data[0]
            
            # Check for invalid structures
            if "label" in first_row or "metric" in first_row:
                logger.warning("Detected wrong structure with 'label'/'metric', attempting fix")
                parsed_result["chart"]["data"] = restructure_chart_data(chart_data, context_rows)
            
            # Validate all rows have same keys
            if len(chart_data) > 1:
                first_keys = set(first_row.keys())
                for i, row in enumerate(chart_data[1:], 1):
                    if set(row.keys()) != first_keys:
                        logger.warning(f"Inconsistent keys at row {i}: {row.keys()} vs {first_keys}")
        
        return parsed_result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Raw response: {result[:500]}")
        return {
            "summary": "Could not parse analysis. The AI response was not in valid JSON format. Please try rephrasing your query.",
            "chart": {"type": chart_hint, "data": []},
            "table": context_rows[:10]
        }
    except Exception as e:
        logger.error(f"Error in llama_answer: {str(e)}")
        return {
            "summary": f"An error occurred: {str(e)}",
            "chart": {"type": chart_hint, "data": []},
            "table": context_rows[:10]
        }

def restructure_chart_data(wrong_data, context_rows):
    """Fix wrong chart data structure when LLM returns label/metric format"""
    try:
        from collections import defaultdict
        grouped = defaultdict(dict)
        
        for item in wrong_data:
            label = item.get("label", "")
            metric = item.get("metric", 0)
            
            # Try to extract location and year from label
            parts = label.split()
            if len(parts) >= 2:
                location = parts[0]
                
                # Find matching context row
                for row in context_rows:
                    loc = str(row.get("final location", "")).strip()
                    if loc.lower() == location.lower():
                        year = row.get("year")
                        if year:
                            grouped[year][location] = metric
                            break
        
        # Convert to proper list format
        result = []
        for year in sorted(grouped.keys()):
            row = {"year": year}
            row.update(grouped[year])
            result.append(row)
        
        if result:
            logger.info(f"Successfully restructured data: {len(result)} rows")
            return result
        else:
            logger.warning("Could not restructure data, returning original")
            return wrong_data
            
    except Exception as e:
        logger.error(f"Error in restructure_chart_data: {str(e)}")
        return wrong_data