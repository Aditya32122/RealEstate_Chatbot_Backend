from .embeddings import generate_embedding
from .qdrant_client import search_vectors
import google.generativeai as genai
import os
import json

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def retrieve_context(query: str, top_k: int = 10):
    """Retrieve similar vectors from Qdrant"""
    query_vector = generate_embedding(query)
    results = search_vectors(query_vector, top_k=top_k)
    return [point.payload for point in results]

def llama_answer(query: str, context_rows: list):
    """Generate chart-ready JSON response using Gemini"""
    
    # Format context data more clearly
    context_text = "\n\n".join([
        f"Record {i+1}:\n" + "\n".join([f"  {k}: {v}" for k, v in row.items() if v is not None])
        for i, row in enumerate(context_rows)
    ])
    
    prompt = f"""You are a real estate data analyst. Based on the retrieved data, generate a structured JSON response.

Available Data:
{context_text}

User Query: {query}

Analyze the data and return ONLY valid JSON with this exact structure:
{{
  "summary": "A concise 2-3 sentence analysis answering the user's question with specific numbers and insights",
  "chart": {{
    "type": "line" or "bar" (choose based on query - use bar for comparisons, line for trends over time),
    "data": [
      {{"label": "value1", "metric": number}},
      {{"label": "value2", "metric": number}}
    ]
  }},
  "table": [
    {{"Location": "value", "Year": value, "Metric1": value, "Metric2": value}},
    ...
  ]
}}

Guidelines:
1. Extract relevant numeric values from the data
2. For comparisons: group by location, property type, or year
3. For trends: show year-over-year or time-based changes
4. Include 5-10 rows in the table with the most relevant columns
5. Use clear labels and round numbers appropriately
6. Focus on the metrics mentioned in the query

Return ONLY the JSON object, no explanation."""

    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
        )
    )
    
    result = response.text.strip()
    
    # Remove markdown code blocks if present
    if result.startswith("```json"):
        result = result.replace("```json", "").replace("```", "").strip()
    elif result.startswith("```"):
        result = result.replace("```", "").strip()
    
    try:
        parsed_result = json.loads(result)
        
        # Ensure the response has the correct structure
        if "summary" not in parsed_result:
            parsed_result["summary"] = "Analysis complete."
        if "chart" not in parsed_result or not parsed_result["chart"]:
            parsed_result["chart"] = {"type": "bar", "data": []}
        if "table" not in parsed_result or not parsed_result["table"]:
            parsed_result["table"] = context_rows[:10]  # Return first 10 rows
            
        return parsed_result
    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails
        return {
            "summary": result if len(result) < 500 else "Analysis could not be completed. Please rephrase your query.",
            "chart": {
                "type": "bar",
                "data": []
            },
            "table": context_rows[:10]
        }