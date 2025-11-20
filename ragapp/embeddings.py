import google.generativeai as genai
import pandas as pd
import json
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def create_chunk(row):
    """Create a comprehensive text chunk from all CSV columns"""
    chunk = f"""Location: {row.get('final location', 'N/A')}
Year: {row.get('year', 'N/A')}
City: {row.get('city', 'N/A')}
Coordinates: ({row.get('loc_lat', 'N/A')}, {row.get('loc_lng', 'N/A')})

Sales Metrics:
- Total Sales (IGR): {row.get('total_sales - igr', 'N/A')}
- Total Sold (IGR): {row.get('total sold - igr', 'N/A')}

Property Types Sold:
- Flats Sold: {row.get('flat_sold - igr', 'N/A')}
- Office Sold: {row.get('office_sold - igr', 'N/A')}
- Shops Sold: {row.get('shop_sold - igr', 'N/A')}
- Others Sold: {row.get('others_sold - igr', 'N/A')}
- Commercial Sold: {row.get('commercial_sold - igr', 'N/A')}
- Other Sold: {row.get('other_sold - igr', 'N/A')}
- Residential Sold: {row.get('residential_sold - igr', 'N/A')}

Weighted Average Rates:
- Flat Rate: {row.get('flat - weighted average rate', 'N/A')}
- Office Rate: {row.get('office - weighted average rate', 'N/A')}
- Others Rate: {row.get('others - weighted average rate', 'N/A')}
- Shop Rate: {row.get('shop - weighted average rate', 'N/A')}

Prevailing Rate Ranges:
- Flat Range: {row.get('flat - most prevailing rate - range', 'N/A')}
- Office Range: {row.get('office - most prevailing rate - range', 'N/A')}
- Others Range: {row.get('others - most prevailing rate - range', 'N/A')}
- Shop Range: {row.get('shop - most prevailing rate - range', 'N/A')}

Supply Metrics:
- Total Units: {row.get('total units', 'N/A')}
- Total Carpet Area (sqft): {row.get('total carpet area supplied (sqft)', 'N/A')}
- Flat Total: {row.get('flat total', 'N/A')}
- Shop Total: {row.get('shop total', 'N/A')}
- Office Total: {row.get('office total', 'N/A')}
- Others Total: {row.get('others total', 'N/A')}
"""
    return chunk

def generate_embedding(text: str):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']