import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from .embeddings import generate_embedding, create_chunk
from .qdrant_client import client as qdrant
import io

# For storing the CSV in memory
DATAFRAME = None

@api_view(["GET"])
def check_data(request):
    """Check if data exists in Qdrant"""
    try:
        collections = qdrant.get_collections().collections
        collection_exists = any(c.name == "realestate" for c in collections)
        
        if collection_exists:
            # Get collection info
            collection_info = qdrant.get_collection("realestate")
            return JsonResponse({
                "exists": True,
                "points_count": collection_info.points_count,
                "message": "Data already loaded in Qdrant"
            })
        else:
            return JsonResponse({
                "exists": False,
                "message": "No data found. Please upload a file."
            })
    except Exception as e:
        return JsonResponse({"exists": False, "error": str(e)}, status=500)


@api_view(["POST"])
def upload_csv(request):
    global DATAFRAME

    try:
        file = request.FILES["file"]
        file_content_bytes = file.read()
        
        # Check file type and read accordingly
        if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            # Excel file
            DATAFRAME = pd.read_excel(io.BytesIO(file_content_bytes))
        elif file.name.endswith('.csv'):
            # CSV file
            try:
                DATAFRAME = pd.read_csv(io.BytesIO(file_content_bytes), encoding='utf-8', on_bad_lines='skip')
            except:
                DATAFRAME = pd.read_csv(io.BytesIO(file_content_bytes), encoding='latin-1', on_bad_lines='skip')
        else:
            return JsonResponse({"error": "Please upload a CSV or Excel file (.csv, .xlsx, .xls)"}, status=400)
        
        # Check if dataframe is empty
        if DATAFRAME.empty:
            return JsonResponse({"error": "File is empty"}, status=400)
        
        # Clean column names (remove leading/trailing spaces)
        DATAFRAME.columns = DATAFRAME.columns.str.strip()
        
        # Verify at least some expected columns exist
        required_columns = ['final location', 'year']
        missing_cols = [col for col in required_columns if col not in DATAFRAME.columns]
        
        if missing_cols:
            return JsonResponse({
                "error": f"Missing required columns: {missing_cols}",
                "found_columns": list(DATAFRAME.columns[:10]),  # Show first 10 columns
                "total_columns": len(DATAFRAME.columns)
            }, status=400)

        # Drop old collection
        try:
            qdrant.delete_collection("realestate")
        except:
            pass

        # Recreate
        from qdrant_client.models import VectorParams, Distance
        qdrant.create_collection(
            collection_name="realestate",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

        # Embed each row with all columns
        payloads = []
        vectors = []
        for i, row in DATAFRAME.iterrows():
            # Create text chunk with all data
            text = create_chunk(row)
            emb = generate_embedding(text)
            
            # Store full row as payload
            payload = row.to_dict()
            # Convert any NaN to None for JSON serialization
            payload = {k: (None if pd.isna(v) else v) for k, v in payload.items()}
            
            payloads.append(payload)
            vectors.append(emb)

        # Batch upsert
        qdrant.upsert(
            collection_name="realestate",
            points=[
                {
                    "id": idx,
                    "vector": vectors[idx],
                    "payload": payloads[idx]
                }
                for idx in range(len(vectors))
            ],
        )

        return JsonResponse({
            "message": "File uploaded and embedded!",
            "rows_processed": len(DATAFRAME),
            "columns": list(DATAFRAME.columns)
        })
    
    except KeyError as e:
        return JsonResponse({"error": f"No file uploaded: {str(e)}"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e), "type": str(type(e).__name__)}, status=500)


@api_view(["POST"])
def query_view(request):
    query = request.data.get("query", "")
    
    if not query:
        return JsonResponse({"error": "Query parameter is required"}, status=400)
    
    try:
        from .llm import llama_answer, retrieve_context
        
        # Check if collection exists
        collections = qdrant.get_collections().collections
        if not any(c.name == "realestate" for c in collections):
            return JsonResponse({
                "error": "No data found in Qdrant. Please upload a file first."
            }, status=400)
        
        # Retrieve context from Qdrant
        context_rows = retrieve_context(query, top_k=10)
        
        if not context_rows:
            return JsonResponse({
                "error": "No relevant data found for your query. Try different keywords."
            }, status=404)
        
        # Get structured JSON response from LLM
        result = llama_answer(query, context_rows)
        
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)