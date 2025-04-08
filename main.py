# -*- coding: utf-8 -*-
import os
import io
import secrets
from datetime import datetime
from typing import List, Optional, Dict, Any
import traceback # エラー詳細表示のため

import requests
import PyPDF2
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends, Body, Response # Responseを追加
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.errors import OperationFailure, ConnectionFailure # MongoDB固有のエラー
from pydantic import BaseModel, HttpUrl, Field
# from sentence_transformers import SentenceTransformer # 不要
import openai # ★ OpenAIライブラリをインポート
from openai import OpenAI # ★ 新しいAPI呼び出し形式 (v1.0.0以降)
from dotenv import load_dotenv
import asyncio
import time

from typing import List, Optional # Optionalを追加
from pydantic import BaseModel, Field, HttpUrl # BaseModel, Field は既存だが明示
from bson import ObjectId # 結果のID変換用

# FastAPI app initialization
app = FastAPI()

load_dotenv()

# --- Environment variables ---
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB default
MONGO_MASTER_URI = os.getenv("MONGO_MASTER_URI")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") # Note: This seems unused if using OpenAI
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # ★ OpenAI APIキー
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002") # ★ 使用するモデル (デフォルト設定)

# --- Validate required environment variables on startup ---
required_env_vars = ["MONGO_MASTER_URI", "ADMIN_API_KEY", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    print(f"FATAL: Missing required environment variables: {', '.join(missing_vars)}")
    raise SystemExit(f"Missing required environment variables: {', '.join(missing_vars)}")

# --- OpenAI Client Initialization ---
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"FATAL: Failed to initialize OpenAI client: {e}")
    raise SystemExit(f"Failed to initialize OpenAI client: {e}")

# --- MongoDB Connection ---
mongo_client = None
auth_db = None
try:
    mongo_client = MongoClient(
        MONGO_MASTER_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=3000
    )
    mongo_client.admin.command('ismaster')
    print("Successfully connected to MongoDB.")
    auth_db = mongo_client["auth_db"]
except ConnectionFailure as e:
    print(f"FATAL: Failed to connect to MongoDB (ConnectionFailure): {e}")
    raise SystemExit(f"MongoDB connection failed: {e}")
except OperationFailure as e:
    print(f"FATAL: Failed to connect to MongoDB (OperationFailure): {e.details}")
    raise SystemExit(f"MongoDB operation failure during connection test: {e}")
except Exception as e:
    print(f"FATAL: An unexpected error occurred during MongoDB connection: {type(e).__name__} - {e}")
    raise SystemExit(f"Unexpected MongoDB connection error: {e}")


# --- Data models ---
class UserRegister(BaseModel):
    supabase_user_id: str = Field(..., min_length=1)

class RegisterResponse(BaseModel):
    api_key: str
    db_name: str
    database_exist: bool = Field(..., description="True if the user database already existed, False if newly created.")

class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$") # Restrict collection names

class ProcessRequest(BaseModel):
    pdf_url: HttpUrl
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$") # Restrict collection names
    metadata: Dict[str, str] = Field(default_factory=dict)

class UserInfoRequest(BaseModel):
    supabase_user_id: str = Field(..., min_length=1, description="The Supabase User ID")
    api_key: str = Field(..., description="The user's API key")

class UserInfoResponse(BaseModel):
    db_name: str
    collections: List[str]

# ★★★ 新しいリクエストモデル (コレクション削除・名前変更用) ★★★
class CollectionActionBase(BaseModel):
    """Base model for actions requiring an API key in the body."""
    api_key: str = Field(..., description="User's API Key")

class DeleteCollectionRequest(CollectionActionBase):
    collection_name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Name of the collection to delete")

class RenameCollectionRequest(CollectionActionBase):
    current_name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Current name of the collection")
    new_name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="New name for the collection")

# ★★★ 新しい汎用レスポンスモデル ★★★
class ActionResponse(BaseModel):
    status: str
    message: str
    details: Optional[str] = None # For additional info, e.g., rename warnings


# --- Dependencies ---
def verify_admin(api_key: str = Header(..., alias="X-API-Key")):
    """Verifies the admin API key provided in the header."""
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required")

# (get_userはヘッダー認証用に変更)
def get_user_header(api_key: str = Header(..., alias="X-API-Key")):
    """
    Retrieves the user based on the API key provided in the X-API-Key header.
    Used for endpoints like POST /collections, /process, /search etc.
    """
    if auth_db is None:
         print("Error in get_user_header: auth_db is not available.")
         raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        if not api_key:
             print("Error in get_user_header: X-API-Key header is missing.")
             raise HTTPException(status_code=401, detail="X-API-Key header is required.") # 401 Unauthorized

        user = auth_db.users.find_one({"api_key": api_key})
        if not user:
            raise HTTPException(status_code=403, detail="Invalid API Key provided in X-API-Key header.") # 403 Forbidden
        return user
    except OperationFailure as e:
        print(f"Error finding user in auth_db (get_user_header): {e.details}")
        raise HTTPException(status_code=503, detail="Database operation failed while validating API key header.")
    except HTTPException as e:
         raise e
    except Exception as e:
        print(f"Unexpected error in get_user_header: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected error occurred during API key header validation.")

# 新しい依存関係: BodyからAPIキーを取得して認証
async def get_user_body(request: CollectionActionBase = Body(...)):
     """Dependency to authenticate user based on API key within the request body model."""
     if auth_db is None:
          print("Error in get_user_body: auth_db is not available.")
          raise HTTPException(status_code=503, detail="Database service unavailable")
     try:
         if not request.api_key:
              print("Error in get_user_body: API key missing in request model.")
              raise HTTPException(status_code=400, detail="API Key not provided in the request body.")

         user = auth_db.users.find_one({"api_key": request.api_key})
         if not user:
             raise HTTPException(status_code=403, detail="Invalid API Key provided in request body.") # 403 Forbidden
         return user
     except OperationFailure as e:
         print(f"Error finding user in auth_db (get_user_body): {e.details}")
         raise HTTPException(status_code=503, detail="Database operation failed while validating API key from body.")
     except HTTPException as e:
          raise e
     except Exception as e:
         print(f"Unexpected error in get_user_body: {type(e).__name__} - {e}")
         traceback.print_exc()
         raise HTTPException(status_code=500, detail="An unexpected error occurred during API key body validation.")


# --- MongoDB manager ---
class MongoDBManager:
    @staticmethod
    def create_user_db(supabase_user_id: str):
        if auth_db is None:
             print("Error in create_user_db: auth_db is not available.")
             raise HTTPException(status_code=503, detail="Database service unavailable")
        db_name = f"userdb_{supabase_user_id[:8]}_{secrets.token_hex(4)}"
        api_key = secrets.token_urlsafe(32)
        try:
            result = auth_db.users.insert_one({
                "supabase_user_id": supabase_user_id,
                "db_name": db_name,
                "api_key": api_key,
                "created_at": datetime.utcnow()
            })
            print(f"Inserted user ({supabase_user_id}) record into auth_db, db_name: {db_name}, ID: {result.inserted_id}")
            return {"api_key": api_key, "db_name": db_name}
        except OperationFailure as e:
            print(f"MongoDB Operation Failure creating user record for {supabase_user_id}: {e.details}")
            raise HTTPException(status_code=500, detail=f"Database operation failed: {e}")
        except Exception as e:
            print(f"Error inserting user {supabase_user_id} into auth_db: {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to create user record: {e}")

    @staticmethod
    def get_user_db(user: Dict):
        if mongo_client is None:
             print("Error in get_user_db: mongo_client is not available.")
             raise HTTPException(status_code=503, detail="Database service unavailable")
        db_name = user.get("db_name")
        if not db_name:
            print(f"Error in get_user_db: User object is missing 'db_name'. User: {user.get('supabase_user_id', 'N/A')}")
            raise HTTPException(status_code=500, detail="Internal server error: User data is inconsistent.")
        try:
            return mongo_client[db_name]
        except Exception as e:
            print(f"Unexpected error accessing user DB '{db_name}': {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=503, detail=f"Failed to access user database '{db_name}': {e}")

    @staticmethod
    def create_collection(db, name: str, user_id: str):
        if db is None:
             print(f"Error in create_collection: Database object is None for user {user_id}.")
             raise HTTPException(status_code=500, detail="Internal Server Error: Invalid database reference.")
        try:
            collection_names = db.list_collection_names()
            if name not in collection_names:
                 collection = db[name]
                 print(f"Collection '{name}' will be created implicitly on first write in database '{db.name}'.")
                 # Note: No need to insert/delete dummy doc, MongoDB creates on first write.
                 return collection
            else:
                 print(f"Collection '{name}' already exists in database '{db.name}'.")
                 return db[name]
        except OperationFailure as e:
            print(f"MongoDB Operation Failure accessing/listing collection '{name}' in '{db.name}': {e.details}")
            raise HTTPException(status_code=500, detail=f"Database operation failed for collection '{name}': {e.details}")
        except ConnectionFailure as e:
            print(f"MongoDB Connection Failure during collection operation '{name}' in '{db.name}': {e}")
            raise HTTPException(status_code=503, detail=f"Database connection lost while working with collection '{name}'.")
        except Exception as e:
            print(f"Error accessing/creating collection '{name}' in db '{db.name}': {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to ensure collection '{name}' exists: {e}")

# --- Text splitting helper ---
def split_text_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 100) -> List[str]:
    """Splits text into chunks with a target size and overlap, word-boundary aware."""
    words = text.split()
    if not words:
        return []
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        end_pos = current_pos
        current_length = 0
        last_valid_end_pos = current_pos
        while end_pos < len(words):
            word_len = len(words[end_pos])
            length_to_add = word_len + (1 if end_pos > current_pos else 0)
            if current_length + length_to_add <= chunk_size:
                current_length += length_to_add
                last_valid_end_pos = end_pos + 1
                end_pos += 1
            else:
                break
        if last_valid_end_pos == current_pos:
             if current_pos == 0 and len(words[0]) > chunk_size:
                 print(f"Warning: Single word exceeds chunk size: '{words[0][:50]}...'")
                 chunks.append(words[0])
                 current_pos += 1
                 continue
             if current_pos < len(words):
                 last_valid_end_pos = current_pos + 1
             else:
                 break
        chunk_words = words[current_pos:last_valid_end_pos]
        chunks.append(" ".join(chunk_words))
        overlap_start_index = last_valid_end_pos - 1
        overlap_char_count = 0
        while overlap_start_index > current_pos:
             overlap_char_count += len(words[overlap_start_index]) + 1
             if overlap_char_count >= overlap:
                  break
             overlap_start_index -= 1
        overlap_start_index = max(current_pos, overlap_start_index)
        # Corrected advancement logic: Use last_valid_end_pos for simple cases,
        # use calculated overlap_start_index if it advances position.
        if overlap_start_index > current_pos:
             current_pos = overlap_start_index
        else:
             current_pos = last_valid_end_pos # Ensure progress if overlap doesn't move forward


    return chunks


# --- OpenAI Embedding Function ---
def get_openai_embedding(text: str) -> List[float]:
    """Generates embedding for the given text using OpenAI API."""
    if not text or text.isspace():
         print("Warning: Attempted to get embedding for empty or whitespace text.")
         return []
    try:
        cleaned_text = ' '.join(text.split())
        if not cleaned_text:
             print("Warning: Text became empty after cleaning whitespace.")
             return []
        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=cleaned_text,
        )
        if response.data and len(response.data) > 0 and response.data[0].embedding:
            return response.data[0].embedding
        else:
             print(f"Warning: OpenAI API returned no embedding data for text: {cleaned_text[:100]}...")
             raise HTTPException(status_code=500, detail="OpenAI API returned unexpected empty data.")
    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to OpenAI API: {e}")
    except openai.APIStatusError as e:
        print(f"OpenAI API Status Error: Status Code {e.status_code}, Response: {e.response}")
        status_code = e.status_code
        detail = f"OpenAI Service Error (Status {status_code}): {e.message or str(e.response)}" # Use str(e.response) for safety
        if status_code == 400:
            detail = f"OpenAI Bad Request: Input may be invalid. {e.message}"
            raise HTTPException(status_code=400, detail=detail)
        elif status_code == 401:
            detail = "OpenAI Authentication Error. Check API Key configuration."
            raise HTTPException(status_code=401, detail=detail)
        elif status_code == 429:
            detail = "OpenAI Rate Limit Exceeded. Please wait and retry."
            raise HTTPException(status_code=429, detail=detail)
        elif status_code >= 500:
            detail = f"OpenAI Server Error (Status {status_code}). Please retry later. {e.message}"
            raise HTTPException(status_code=502, detail=detail) # 502 Bad Gateway
        else:
             raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        print(f"An unexpected error occurred while getting OpenAI embedding: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding generation failed due to an unexpected error: {e}")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Name of the collection to search within")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    num_candidates: int = Field(100, ge=10, le=1000, description="Number of candidates to consider for vector search")

class SearchResultItem(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float

class VectorSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Name of the collection to search within")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    num_candidates: int = Field(100, ge=10, le=1000, description="Number of candidates for initial vector search phase")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filter criteria for metadata")

class VectorSearchResponse(BaseModel):
    results: List[SearchResultItem]

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

# --- API endpoints ---
@app.get("/health")
def health_check():
    db_status = "connected" if mongo_client and auth_db else "disconnected"
    openai_status = "initialized" if openai_client else "not_initialized"
    if db_status == "connected":
        try:
            mongo_client.admin.command('ping')
        except (ConnectionFailure, OperationFailure) as e:
            db_status = f"error ({type(e).__name__})"
    return {"status": "ok", "database": db_status, "openai_client": openai_status}

@app.get("/auth-db", dependencies=[Depends(verify_admin)])
def get_auth_db_contents():
    if auth_db is None:
         raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        users = list(auth_db.users.find({}, {"_id": 0, "api_key": 0}))
        return {"users": users}
    except OperationFailure as e:
        print(f"Error reading from auth_db: {e.details}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user data: {e.details}")
    except Exception as e:
        print(f"Unexpected error reading from auth_db: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected error occurred retrieving user data.")

@app.post("/register", response_model=RegisterResponse)
def register_user(request: UserRegister, response: Response):
    if auth_db is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        existing_user = auth_db.users.find_one({"supabase_user_id": request.supabase_user_id})
        if existing_user:
            api_key = existing_user.get("api_key")
            db_name = existing_user.get("db_name")
            if not api_key or not db_name:
                raise HTTPException(status_code=500, detail="Internal Server Error: User data inconsistent.")
            response.status_code = 200
            return RegisterResponse(api_key=api_key, db_name=db_name, database_exist=True)
        else:
            db_info = MongoDBManager.create_user_db(request.supabase_user_id)
            response.status_code = 201
            return RegisterResponse(api_key=db_info["api_key"], db_name=db_info["db_name"], database_exist=False)
    except HTTPException as e: raise e
    except OperationFailure as e:
        raise HTTPException(status_code=500, detail=f"Database error during registration: {e.details}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error during registration: {e}")

# POST /collections uses Header auth
@app.post("/collections", status_code=201)
def create_collection_endpoint(
    request: CollectionCreate,
    user: Dict = Depends(get_user_header) # Use header-based auth
):
    """Creates a new collection within the user's database. Requires X-API-Key header."""
    try:
        db = MongoDBManager.get_user_db(user)
        collection = MongoDBManager.create_collection(db, request.name, user["supabase_user_id"])
        return {"status": "created", "collection_name": collection.name}
    except HTTPException as e: raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {e}")

# /process uses Header auth
@app.post("/process")
async def process_pdf(
    request: ProcessRequest,
    user: Dict = Depends(get_user_header) # Use header-based auth
):
    """Downloads, processes PDF, and stores embeddings. Requires X-API-Key header."""
    index_name = "vector_index"
    index_status = "not_checked"
    first_error = None
    duplicates_removed_count = 0
    inserted_count = 0
    processed_chunks_count = 0
    errors = []
    start_time_total = time.time()
    print(f"Processing PDF for user {user.get('supabase_user_id', 'N/A')}: {request.pdf_url}")

    try:
        # --- 1. Download PDF ---
        pdf_content: io.BytesIO
        try:
            start_time_download = time.time()
            # Run blocking requests.get in a separate thread
            response = await asyncio.to_thread(requests.get, str(request.pdf_url), timeout=60)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            end_time_download = time.time()
            print(f"PDF downloaded successfully in {end_time_download - start_time_download:.2f}s.")

            # Check file size based on Content-Length header (if available) or after download
            content_length = response.headers.get('Content-Length')
            pdf_bytes = response.content # Read content into memory
            pdf_size = len(pdf_bytes)
            if pdf_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413, # Payload Too Large
                    detail=f"PDF file size ({pdf_size / (1024*1024):.2f} MB) exceeds the limit of {MAX_FILE_SIZE / (1024*1024)} MB."
                )
            pdf_content = io.BytesIO(pdf_bytes)
            size_source = f"from Content-Length header: {content_length}" if content_length else "checked after download"
            print(f"PDF size: {pdf_size / (1024*1024):.2f} MB ({size_source}).")

        except requests.exceptions.Timeout:
             print(f"Timeout downloading PDF: {request.pdf_url}")
             raise HTTPException(status_code=408, detail="PDF download timed out after 60 seconds.")
        except requests.exceptions.SSLError as ssl_err:
             print(f"SSL Error downloading PDF: {request.pdf_url} - {ssl_err}")
             raise HTTPException(status_code=502, detail=f"Could not download PDF due to SSL error: {ssl_err}. The server's SSL certificate might be invalid.")
        except requests.exceptions.RequestException as req_error:
             print(f"PDF download failed: {type(req_error).__name__} - {str(req_error)}")
             status_code = 502 if isinstance(req_error, requests.exceptions.ConnectionError) else 400
             detail = f"PDF download failed: {str(req_error)}"
             # Include response details if available for better debugging
             if hasattr(req_error, 'response') and req_error.response is not None:
                 detail += f" (Status: {req_error.response.status_code})"
             raise HTTPException(status_code=status_code, detail=detail)

        # --- 2. Extract text ---
        text: str = ""
        try:
            print("Extracting text from PDF...")
            start_time_extract = time.time()
            def extract_text_sync(pdf_stream):
                 reader = PyPDF2.PdfReader(pdf_stream)
                 if reader.is_encrypted:
                      raise ValueError("Encrypted PDF files are not supported.")
                 extracted_text = ""
                 for page_num, page in enumerate(reader.pages):
                      try:
                          page_text = page.extract_text()
                          if page_text: extracted_text += page_text + "\n"
                      except Exception as page_error:
                           print(f"Warning: Could not extract text from page {page_num + 1}: {page_error}")
                           continue
                 return extracted_text

            text = await asyncio.to_thread(extract_text_sync, pdf_content)
            end_time_extract = time.time()

            if not text.strip():
                 print("Warning: No text could be extracted from the PDF.")
                 return JSONResponse(
                      content={
                           "status": "success", "message": "No text content found in PDF.",
                           "chunks_processed": 0, "chunks_inserted": 0, "duplicates_removed": 0,
                           "vector_index_name": index_name, "vector_index_status": "skipped_no_text",
                           "processing_time_seconds": round(time.time() - start_time_total, 2)
                           }, status_code=200)
            print(f"Text extracted successfully ({len(text)} chars) in {end_time_extract - start_time_extract:.2f}s.")
        except PyPDF2.errors.PdfReadError as pdf_error: raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF file: {pdf_error}")
        except ValueError as val_err: raise HTTPException(status_code=400, detail=str(val_err))
        except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")

        # --- 3. Split text ---
        start_time_split = time.time()
        chunks = split_text_into_chunks(text, chunk_size=1500, overlap=100)
        end_time_split = time.time()
        print(f"Text split into {len(chunks)} chunks in {end_time_split - start_time_split:.2f}s.")
        if not chunks:
             return JSONResponse(
                  content={
                      "status": "success", "message": "No processable text chunks generated.",
                      "chunks_processed": 0, "chunks_inserted": 0, "duplicates_removed": 0,
                      "vector_index_name": index_name, "vector_index_status": "skipped_no_chunks",
                      "processing_time_seconds": round(time.time() - start_time_total, 2)
                  }, status_code=200)

        # --- 4. DB Setup ---
        db = MongoDBManager.get_user_db(user)
        collection = MongoDBManager.create_collection(db, request.collection_name, user["supabase_user_id"])
        print(f"Using collection '{request.collection_name}' in database '{db.name}'.")

        # --- 5. Process Chunks ---
        start_time_chunks = time.time()
        print(f"Starting processing of {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            if not chunk or chunk.isspace():
                 print(f"Skipping empty chunk {i+1}/{len(chunks)}.")
                 continue
            processed_chunks_count += 1
            try:
                print(f"Processing chunk {i+1}/{len(chunks)} (Length: {len(chunk)})...")
                start_time_embed = time.time()
                embedding = await asyncio.to_thread(get_openai_embedding, chunk)
                end_time_embed = time.time()
                if not embedding:
                     errors.append({"chunk_index": i, "error": "Skipped due to empty embedding result", "status_code": 400})
                     continue
                print(f"Embedding generated for chunk {i+1} in {end_time_embed - start_time_embed:.2f}s.")
                doc = {
                    "text": chunk, "embedding": embedding,
                    "metadata": {**request.metadata, "chunk_index": i, "original_url": str(request.pdf_url), "processed_at": datetime.utcnow()},
                    "created_at": datetime.utcnow()
                }
                start_time_insert = time.time()
                res = collection.insert_one(doc)
                end_time_insert = time.time()
                if res.inserted_id:
                    inserted_count += 1
                    print(f"Chunk {i+1} inserted (ID: {res.inserted_id}) in {end_time_insert - start_time_insert:.2f}s.")
                else:
                    errors.append({"chunk_index": i, "error": "Insertion failed silently", "status_code": 500})

            except HTTPException as http_exc:
                errors.append({
                    "chunk_index": i,
                    "error": http_exc.detail,
                    "status_code": http_exc.status_code
                })
                is_critical = http_exc.status_code in [429, 500, 502, 503]
                if is_critical and first_error is None:
                    first_error = http_exc
                    print("Stopping: critical HTTP error.")
                    break # Exit the loop on critical error

            except (OperationFailure, ConnectionFailure) as db_err:
                error_detail = getattr(db_err, 'details', str(db_err))
                errors.append({
                    "chunk_index": i,
                    "error": f"DB error: {error_detail}",
                    "status_code": 503
                })
                if first_error is None:
                    first_error = db_err
                    print("Stopping: DB error.")
                    break # Exit the loop

            except Exception as chunk_err:
                traceback.print_exc()
                errors.append({
                    "chunk_index": i,
                    "error": f"Unexpected: {chunk_err}",
                    "status_code": 500
                })
                if first_error is None:
                     first_error = chunk_err
                     print("Stopping: unexpected error.")
                     break # Exit the loop

        end_time_chunks = time.time()
        print(f"Chunk processing finished in {end_time_chunks - start_time_chunks:.2f}s. Processed: {processed_chunks_count}, Inserted: {inserted_count}, Errors: {len(errors)}")

        # --- 6. Remove Duplicates ---
        if inserted_count > 0 and not first_error:
            start_time_dedup = time.time()
            print(f"Checking for duplicates for URL: {request.pdf_url}...")
            try:
                pipeline = [
                    {"$match": {"metadata.original_url": str(request.pdf_url)}},
                    {"$group": {"_id": "$text", "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
                    {"$match": {"count": {"$gt": 1}}}
                ]
                duplicate_groups = list(collection.aggregate(pipeline))
                ids_to_delete = [oid for group in duplicate_groups for oid in group['ids'][1:]]
                if ids_to_delete:
                    print(f"Attempting to delete {len(ids_to_delete)} duplicate documents...")
                    res = collection.delete_many({"_id": {"$in": ids_to_delete}})
                    duplicates_removed_count = res.deleted_count
                    print(f"Deleted {duplicates_removed_count} duplicates.")
                else:
                    print("No duplicate documents found to remove.")
                print(f"Deduplication check done in {time.time() - start_time_dedup:.2f}s.")
            except Exception as agg_error:
                print(f"Deduplication error: {agg_error}")
                index_status = "duplicates_check_failed"
        elif first_error:
            index_status = "duplicates_skipped_error"
            print("Skipping duplicate removal due to processing errors.")
        else:
            index_status = "duplicates_skipped_no_inserts"
            print("Skipping duplicate removal as no chunks were inserted.")

        # --- 7. Manage Vector Index ---
        data_changed = inserted_count > 0 or duplicates_removed_count > 0
        if data_changed and not first_error:
            attempt_creation, index_dropped = True, False
            print(f"Data changed. Managing vector search index '{index_name}'...")
            start_time_index = time.time()
            try:
                existing = list(collection.list_search_indexes(index_name))
                index_exists = bool(existing)
                if index_exists:
                    print(f"Index '{index_name}' found. Attempting drop...")
                    try:
                        collection.drop_search_index(index_name)
                        wait = 20 # seconds
                        print(f"Waiting {wait}s for index drop...")
                        await asyncio.sleep(wait)
                        index_dropped = True
                        print("Proceeding to create new index.")
                    except Exception as drop_err:
                        print(f"Index drop error: {drop_err}")
                        index_status = f"failed_drop: {getattr(drop_err, 'codeName', '')}"
                        attempt_creation = False
                if attempt_creation:
                    print(f"Creating/recreating index '{index_name}'...")
                    dims = 1536 # For text-embedding-ada-002
                    index_def = {
                        "mappings": {
                            "dynamic": False,
                            "fields": {
                                "embedding": {"type": "knnVector", "dimensions": dims, "similarity": "cosine"},
                                "text": {"type": "string", "analyzer": "lucene.standard"}
                            }
                        }
                    }
                    model = {"name": index_name, "definition": index_def}
                    try:
                        collection.create_search_index(model)
                        index_status = f"{'re' if index_dropped else ''}created"
                        print(f"Index '{index_name}' creation initiated.")
                    except Exception as create_err:
                        print(f"Index create error: {create_err}")
                        index_status = f"failed_create: {getattr(create_err, 'codeName', '')}"
            except Exception as outer_idx_err:
                print(f"Index management setup error: {outer_idx_err}")
                if index_status == "not_checked": index_status = f"failed_mgmt_setup"
            print(f"Index management done in {time.time() - start_time_index:.2f}s. Status: {index_status}")
        elif first_error:
            index_status = "skipped_processing_error"
            print(f"Skipping index management due to critical error: {type(first_error).__name__}")
        else:
            index_status = "skipped_no_data_change"
            print("Skipping index management as no data changed.")

        # --- 8. Return Response ---
        processing_time = round(time.time() - start_time_total, 2)
        final_status_code = 200
        msg = "Processed successfully."
        if errors:
            if inserted_count > 0:
                final_status_code = 207 # Multi-Status
                msg = f"Processed with {len(errors)} errors out of {processed_chunks_count} chunks attempted."
            else:
                final_status_code = 400 # Assume client-side error initially
                msg = f"Processing failed. Encountered {len(errors)} errors on {processed_chunks_count} chunks."
                if first_error and hasattr(first_error, 'status_code') and first_error.status_code in [429, 500, 502, 503]:
                    final_status_code = first_error.status_code # Use critical error status
                elif first_error and isinstance(first_error, (OperationFailure, ConnectionFailure)):
                    final_status_code = 503 # DB errors are service unavailable

        resp_body = {
            "status": "success" if final_status_code == 200 else ("partial_success" if final_status_code == 207 else "failed"),
            "message": msg,
            "chunks_processed": processed_chunks_count,
            "chunks_inserted": inserted_count,
            "duplicates_removed": duplicates_removed_count,
            "vector_index_name": index_name,
            "vector_index_status": index_status,
            "processing_time_seconds": processing_time,
        }
        if errors:
            resp_body["errors_sample"] = errors[:10] # Show a sample of errors

        return JSONResponse(content=resp_body, status_code=final_status_code)
    except HTTPException as http_exc:
        raise http_exc # Re-raise caught HTTP exceptions
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during processing: {e}")


# /search uses Header auth
@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    user: Dict = Depends(get_user_header) # Use header-based auth
):
    """Performs hybrid search (vector + full-text RRF). Requires X-API-Key header."""
    vector_search_index_name = "vector_index"
    print(f"Received hybrid search request: query='{request.query[:50]}...', collection='{request.collection_name}', limit={request.limit}, user={user.get('supabase_user_id')}")
    try:
        # 1. Get User DB and Collection
        db = MongoDBManager.get_user_db(user)
        try:
            if request.collection_name not in db.list_collection_names():
                 raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found in database '{db.name}'.")
        except OperationFailure as e:
             print(f"Database error checking collection existence: {e.details}")
             raise HTTPException(status_code=503, detail=f"Database error accessing collection list: {e.details}")
        collection = db[request.collection_name]
        print(f"Searching in collection '{request.collection_name}' of database '{db.name}'")

        # 2. Generate Query Embedding
        try:
            print("Generating embedding for the search query...")
            start_time_embed = time.time()
            query_vector = await asyncio.to_thread(get_openai_embedding, request.query)
            end_time_embed = time.time()
            if not query_vector: raise HTTPException(status_code=400, detail="Could not generate embedding for the provided query.")
            print(f"Query embedding generated in {end_time_embed - start_time_embed:.2f}s.")
        except HTTPException as embed_exc: raise embed_exc
        except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail="Failed to generate query embedding due to an unexpected error.")

        # 3. Construct Atlas Search Aggregation Pipeline (Hybrid with RRF)
        num_candidates = max(request.limit * 10, min(request.num_candidates, 1000)); rrf_k = 60
        print(f"Using numCandidates: {num_candidates}")
        pipeline = [
            {"$vectorSearch": {"index": vector_search_index_name, "path": "embedding", "queryVector": query_vector, "numCandidates": num_candidates, "limit": num_candidates}},
            {"$group": {"_id": None, "docs": {"$push": {"doc": "$$ROOT", "vector_score": {"$meta": "vectorSearchScore"}}}}},
            {"$unwind": {"path": "$docs", "includeArrayIndex": "vector_rank_temp"}},
            {"$replaceRoot": {"newRoot": { "$mergeObjects": [ "$docs.doc", { "vector_rank": { "$add": [ "$vector_rank_temp", 1 ] } } ] } } },
            {"$project": {"_id": 1, "text": 1, "metadata": 1, "vector_rank": 1, "embedding": 0 }},
            {"$unionWith": {
                "coll": request.collection_name,
                "pipeline": [
                    {"$search": {"index": vector_search_index_name, "text": {"query": request.query, "path": "text"}}},
                    {"$limit": num_candidates},
                    {"$group": {"_id": None, "docs": {"$push": {"doc": "$$ROOT", "text_score": {"$meta": "searchScore"}}}}},
                    {"$unwind": {"path": "$docs", "includeArrayIndex": "text_rank_temp"}},
                    {"$replaceRoot": {"newRoot": { "$mergeObjects": [ "$docs.doc", { "text_rank": { "$add": [ "$text_rank_temp", 1 ] } } ] } } },
                    {"$project": {"_id": 1, "text": 1, "metadata": 1, "text_rank": 1, "embedding": 0 }}
                ]
            }},
            {"$group": {"_id": "$_id", "text": {"$first": "$text"}, "metadata": {"$first": "$metadata"}, "vector_rank": {"$min": "$vector_rank"}, "text_rank": {"$min": "$text_rank"}}},
            {"$addFields": {"rrf_score": {"$sum": [{"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$vector_rank"]}]}, 0]}, {"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$text_rank"]}]}, 0]}]}}},
            {"$sort": {"rrf_score": -1}},
            {"$limit": request.limit},
            {"$project": {"_id": 0, "id": {"$toString": "$_id"}, "text": 1, "metadata": 1, "score": "$rrf_score"}}
        ]

        # 4. Execute Aggregation Pipeline
        print("Executing hybrid search pipeline...")
        start_time_search = time.time()
        search_results = list(collection.aggregate(pipeline))
        end_time_search = time.time()
        print(f"Hybrid search completed in {end_time_search - start_time_search:.2f} seconds. Found {len(search_results)} results.")

        # 5. Return Formatted Results
        return SearchResponse(results=search_results)

    except OperationFailure as mongo_error:
        print(f"MongoDB Operation Failure during search: {mongo_error.details}")
        detail=f"DB operation failed: {mongo_error.details}"; code=500
        if "index not found" in str(detail).lower() or (hasattr(mongo_error, 'codeName') and mongo_error.codeName == 'IndexNotFound'):
            code=404; detail = f"Search index '{vector_search_index_name}' not found or not ready in collection '{request.collection_name}'. Ensure it has been created via the /process endpoint and is active."
        elif getattr(mongo_error, 'code', 0) == 13: code=403; detail="Authorization failed for search operation."
        raise HTTPException(status_code=code, detail=detail)
    except HTTPException as http_exc: raise http_exc
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during search: {e}")

# /vector-search uses Header auth
@app.post("/vector-search", response_model=VectorSearchResponse)
async def vector_search_documents(
    request: VectorSearchRequest,
    user: Dict = Depends(get_user_header) # Use header-based auth
):
    """Performs vector-only search. Requires X-API-Key header."""
    vector_search_index_name = "vector_index"
    print(f"Received vector search request: query='{request.query[:50]}...', collection='{request.collection_name}', limit={request.limit}, filter={request.filter}, user={user.get('supabase_user_id')}")
    try:
        # 1. Get User DB and Collection
        db = MongoDBManager.get_user_db(user)
        try:
            if request.collection_name not in db.list_collection_names(): raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found.")
        except OperationFailure as e: print(f"DB error checking collection: {e.details}"); raise HTTPException(status_code=503, detail=f"DB error listing collections: {e.details}")
        collection = db[request.collection_name]
        print(f"Vector searching in collection '{request.collection_name}' of db '{db.name}'")

        # 2. Generate Query Embedding
        try:
            print("Generating query embedding...")
            start_time_embed = time.time()
            query_vector = await asyncio.to_thread(get_openai_embedding, request.query)
            end_time_embed = time.time()
            if not query_vector: raise HTTPException(status_code=400, detail="Could not generate query embedding.")
            print(f"Query embedding generated in {end_time_embed - start_time_embed:.2f}s.")
        except HTTPException as embed_exc: raise embed_exc
        except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail="Failed query embedding: unexpected error.")

        # 3. Construct Atlas Vector Search Pipeline
        print(f"Using numCandidates: {request.num_candidates}")
        vs_stage = { "$vectorSearch": { "index": vector_search_index_name, "path": "embedding", "queryVector": query_vector, "numCandidates": request.num_candidates, "limit": request.limit } }
        if request.filter:
            print(f"Applying filter: {request.filter}")
            vs_stage["$vectorSearch"]["filter"] = request.filter
        pipeline = [ vs_stage, {"$project": {"_id": 0, "id": {"$toString": "$_id"}, "text": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}} ]

        # 4. Execute Pipeline
        print("Executing vector search pipeline...")
        start_time_search = time.time()
        search_results = list(collection.aggregate(pipeline))
        end_time_search = time.time()
        print(f"Vector search done in {end_time_search - start_time_search:.2f}s. Found {len(search_results)} results.")

        # 5. Return Results
        return VectorSearchResponse(results=search_results)
    except OperationFailure as mongo_error:
        print(f"DB Operation Failure during vector search: {mongo_error.details}")
        detail=f"DB error: {mongo_error.details}"; code=500
        if "index not found" in str(detail).lower() or (hasattr(mongo_error, 'codeName') and mongo_error.codeName == 'IndexNotFound'):
            code=404; detail = f"Index '{vector_search_index_name}' not found."
        elif getattr(mongo_error, 'code', 0) == 13: code=403; detail="Auth failed for search."
        raise HTTPException(status_code=code, detail=detail)
    except HTTPException as http_exc: raise http_exc
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected vector search error: {e}")

# /user-collections uses body for API key check internally
@app.post("/user-collections", response_model=UserInfoResponse)
def get_user_collections(request: UserInfoRequest):
    """
    Retrieves the database name and list of collection names associated with a user,
    verified by their Supabase User ID and API key in the body.
    """
    print(f"Received request for user collections: supabase_user_id='{request.supabase_user_id}'")
    if not request.api_key: raise HTTPException(status_code=400, detail="API key is required.")
    if auth_db is None or mongo_client is None: raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        user = auth_db.users.find_one({"supabase_user_id": request.supabase_user_id})
        if not user: raise HTTPException(status_code=404, detail="User not found for the provided Supabase User ID.")
        print("User found. Verifying API key...")
        if not secrets.compare_digest(user.get("api_key", ""), request.api_key): raise HTTPException(status_code=403, detail="API key could be wrong. Please check API key again.")
        db_name = user.get("db_name")
        if not db_name: raise HTTPException(status_code=500, detail="Internal Server Error: User data is incomplete.")
        print(f"API key verified. Accessing user database: {db_name}")
        try:
            user_db = mongo_client[db_name]; collection_names = user_db.list_collection_names()
            print(f"Found collections in {db_name}: {collection_names}")
            return UserInfoResponse(db_name=db_name, collections=collection_names)
        except (OperationFailure, ConnectionFailure) as db_err: raise HTTPException(status_code=503, detail=f"Database operation failed accessing user data: {getattr(db_err, 'details', str(db_err))}")
        except Exception as db_err: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"An unexpected error occurred accessing user database: {db_err}")
    except OperationFailure as auth_op_fail: raise HTTPException(status_code=503, detail=f"Database operation failed searching for user: {auth_op_fail.details}")
    except HTTPException as e: raise e
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")


# --- ★★★ NEW ENDPOINTS ★★★ ---

@app.delete("/collections", response_model=ActionResponse)
async def delete_collection_endpoint(
    request: DeleteCollectionRequest,
    user: Dict = Depends(get_user_body) # Use body-based auth dependency
):
    """
    Deletes a specified collection within the user's database.
    Requires API key and collection name in the request body.
    WARNING: This operation is irreversible.
    """
    collection_name = request.collection_name
    print(f"Request to delete collection '{collection_name}' for user {user.get('supabase_user_id')}")
    try:
        db = MongoDBManager.get_user_db(user)
        db_name = db.name
        # Check if collection exists before attempting to drop
        collection_names = db.list_collection_names()
        if collection_name not in collection_names:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")

        print(f"Attempting to drop collection '{collection_name}' in db '{db_name}'...")
        db.drop_collection(collection_name)
        print(f"Successfully initiated drop for collection '{collection_name}'.")
        # Optional: Verify drop by listing collections again (might have delay)
        # if collection_name in db.list_collection_names():
        #    print(f"Warning: Collection '{collection_name}' still listed after drop command.")
        #    return ActionResponse(status="warning", message=f"Drop command issued for '{collection_name}', but it might still exist briefly.")

        return ActionResponse(status="success", message=f"Collection '{collection_name}' deleted successfully.")
    except OperationFailure as e:
        print(f"DB Operation Failure deleting collection '{collection_name}': {e.details}")
        # Provide more specific error if possible, e.g., permissions error
        raise HTTPException(status_code=500, detail=f"Database operation failed to delete collection: {e.details}")
    except ConnectionFailure as e:
        print(f"DB Connection Failure during collection drop '{collection_name}': {e}")
        raise HTTPException(status_code=503, detail=f"Database connection lost during deletion.")
    except HTTPException as e: raise e # Re-raise specific exceptions (like 404, 403 from get_user_body)
    except Exception as e:
        print(f"Unexpected error deleting collection '{collection_name}': {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during deletion: {e}")


@app.put("/collections", response_model=ActionResponse)
async def rename_collection_endpoint(
    request: RenameCollectionRequest,
    user: Dict = Depends(get_user_body) # Use body-based auth dependency
):
    """
    Renames a specified collection within the user's database.
    Requires API key, current name, and new name in the request body.
    Note: Associated Atlas Search indexes are NOT automatically migrated and must be recreated.
    """
    current_name = request.current_name
    new_name = request.new_name
    print(f"Request to rename collection '{current_name}' to '{new_name}' for user {user.get('supabase_user_id')}")

    if current_name == new_name:
        raise HTTPException(status_code=400, detail="New name cannot be the same as the current name.")
    # Basic validation for new name (redundant with pattern but explicit)
    if not re.match(r"^[a-zA-Z0-9_.-]+$", new_name):
        raise HTTPException(status_code=400, detail="New collection name contains invalid characters.")

    try:
        db = MongoDBManager.get_user_db(user)
        db_name = db.name
        collection_names = db.list_collection_names()
        if current_name not in collection_names:
            raise HTTPException(status_code=404, detail=f"Source collection '{current_name}' not found.")
        if new_name in collection_names:
            raise HTTPException(status_code=409, detail=f"Target collection name '{new_name}' already exists.")

        print(f"Attempting rename '{current_name}' to '{new_name}' in db '{db_name}'...")
        # MongoDB rename is an admin command, typically executed on the 'admin' database
        # Format: client.admin.command('renameCollection', f'{db_name}.{current_name}', to=f'{db_name}.{new_name}')
        try:
            mongo_client.admin.command(
                'renameCollection',
                f'{db_name}.{current_name}',
                to=f'{db_name}.{new_name}'
                # dropTarget=False # Default is False. If True, would drop target if it exists (we check above)
            )
            print(f"Successfully renamed '{current_name}' to '{new_name}'.")
            warning_message = ("Important: Rename successful, but any Atlas Search indexes associated with the old name must be manually dropped and recreated for the new collection name.")
            return ActionResponse(
                status="success",
                message=f"Collection '{current_name}' renamed to '{new_name}' successfully.",
                details=warning_message
                )
        except OperationFailure as e:
            # Handle specific rename errors
            print(f"DB Operation Failure renaming '{current_name}' to '{new_name}': {e.details} (Code: {e.code})")
            detail = f"DB operation failed rename: {e.details}"; code=500
            if e.code == 2: # Example: Database error (might cover various issues)
                 detail = f"Database error during rename: {e.details}"
            elif e.code == 13: # Example: Unauthorized
                 code=403; detail = "Authorization error: Insufficient permissions to rename collection."
            elif e.code == 72: # InvalidNamespace
                 code=400; detail = f"Invalid target name format '{new_name}'."
            elif e.code == 10026: # NamespaceExists (should be caught by earlier check but handle defensively)
                 code=409; detail = f"Target collection '{new_name}' exists (OpFail Code 10026)."
            elif e.code == 26: # NamespaceNotFound (should be caught by earlier check)
                 code=404; detail = f"Source collection '{current_name}' not found (OpFail Code 26)."
            raise HTTPException(status_code=code, detail=detail)

    except ConnectionFailure as e:
        print(f"DB Connection Failure during rename '{current_name}': {e}")
        raise HTTPException(status_code=503, detail=f"Database connection lost during rename.")
    except HTTPException as e: raise e # Re-raise specific exceptions (like 404, 403, 409)
    except Exception as e:
        # Catch other unexpected errors (e.g., during list_collection_names)
        print(f"Unexpected error renaming '{current_name}': {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during rename: {e}")


# --- Application startup ---
if __name__ == "__main__":
    # Import re for the rename validation pattern
    import re
    if mongo_client is None or auth_db is None or openai_client is None:
         print("FATAL: Required clients not initialized properly.")
         if mongo_client is None: print(" - MongoDB Client is None")
         if auth_db is None: print(" - MongoDB Auth DB is None")
         if openai_client is None: print(" - OpenAI Client is None")
         raise SystemExit("Server cannot start due to initialization failures.")
    print("Starting FastAPI server on host 0.0.0.0, port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)