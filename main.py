# -*- coding: utf-8 -*-
import os
import io
import secrets
from datetime import datetime
from typing import List, Optional, Dict, Any
import traceback # エラー詳細表示のため
import re # rename エンドポイントの検証用に追加

import requests
import PyPDF2
import uvicorn
# Path をインポート
from fastapi import FastAPI, HTTPException, Header, Depends, Body, Response, Path
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

class UserCollectionsRequest(BaseModel):
    supabase_user_id: str = Field(..., min_length=1, description="The Supabase User ID")

class UserInfoResponse(BaseModel):
    db_name: str
    collections: List[str]

# ★★★ DeleteCollectionBody は不要 ★★★

# コレクション名変更用リクエストモデル (Body用、APIキーなし)
class RenameCollectionBody(BaseModel):
    new_name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="New name for the collection")

# 汎用レスポンスモデル
class ActionResponse(BaseModel):
    status: str
    message: str
    details: Optional[str] = None


# --- Dependencies ---
def verify_admin(api_key: str = Header(..., alias="X-API-Key")):
    """Verifies the admin API key provided in the header."""
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required")

# HeaderからAPIキーを取得して認証する依存関係
def get_user_header(api_key: str = Header(..., alias="X-API-Key")):
    """
    Retrieves the user based on the API key provided in the X-API-Key header.
    Used for endpoints requiring user authentication via header.
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
         raise e # Raise already crafted HTTPExceptions
    except Exception as e:
        print(f"Unexpected error in get_user_header: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected error occurred during API key header validation.")

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
    words = text.split()
    if not words: return []
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        end_pos = current_pos; current_length = 0; last_valid_end_pos = current_pos
        while end_pos < len(words):
            word_len = len(words[end_pos]); length_to_add = word_len + (1 if end_pos > current_pos else 0)
            if current_length + length_to_add <= chunk_size: current_length += length_to_add; last_valid_end_pos = end_pos + 1; end_pos += 1
            else: break
        if last_valid_end_pos == current_pos:
             if current_pos == 0 and len(words[0]) > chunk_size: print(f"Warn: Word > chunk size: '{words[0][:50]}...'"); chunks.append(words[0]); current_pos += 1; continue
             if current_pos < len(words): last_valid_end_pos = current_pos + 1
             else: break
        chunk_words = words[current_pos:last_valid_end_pos]; chunks.append(" ".join(chunk_words))
        overlap_start_index = last_valid_end_pos - 1; overlap_char_count = 0
        while overlap_start_index > current_pos:
             overlap_char_count += len(words[overlap_start_index]) + 1
             if overlap_char_count >= overlap: break
             overlap_start_index -= 1
        overlap_start_index = max(current_pos, overlap_start_index)
        if overlap_start_index > current_pos: current_pos = overlap_start_index
        else: current_pos = last_valid_end_pos
    return chunks

# --- OpenAI Embedding Function ---
def get_openai_embedding(text: str) -> List[float]:
    if not text or text.isspace(): print("Warn: Embedding empty text."); return []
    try:
        cleaned_text = ' '.join(text.split())
        if not cleaned_text: print("Warn: Text empty after cleaning."); return []
        response = openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=cleaned_text)
        if response.data and response.data[0].embedding: return response.data[0].embedding
        else: print(f"Warn: OpenAI no embedding data: {cleaned_text[:100]}..."); raise HTTPException(status_code=500, detail="OpenAI empty data.")
    except openai.APIConnectionError as e: print(f"OpenAI Conn Error: {e}"); raise HTTPException(status_code=503, detail=f"OpenAI Conn fail: {e}")
    except openai.APIStatusError as e:
        print(f"OpenAI Status Error: Code {e.status_code}, Resp: {e.response}"); status_code = e.status_code
        detail = f"OpenAI Err (Status {status_code}): {e.message or str(e.response)}"
        if status_code == 400: detail = f"OpenAI Bad Req: {e.message}"; raise HTTPException(status_code=400, detail=detail)
        elif status_code == 401: detail = "OpenAI Auth Err. Check Key."; raise HTTPException(status_code=401, detail=detail)
        elif status_code == 429: detail = "OpenAI Rate Limit."; raise HTTPException(status_code=429, detail=detail)
        elif status_code >= 500: detail = f"OpenAI Server Err ({status_code}). {e.message}"; raise HTTPException(status_code=502, detail=detail)
        else: raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e: print(f"Unexpected embed error: {type(e).__name__} - {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Embed unexpected fail: {e}")

# ... Search models remain the same ...
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
        try: mongo_client.admin.command('ping')
        except (ConnectionFailure, OperationFailure) as e: db_status = f"error ({type(e).__name__})"
    return {"status": "ok", "database": db_status, "openai_client": openai_status}

@app.get("/auth-db", dependencies=[Depends(verify_admin)])
def get_auth_db_contents():
    if auth_db is None: raise HTTPException(status_code=503, detail="DB service unavailable")
    try: users = list(auth_db.users.find({}, {"_id": 0, "api_key": 0})); return {"users": users}
    except OperationFailure as e: print(f"Err reading auth_db: {e.details}"); raise HTTPException(status_code=500, detail=f"Failed user data: {e.details}")
    except Exception as e: print(f"Unexpected auth_db read err: {type(e).__name__}"); traceback.print_exc(); raise HTTPException(status_code=500, detail="Unexpected error user data.")

@app.post("/register", response_model=RegisterResponse)
def register_user(request: UserRegister, response: Response):
    if auth_db is None: raise HTTPException(status_code=503, detail="DB service unavailable")
    try:
        existing_user = auth_db.users.find_one({"supabase_user_id": request.supabase_user_id})
        if existing_user:
            api_key = existing_user.get("api_key"); db_name = existing_user.get("db_name")
            if not api_key or not db_name: raise HTTPException(status_code=500, detail="Inconsistent user data.")
            response.status_code = 200; return RegisterResponse(api_key=api_key, db_name=db_name, database_exist=True)
        else:
            db_info = MongoDBManager.create_user_db(request.supabase_user_id)
            response.status_code = 201; return RegisterResponse(api_key=db_info["api_key"], db_name=db_info["db_name"], database_exist=False)
    except HTTPException as e: raise e
    except OperationFailure as e: raise HTTPException(status_code=500, detail=f"DB registration error: {e.details}")
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected registration error: {e}")

# POST /collections uses Header auth
@app.post("/collections", status_code=201)
def create_collection_endpoint(request: CollectionCreate, user: Dict = Depends(get_user_header)):
    """Creates new collection. Requires X-API-Key header."""
    try:
        db = MongoDBManager.get_user_db(user)
        collection = MongoDBManager.create_collection(db, request.name, user["supabase_user_id"])
        return {"status": "created", "collection_name": collection.name}
    except HTTPException as e: raise e
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Failed collection creation: {e}")

# /process uses Header auth
@app.post("/process")
async def process_pdf(request: ProcessRequest, user: Dict = Depends(get_user_header)):
    """Downloads, processes PDF, stores embeddings. Requires X-API-Key header."""
    index_name = "vector_index" # ★ インデックス名を定義
    index_status = "not_checked" # ★ 初期ステータス
    first_error = None
    duplicates_removed_count = 0
    inserted_count = 0
    processed_chunks_count = 0
    errors = []
    start_time_total = time.time()
    print(f"Process PDF request for user {user.get('supabase_user_id', 'N/A')}: URL={request.pdf_url}, Collection='{request.collection_name}'")

    try:
        # --- 1. Download PDF ---
        try:
            start_time_download = time.time()
            # Use asyncio.to_thread for blocking requests call
            response = await asyncio.to_thread(requests.get, str(request.pdf_url), timeout=60)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            end_time_download = time.time()
            print(f"PDF downloaded successfully ({end_time_download - start_time_download:.2f}s).")

            content_length = response.headers.get('Content-Length')
            pdf_bytes = response.content
            pdf_size = len(pdf_bytes)

            if pdf_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"PDF file size ({pdf_size / (1024 * 1024):.2f}MB) exceeds the maximum limit ({MAX_FILE_SIZE / (1024 * 1024)}MB).")

            pdf_content = io.BytesIO(pdf_bytes)
            size_source = f"Content-Length header: {content_length}" if content_length else "checked post-download"
            print(f"PDF size: {pdf_size / (1024 * 1024):.2f} MB ({size_source}).")

        except requests.exceptions.Timeout:
            raise HTTPException(status_code=408, detail="Request timed out while downloading the PDF.")
        except requests.exceptions.SSLError as ssl_err:
             # Provide more context for SSL errors if possible
             raise HTTPException(status_code=502, detail=f"SSL error occurred during PDF download: {ssl_err}. Check the server's SSL certificate or network configuration.")
        except requests.exceptions.RequestException as req_error:
             status_code = 502 if isinstance(req_error, (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError)) else 400
             detail = f"Failed to download PDF from URL: {req_error}"
             if hasattr(req_error, 'response') and req_error.response is not None:
                 detail += f" (Status Code: {req_error.response.status_code})"
             print(f"PDF download failed: {detail}")
             raise HTTPException(status_code=status_code, detail=detail)

        # --- 2. Extract text ---
        try:
            start_time_extract = time.time()
            # Use asyncio.to_thread for blocking PyPDF2 call
            def extract_text_sync(pdf_stream):
                 try:
                     reader = PyPDF2.PdfReader(pdf_stream)
                     if reader.is_encrypted:
                         # PyPDF2 can sometimes handle basic password-protected PDFs if password provided,
                         # but generally, encrypted PDFs are problematic.
                         # Consider adding password handling if needed, otherwise raise error.
                         raise ValueError("Processing encrypted PDFs is not supported.")
                     full_text = ""
                     for page in reader.pages:
                         page_text = page.extract_text()
                         if page_text: # Add text only if extraction was successful for the page
                             full_text += page_text + "\n" # Add newline between pages
                     return full_text
                 except PyPDF2.errors.PdfReadError as pdf_err:
                     # Catch specific PyPDF2 read errors
                     raise ValueError(f"Invalid or corrupted PDF file: {pdf_err}") from pdf_err
                 except Exception as inner_e:
                     # Catch other potential errors during extraction
                     raise RuntimeError(f"Error during text extraction: {inner_e}") from inner_e

            text = await asyncio.to_thread(extract_text_sync, pdf_content)
            end_time_extract = time.time()

            if not text or text.isspace():
                print("No text could be extracted from the PDF or the PDF was empty.")
                # Return success but indicate no processing occurred
                return JSONResponse(
                    content={
                        "status": "success",
                        "message": "PDF processed, but no text was extracted.",
                        "chunks_processed": 0,
                        "chunks_inserted": 0,
                        "duplicates_removed": 0,
                        "vector_index_name": index_name,
                        "vector_index_status": "skipped_no_text",
                        "processing_time_seconds": round(time.time() - start_time_total, 2)
                    },
                    status_code=200
                )
            print(f"Text extracted successfully ({len(text)} characters, {end_time_extract - start_time_extract:.2f}s).")

        except ValueError as e: # Catch specific errors like encryption or invalid PDF
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e: # Catch other extraction errors
             print(f"Text extraction runtime error: {e}")
             traceback.print_exc()
             raise HTTPException(status_code=500, detail=f"Failed during text extraction phase: {e}")
        except Exception as e: # Catch any other unexpected errors
            print(f"Unexpected error during text extraction setup: {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during text extraction: {e}")

        # --- 3. Split text ---
        start_time_split = time.time()
        chunks = split_text_into_chunks(text, chunk_size=1500, overlap=100) # Use defined function
        end_time_split = time.time()
        print(f"Text split into {len(chunks)} chunks ({end_time_split - start_time_split:.2f}s).")

        if not chunks:
            print("Text extracted but resulted in zero chunks after splitting.")
            # Return success but indicate no chunks to process
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "PDF processed and text extracted, but no processable chunks were generated.",
                    "chunks_processed": 0,
                    "chunks_inserted": 0,
                    "duplicates_removed": 0,
                    "vector_index_name": index_name,
                    "vector_index_status": "skipped_no_chunks",
                    "processing_time_seconds": round(time.time() - start_time_total, 2)
                },
                status_code=200
            )

        # --- 4. DB Setup ---
        db = MongoDBManager.get_user_db(user)
        collection = MongoDBManager.create_collection(db, request.collection_name, user["supabase_user_id"])
        print(f"Using database '{db.name}' and collection '{collection.name}'.")

        # --- 5. Process Chunks (Embedding and Insertion) ---
        start_time_chunks = time.time()
        print(f"Starting processing for {len(chunks)} text chunks...")

        # Process chunks concurrently using asyncio.gather (optional, adds complexity)
        # Or process sequentially as shown below:
        for i, chunk in enumerate(chunks):
            # Basic check to skip empty or whitespace-only chunks
            if not chunk or chunk.isspace():
                print(f"Skipping empty chunk at index {i}.")
                continue

            processed_chunks_count += 1 # Count chunks attempted for processing

            try:
                # --- Get Embedding ---
                start_time_embed = time.time()
                # Use asyncio.to_thread for blocking OpenAI call
                embedding = await asyncio.to_thread(get_openai_embedding, chunk)
                end_time_embed = time.time()

                if not embedding:
                    # Should not happen if get_openai_embedding raises HTTPException on failure, but as a safeguard:
                    print(f"Warning: Skipping chunk {i+1}/{len(chunks)} due to empty embedding result.")
                    errors.append({"chunk_index": i, "error": "Received empty embedding without error", "status_code": 500})
                    continue # Skip this chunk

                # --- Prepare Document ---
                doc = {
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {
                        **request.metadata, # User-provided metadata
                        "chunk_index": i,
                        "original_url": str(request.pdf_url), # Store original URL
                        "processed_at": datetime.utcnow()
                    },
                    "created_at": datetime.utcnow() # Record creation time
                }

                # --- Insert Document ---
                start_time_insert = time.time()
                insert_result = collection.insert_one(doc)
                end_time_insert = time.time()

                if insert_result.inserted_id:
                    inserted_count += 1
                    # Log progress periodically or for each chunk if needed
                    if (i + 1) % 10 == 0 or (i + 1) == len(chunks): # Log every 10 chunks and the last one
                         print(f"Processed chunk {i+1}/{len(chunks)} (Embed: {end_time_embed-start_time_embed:.2f}s, Insert: {end_time_insert-start_time_insert:.2f}s)")
                else:
                    # This case is unlikely with insert_one unless an error is missed
                    print(f"Warning: Chunk {i+1}/{len(chunks)} insertion did not return an ID.")
                    errors.append({"chunk_index": i, "error": "MongoDB insert_one did not return an inserted_id", "status_code": 500})

            except HTTPException as e:
                # Handle errors from get_openai_embedding (e.g., OpenAI API errors, rate limits)
                print(f"Error processing chunk {i+1}/{len(chunks)}: HTTP {e.status_code} - {e.detail}")
                errors.append({"chunk_index": i, "error": e.detail, "status_code": e.status_code})
                # Decide if the error is critical enough to stop processing
                is_critical_error = e.status_code in [401, 403, 429, 500, 502, 503] # Example critical errors
                if is_critical_error and first_error is None:
                    print(f"Stopping processing due to critical error: {e.status_code}")
                    first_error = e # Store the first critical error encountered
                    break # Stop processing further chunks

            except (OperationFailure, ConnectionFailure) as db_error:
                # Handle MongoDB specific errors during insertion
                error_detail = getattr(db_error, 'details', str(db_error))
                print(f"Database error processing chunk {i+1}/{len(chunks)}: {type(db_error).__name__} - {error_detail}")
                errors.append({"chunk_index": i, "error": f"Database operation/connection failed: {error_detail}", "status_code": 503})
                if first_error is None:
                    print("Stopping processing due to critical database error.")
                    first_error = db_error
                    break # Stop processing further chunks

            except Exception as e:
                # Handle any other unexpected errors during chunk processing
                print(f"Unexpected error processing chunk {i+1}/{len(chunks)}: {type(e).__name__} - {e}")
                traceback.print_exc()
                errors.append({"chunk_index": i, "error": f"Unexpected error: {str(e)}", "status_code": 500})
                if first_error is None:
                    print("Stopping processing due to unexpected critical error.")
                    first_error = e
                    break # Stop processing further chunks

        end_time_chunks = time.time()
        print(f"Finished chunk processing ({end_time_chunks - start_time_chunks:.2f}s). "
              f"Attempted: {processed_chunks_count}, Inserted: {inserted_count}, Errors: {len(errors)}")

        # --- 6. Remove Duplicates (based on text content and URL) ---
        # Run deduplication only if new chunks were inserted and no critical error stopped processing early
        if inserted_count > 0 and not first_error:
            start_time_dedup = time.time()
            print(f"Starting duplicate check for documents from URL: {request.pdf_url}...")
            duplicates_removed_count = 0 # Initialize counter
            try:
                # Pipeline to find duplicate text within the same URL context
                pipeline = [
                    {"$match": {"metadata.original_url": str(request.pdf_url)}}, # Filter by URL
                    {"$group": {
                        "_id": "$text",  # Group by the text content
                        "ids": {"$push": "$_id"}, # Collect all document IDs for this text
                        "count": {"$sum": 1}     # Count occurrences
                    }},
                    {"$match": {"count": {"$gt": 1}}} # Filter groups with more than one document (duplicates)
                ]
                duplicate_groups = list(collection.aggregate(pipeline))

                ids_to_delete = []
                for group in duplicate_groups:
                    # Keep the first ID, mark the rest for deletion
                    ids_to_delete.extend(group['ids'][1:])

                if ids_to_delete:
                    print(f"Found {len(ids_to_delete)} duplicate document(s) to remove.")
                    delete_result = collection.delete_many({"_id": {"$in": ids_to_delete}})
                    duplicates_removed_count = delete_result.deleted_count
                    print(f"Successfully removed {duplicates_removed_count} duplicate document(s).")
                else:
                    print("No duplicate documents found for this URL.")

                print(f"Duplicate check and removal finished ({time.time() - start_time_dedup:.2f}s).")
                index_status = "duplicates_checked" # Indicate check was performed

            except (OperationFailure, ConnectionFailure) as db_error:
                 print(f"Database error during duplicate removal: {type(db_error).__name__} - {getattr(db_error, 'details', str(db_error))}")
                 index_status = "duplicates_check_failed_db_error"
            except Exception as e:
                 print(f"Unexpected error during duplicate removal: {type(e).__name__} - {e}")
                 traceback.print_exc()
                 index_status = "duplicates_check_failed_unexpected"
        elif first_error:
            index_status = "duplicates_skipped_due_to_error"
            print("Skipping duplicate removal because processing stopped due to an error.")
        else: # inserted_count == 0
            index_status = "duplicates_skipped_no_inserts"
            print("Skipping duplicate removal as no new documents were inserted.")


        # --- ★ 7. Manage Vector Search Index (Based on previous successful logic) ---
        data_changed = inserted_count > 0 or duplicates_removed_count > 0
        # Only manage index if data changed and no critical errors occurred during processing
        if data_changed and not first_error:
            attempt_creation = True # Flag to control creation attempt
            index_dropped = False   # Flag to track if index was dropped
            start_time_index = time.time()
            print(f"Data has changed (Inserted: {inserted_count}, Duplicates Removed: {duplicates_removed_count}). Managing vector search index '{index_name}'...")

            try:
                # 7a. Check for existing index (using list_search_indexes and filtering in Python)
                print(f"Checking for existing vector search index named '{index_name}'...")
                # Fetch all search indexes for the collection
                existing_indexes = list(collection.list_search_indexes())
                index_exists = any(idx.get('name') == index_name for idx in existing_indexes)

                # 7b. Drop index if it exists
                if index_exists:
                    print(f"Index '{index_name}' found. Attempting to drop it...")
                    try:
                        collection.drop_search_index(index_name)
                        wait_time = 20 # Seconds to wait for drop operation to propagate (adjust if needed)
                        print(f"Successfully initiated drop for index '{index_name}'. Waiting {wait_time}s for propagation...")
                        await asyncio.sleep(wait_time) # Use asyncio.sleep in async context
                        index_dropped = True
                        print("Proceeding to create new index after waiting.")
                    except OperationFailure as drop_err:
                        code_name = getattr(drop_err, 'codeName', 'UnknownCode')
                        print(f"MongoDB Operation Failure dropping index '{index_name}': CodeName={code_name}, Details={drop_err.details}")
                        index_status = f"failed_drop_operation_{code_name}"
                        attempt_creation = False # Do not attempt creation if drop failed
                    except Exception as drop_err:
                        print(f"Unexpected error dropping index '{index_name}': {type(drop_err).__name__} - {drop_err}")
                        traceback.print_exc()
                        index_status = f"failed_drop_unexpected"
                        attempt_creation = False # Do not attempt creation if drop failed unexpectedly
                else:
                    print(f"Index '{index_name}' not found. Will create a new one.")

                # 7c. Create the index if conditions allow
                if attempt_creation:
                    print(f"Attempting to create vector search index '{index_name}'...")
                    # Define the index structure (ensure dimensions match your embedding model)
                    index_definition = {
                        "mappings": {
                            "dynamic": False, # Explicitly define fields to index
                            "fields": {
                                "embedding": {
                                    "type": "knnVector",
                                    "dimensions": 1536, # For OpenAI text-embedding-ada-002
                                    "similarity": "cosine" # Or "euclidean" / "dotProduct"
                                },
                                "text": { # For hybrid search ($search stage)
                                    "type": "string",
                                    "analyzer": "lucene.standard" # Standard analyzer
                                    # Consider "lucene.kuromoji" for Japanese, etc.
                                }
                                # Add metadata fields here if they need to be searchable via $search
                                # "metadata.field_name": { "type": "string" }
                            }
                        }
                    }
                    # Use the 'model' parameter format as in the previous successful code
                    search_index_model = {"name": index_name, "definition": index_definition}
                    try:
                        collection.create_search_index(model=search_index_model)
                        # Set status based on whether it was created or recreated
                        index_status = f"{'re' if index_dropped else ''}created"
                        print(f"Index creation {'initiated' if not index_dropped else 're-initiated'} for '{index_name}'. It may take some time for the index to become fully queryable.")
                    except OperationFailure as create_err:
                        code_name = getattr(create_err, 'codeName', 'UnknownCode')
                        print(f"MongoDB Operation Failure creating index '{index_name}': CodeName={code_name}, Details={create_err.details}")
                        index_status = f"failed_create_operation_{code_name}"
                    except Exception as create_err:
                        print(f"Unexpected error creating index '{index_name}': {type(create_err).__name__} - {create_err}")
                        traceback.print_exc()
                        index_status = f"failed_create_unexpected"

                # Handle case where creation was skipped due to drop failure
                elif not attempt_creation and index_status.startswith("failed_drop"):
                    print(f"Index creation skipped because the drop operation failed. Final index status: {index_status}")
                else: # Should not happen unless logic error
                     print(f"WARN: Index creation was not attempted for an unexpected reason. Status: {index_status}")
                     if index_status == "not_checked": index_status = "skipped_unknown_reason"


            except OperationFailure as list_err:
                 # Handle errors during list_search_indexes
                 code_name = getattr(list_err, 'codeName', 'UnknownCode')
                 print(f"MongoDB Operation Failure listing search indexes: CodeName={code_name}, Details={list_err.details}")
                 index_status = f"failed_list_indexes_{code_name}"
            except ConnectionFailure as conn_err:
                 # Handle connection errors during index management
                 print(f"MongoDB Connection Failure during index management: {conn_err}")
                 index_status = "failed_connection_during_index_mgmt"
            except Exception as outer_idx_err:
                # Catch any other errors during the index management setup phase
                print(f"Error during index management setup phase: {type(outer_idx_err).__name__} - {outer_idx_err}")
                traceback.print_exc()
                # Ensure index_status reflects this failure if not already set
                if index_status == "not_checked" or index_status == "duplicates_checked": # If status hasn't recorded an index op failure yet
                    index_status = f"failed_management_setup"

            # Final check for status (should always be set by now)
            if index_status == "not_checked" or index_status == "duplicates_checked":
                 index_status = "status_logic_error" # Should not happen
                 print(f"WARN: Index status logic might have an issue. Final status: {index_status}")

            print(f"Index management finished ({time.time() - start_time_index:.2f}s). Final Status: {index_status}")

        # --- Handle cases where index management was skipped ---
        elif first_error:
            index_status = "skipped_due_to_processing_error"
            print(f"Skipping index management due to critical error during chunk processing (Error: {type(first_error).__name__}).")
        else: # data_changed is False (no inserts or removals)
            index_status = "skipped_no_data_change"
            print("Skipping index management as no data was inserted or removed.")


        # --- 8. Return Response (Adjusted based on previous logic and current context) ---
        processing_time = round(time.time() - start_time_total, 2)
        final_status_code = 200 # Default to success
        response_status = "success"
        message = "PDF processed successfully."

        if errors:
            # If there were errors, adjust status and message
            if inserted_count > 0:
                # Partial success if some chunks were inserted despite errors
                final_status_code = 207 # Partial Content
                response_status = "partial_success"
                message = f"Processed {processed_chunks_count} chunks with {len(errors)} errors. {inserted_count} chunks inserted."
            else:
                # Failed if errors occurred and no chunks were inserted
                # Determine if it's a client error (4xx) or server error (5xx) based on first critical error
                if first_error and isinstance(first_error, HTTPException) and 400 <= first_error.status_code < 500:
                     final_status_code = first_error.status_code
                elif first_error: # Includes DB errors or 5xx HTTP errors or unexpected errors
                    final_status_code = 500 # Or maybe 503 for DB? Let's use 500 generally here.
                else: # Errors occurred but no single critical one identified to stop early? Default to 400.
                    final_status_code = 400 # Bad Request assumed if processing failed without inserts
                response_status = "failed"
                message = f"Processing failed with {len(errors)} errors. No chunks were inserted."

            # If a critical error forced processing to stop, ensure the status code reflects it
            if first_error:
                 critical_error_message = f" Processing stopped early due to critical error: {type(first_error).__name__}."
                 if isinstance(first_error, HTTPException):
                     if first_error.status_code >= 500: final_status_code = first_error.status_code # Prioritize 5xx
                     elif first_error.status_code == 429: final_status_code = 429 # Prioritize 429
                     elif final_status_code < 400: final_status_code = first_error.status_code # Use 4xx if not already error
                     critical_error_message += f" (HTTP {first_error.status_code}: {first_error.detail})"
                 elif isinstance(first_error, (OperationFailure, ConnectionFailure)):
                     final_status_code = 503 # Service Unavailable for DB errors
                     critical_error_message += f" (DB Error: {getattr(first_error, 'details', str(first_error))})"
                 else: # Unexpected error
                     if final_status_code < 500: final_status_code = 500 # Ensure it's a server error
                     critical_error_message += f" (Error: {str(first_error)})"
                 message += critical_error_message
                 # Update response_status based on final code
                 if final_status_code >= 500 or final_status_code == 429: response_status = "failed"
                 elif final_status_code == 207: response_status = "partial_success"
                 elif final_status_code >= 400: response_status = "failed" # e.g. 400, 401, 403, 413


        response_body = {
            "status": response_status,
            "message": message,
            "chunks_processed": processed_chunks_count, # How many chunks were *attempted*
            "chunks_inserted": inserted_count,        # How many *successfully* inserted
            "duplicates_removed": duplicates_removed_count,
            "vector_index_name": index_name,
            "vector_index_status": index_status, # The detailed status from step 7
            "processing_time_seconds": processing_time,
        }
        # Include a sample of errors if any occurred
        if errors:
            response_body["errors_sample"] = [
                {
                    "chunk_index": e.get("chunk_index", "N/A"),
                    "status_code": e.get("status_code", 500),
                    "error": str(e.get("error", "Unknown error")) # Ensure error is string
                 }
                 for e in errors[:10] # Limit sample size
            ]

        print(f"Responding with status code {final_status_code}. Index status: {index_status}")
        return JSONResponse(content=response_body, status_code=final_status_code)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions raised intentionally in the process
        print(f"Caught HTTPException: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch-all for any other unexpected errors in the main try block
        print(f"Unexpected top-level error in '/process' endpoint: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during PDF processing: {str(e)}")

# /search uses Header auth
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, user: Dict = Depends(get_user_header)):
    """Performs hybrid search. Requires X-API-Key header."""
    idx_name = "vector_index"; print(f"Hybrid search req: q='{request.query[:50]}...', coll='{request.collection_name}', user={user.get('supabase_user_id')}")
    try:
        db = MongoDBManager.get_user_db(user)
        try:
            if request.collection_name not in db.list_collection_names(): raise HTTPException(status_code=404, detail=f"Coll '{request.collection_name}' not found.")
        except OperationFailure as e: print(f"DB err list coll: {e.details}"); raise HTTPException(status_code=503, detail=f"DB err listing coll: {e.details}")
        collection = db[request.collection_name]
        try: q_vec = await asyncio.to_thread(get_openai_embedding, request.query); assert q_vec, "Empty query embedding"
        except HTTPException as e: raise e
        except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail="Query embed fail.")
        num_cand = max(request.limit * 10, min(request.num_candidates, 1000)); rrf_k = 60; print(f"NumCandidates: {num_cand}")
        pipeline = [ {"$vectorSearch": {"index": idx_name, "path": "embedding", "queryVector": q_vec, "numCandidates": num_cand, "limit": num_cand}}, {"$group": {"_id": None, "docs": {"$push": {"doc": "$$ROOT", "vector_score": {"$meta": "vectorSearchScore"}}}}}, {"$unwind": {"path": "$docs", "includeArrayIndex": "vr_tmp"}}, {"$replaceRoot": {"newRoot": {"$mergeObjects": ["$docs.doc", {"vr": {"$add": ["$vr_tmp", 1]}}]}}}, {"$project": {"_id": 1, "text": 1, "metadata": 1, "vr": 1, "embedding": 0}}, {"$unionWith": {"coll": request.collection_name, "pipeline": [{"$search": {"index": idx_name, "text": {"query": request.query, "path": "text"}}}, {"$limit": num_cand}, {"$group": {"_id": None, "docs": {"$push": {"doc": "$$ROOT", "text_score": {"$meta": "searchScore"}}}}}, {"$unwind": {"path": "$docs", "includeArrayIndex": "tr_tmp"}}, {"$replaceRoot": {"newRoot": {"$mergeObjects": ["$docs.doc", {"tr": {"$add": ["$tr_tmp", 1]}}]}}}, {"$project": {"_id": 1, "text": 1, "metadata": 1, "tr": 1, "embedding": 0}}]}}, {"$group": {"_id": "$_id", "text": {"$first": "$text"}, "metadata": {"$first": "$metadata"}, "vr": {"$min": "$vr"}, "tr": {"$min": "$tr"}}}, {"$addFields": {"rrf_score": {"$sum": [{"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$vr"]}]}, 0]}, {"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$tr"]}]}, 0]}]}}}, {"$sort": {"rrf_score": -1}}, {"$limit": request.limit}, {"$project": {"_id": 0, "id": {"$toString": "$_id"}, "text": 1, "metadata": 1, "score": "$rrf_score"}} ]
        start = time.time(); results = list(collection.aggregate(pipeline)); end = time.time()
        print(f"Hybrid search done ({end - start:.2f}s). Found {len(results)} results.")
        return SearchResponse(results=results)
    except OperationFailure as e:
        print(f"DB search error: {e.details}"); detail=f"DB error: {e.details}"; code=500
        if "index not found" in str(detail).lower() or getattr(e, 'codeName', '') == 'IndexNotFound': code=404; detail = f"Index '{idx_name}' missing/not ready."
        elif getattr(e, 'code', 0) == 13: code=403; detail="Auth fail search."
        raise HTTPException(status_code=code, detail=detail)
    except HTTPException as e: raise e
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected search error: {e}")

# /vector-search uses Header auth
@app.post("/vector-search", response_model=VectorSearchResponse)
async def vector_search_documents(request: VectorSearchRequest, user: Dict = Depends(get_user_header)):
    """Performs vector-only search. Requires X-API-Key header."""
    idx_name = "vector_index"; print(f"Vector search req: q='{request.query[:50]}...', coll='{request.collection_name}', filter={request.filter}, user={user.get('supabase_user_id')}")
    try:
        db = MongoDBManager.get_user_db(user)
        try:
            if request.collection_name not in db.list_collection_names(): raise HTTPException(status_code=404, detail=f"Coll '{request.collection_name}' not found.")
        except OperationFailure as e: print(f"DB err list coll: {e.details}"); raise HTTPException(status_code=503, detail=f"DB err listing coll: {e.details}")
        collection = db[request.collection_name]
        try: q_vec = await asyncio.to_thread(get_openai_embedding, request.query); assert q_vec, "Empty query embedding"
        except HTTPException as e: raise e
        except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail="Query embed fail.")
        vs_stage = {"$vectorSearch": {"index": idx_name, "path": "embedding", "queryVector": q_vec, "numCandidates": request.num_candidates, "limit": request.limit}}
        if request.filter: print(f"Applying filter: {request.filter}"); vs_stage["$vectorSearch"]["filter"] = request.filter
        pipeline = [vs_stage, {"$project": {"_id": 0, "id": {"$toString": "$_id"}, "text": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}}]
        start = time.time(); results = list(collection.aggregate(pipeline)); end = time.time()
        print(f"Vector search done ({end - start:.2f}s). Found {len(results)} results.")
        return VectorSearchResponse(results=results)
    except OperationFailure as e:
        print(f"DB vector search error: {e.details}"); detail=f"DB error: {e.details}"; code=500
        if "index not found" in str(detail).lower() or getattr(e, 'codeName', '') == 'IndexNotFound': code=404; detail = f"Index '{idx_name}' missing/not ready."
        elif getattr(e, 'code', 0) == 13: code=403; detail="Auth fail search."
        raise HTTPException(status_code=code, detail=detail)
    except HTTPException as e: raise e
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected vector search error: {e}")

# /user-collections uses Header auth
@app.post("/user-collections", response_model=UserInfoResponse)
def get_user_collections_header_auth(request: UserCollectionsRequest, user: Dict = Depends(get_user_header)):
    """Retrieves user DB name and collections. Requires X-API-Key header and matching Supabase User ID in body."""
    auth_user_id = user.get("supabase_user_id"); req_user_id = request.supabase_user_id
    print(f"Req collections for {req_user_id}, authenticated as {auth_user_id}")
    if auth_user_id != req_user_id: raise HTTPException(status_code=403, detail="Auth user mismatch request ID.")
    if mongo_client is None: raise HTTPException(status_code=503, detail="DB service unavailable")
    try:
        db_name = user.get("db_name")
        if not db_name: raise HTTPException(status_code=500, detail="User data incomplete (no db_name).")
        print(f"Accessing DB: {db_name} for user {auth_user_id}")
        try:
            user_db = mongo_client[db_name]; coll_names = user_db.list_collection_names()
            print(f"Found collections: {coll_names}")
            return UserInfoResponse(db_name=db_name, collections=coll_names)
        except (OperationFailure, ConnectionFailure) as e: raise HTTPException(status_code=503, detail=f"DB access error: {getattr(e, 'details', str(e))}")
        except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected DB access error: {e}")
    except HTTPException as e: raise e
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# --- ★★★ Collection Management Endpoints (Header Auth) ★★★ ---

@app.delete("/collections/{collection_name}", response_model=ActionResponse)
async def delete_collection_endpoint(
    collection_name: str = Path(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Name of the collection to delete"),
    user: Dict = Depends(get_user_header) # Depends は最後に
):
    """
    Deletes specified collection identified by path parameter.
    Requires X-API-Key header. WARNING: Irreversible.
    """
    print(f"Request delete coll '{collection_name}' for user {user.get('supabase_user_id')}")
    try:
        db = MongoDBManager.get_user_db(user); db_name = db.name
        if collection_name not in db.list_collection_names():
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")

        print(f"Dropping coll '{collection_name}' in db '{db_name}'...")
        db.drop_collection(collection_name)
        print(f"Drop initiated for '{collection_name}'.")
        return ActionResponse(status="success", message=f"Collection '{collection_name}' deleted.")
    except OperationFailure as e:
        print(f"DB delete error: {e.details}")
        raise HTTPException(status_code=500, detail=f"DB delete fail: {e.details}")
    except ConnectionFailure as e:
        print(f"DB conn fail delete: {e}")
        raise HTTPException(status_code=503, detail=f"DB conn lost delete.")
    except HTTPException as e: raise e
    except Exception as e:
        print(f"Unexpected delete err: {type(e).__name__}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected delete error: {e}")

@app.put("/collections/{current_name}", response_model=ActionResponse)
async def rename_collection_endpoint(
    # ★★★ パラメータの順序修正 ★★★
    current_name: str = Path(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Current name of the collection"),
    request: RenameCollectionBody = Body(...), # Body(...) を明示
    user: Dict = Depends(get_user_header)      # Depends は最後に
):
    """
    Renames specified collection identified by path parameter. Requires X-API-Key header & new_name in body.
    Note: Atlas Search indexes need manual recreation.
    """
    new_name = request.new_name
    print(f"Request rename '{current_name}' to '{new_name}' for user {user.get('supabase_user_id')}")

    if current_name == new_name:
        raise HTTPException(status_code=400, detail="New name same as current.")
    # new_name validation by Pydantic model

    try:
        db = MongoDBManager.get_user_db(user); db_name = db.name
        coll_names = db.list_collection_names()
        if current_name not in coll_names:
            raise HTTPException(status_code=404, detail=f"Source collection '{current_name}' not found.")
        if new_name in coll_names:
            raise HTTPException(status_code=409, detail=f"Target collection name '{new_name}' already exists.")

        print(f"Renaming '{current_name}' to '{new_name}' in db '{db_name}'...")
        try:
            if mongo_client is None: raise HTTPException(status_code=503, detail="DB client unavailable for rename.")
            mongo_client.admin.command('renameCollection', f'{db_name}.{current_name}', to=f'{db_name}.{new_name}')
            print(f"Rename successful.")
            warning = "Important: Manually recreate Atlas Search indexes for the new name."
            return ActionResponse(status="success", message=f"Renamed '{current_name}' to '{new_name}'.", details=warning)
        except OperationFailure as e:
            print(f"DB rename error: {e.details} (Code: {e.code})"); detail = f"DB rename fail: {e.details}"; code=500
            if e.code == 13: code=403; detail = "Auth error: Insufficient permissions."
            elif e.code == 72: code=400; detail = f"Invalid target name '{new_name}'."
            elif e.code == 10026: code=409; detail = f"Target '{new_name}' exists (OpFail 10026)."
            elif e.code == 26: code=404; detail = f"Source '{current_name}' not found (OpFail 26)."
            raise HTTPException(status_code=code, detail=detail)
    except ConnectionFailure as e: print(f"DB conn fail rename: {e}"); raise HTTPException(status_code=503, detail=f"DB conn lost rename.")
    except HTTPException as e: raise e
    except Exception as e: print(f"Unexpected rename err: {type(e).__name__}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected rename error: {e}")

# --- Application startup ---
if __name__ == "__main__":
    if mongo_client is None or auth_db is None or openai_client is None:
         print("FATAL: Required clients not initialized properly."); sys_exit = True
         if mongo_client is None: print(" - MongoDB Client is None")
         if auth_db is None: print(" - MongoDB Auth DB is None")
         if openai_client is None: print(" - OpenAI Client is None")
         raise SystemExit("Server cannot start due to initialization failures.")
    print("Starting FastAPI server on host 0.0.0.0, port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)