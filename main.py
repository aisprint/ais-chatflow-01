# -*- coding: utf-8 -*-
import os
import io
import secrets
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple # Tuple を追加
import traceback # エラー詳細表示のため
import re # rename エンドポイントの検証用に追加
import sys # SystemExit用に追加
import json # JSON処理用に追加

import requests
import PyPDF2
import uvicorn
# Path をインポート
from fastapi import FastAPI, HTTPException, Header, Depends, Body, Response, Path
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.errors import OperationFailure, ConnectionFailure # MongoDB固有のエラー
from pydantic import BaseModel, HttpUrl, Field, field_validator # field_validatorを追加
# from sentence_transformers import SentenceTransformer # 不要
import openai # ★ OpenAIライブラリをインポート
from openai import OpenAI # ★ 新しいAPI呼び出し形式 (v1.0.0以降)
from dotenv import load_dotenv
import asyncio
import time

from typing import List, Optional # Optionalを追加
from pydantic import BaseModel, Field, HttpUrl # BaseModel, Field は既存だが明示
from bson import ObjectId # 結果のID変換用

# ★★★ CORSミドルウェアをインポート ★★★
from fastapi.middleware.cors import CORSMiddleware


# FastAPI app initialization
app = FastAPI()

# ★★★ CORSミドルウェアの設定を追加 ★★★
# 許可するオリジン (開発中は "*" で全て許可し、本番環境ではフロントエンドのドメインを指定)
origins = [
    "*",  # すべてのオリジンを許可 (テスト用)
    # "http://localhost",
    # "http://localhost:3000", # フロントエンドが動作しているオリジンなど
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 許可するオリジンのリスト
    allow_credentials=True,  # クレデンシャル（Cookie、Authorizationヘッダーなど）を許可
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 許可するHTTPメソッド (DELETEとOPTIONSを含む)
    allow_headers=["*", "X-API-Key"],  # 許可するHTTPヘッダー (X-API-Keyを含む)
)


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
        serverSelectionTimeoutMS=5000, # 5 seconds timeout for server selection
        connectTimeoutMS=3000 # 3 seconds timeout for initial connection
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
    traceback.print_exc()
    raise SystemExit(f"Unexpected MongoDB connection error: {e}")


# --- Data models ---
class UserRegister(BaseModel):
    supabase_user_id: str = Field(..., min_length=1)

class RegisterResponse(BaseModel):
    api_key: str
    db_name: str
    database_exist: bool = Field(..., description="True if the user database already existed, False if newly created.")

class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$")

class ProcessRequest(BaseModel):
    pdf_url: HttpUrl # Note: This field will be used for PDF, TXT, or JSON URLs
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UserCollectionsRequest(BaseModel):
    supabase_user_id: str = Field(..., min_length=1, description="The Supabase User ID")

class UserInfoResponse(BaseModel):
    db_name: str
    collections: List[str]

class RenameCollectionBody(BaseModel):
    new_name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="New name for the collection")

class ActionResponse(BaseModel):
    status: str
    message: str
    details: Optional[str] = None

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
    collection_names: List[str] = Field(..., min_length=1, description="List of collection names to search within (minimum 1)")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return across all collections")
    num_candidates: int = Field(100, ge=10, le=1000, description="Number of candidates for initial vector search phase within EACH collection")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filter criteria for metadata (applied within each collection)")

    @field_validator('collection_names', mode='before')
    @classmethod
    def check_collection_names(cls, v):
        if not isinstance(v, list):
            raise ValueError("collection_names must be a list")
        if not v:
            raise ValueError("collection_names must not be empty")
        pattern = r"^[a-zA-Z0-9_.-]+$"
        for name in v:
            if not isinstance(name, str) or not re.match(pattern, name):
                raise ValueError(f"Invalid collection name format: '{name}'. Must match pattern: {pattern}")
        return v

class VectorSearchResponse(BaseModel):
    results: List[SearchResultItem]
    warnings: Optional[List[str]] = Field(None, description="List of warnings, e.g., collections not found or search errors")

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

class CollectionStats(BaseModel):
    ns: str = Field(..., description="Namespace (database.collection)")
    count: int = Field(..., description="Total number of documents (chunks) in the collection")
    size: int = Field(..., description="Total size of the documents in bytes (excluding indexes)")
    avgObjSize: Optional[float] = Field(None, description="Average document size in bytes")
    storageSize: int = Field(..., description="Total size allocated for the collection on disk in bytes")
    nindexes: int = Field(..., description="Number of indexes on the collection")
    totalIndexSize: int = Field(..., description="Total size of all indexes in bytes")

class CollectionStatsResponse(BaseModel):
    collection_name: str
    stats: CollectionStats
    distinct_source_urls_count: Optional[int] = Field(None, description="Number of distinct source PDF URLs processed into this collection. Can be null if check fails or no metadata found.")
    message: Optional[str] = None

# --- Dependencies ---
def verify_admin(api_key: str = Header(..., alias="X-API-Key")):
    if not ADMIN_API_KEY:
         print("CRITICAL: ADMIN_API_KEY is not set in the environment.")
         raise HTTPException(status_code=500, detail="Server configuration error: Admin key missing.")
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required. Invalid API Key provided.")

def get_user_header(api_key: str = Header(..., alias="X-API-Key")):
    if auth_db is None:
         print("Error in get_user_header: auth_db is not available.")
         raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        if not api_key:
             print("Error in get_user_header: X-API-Key header is missing.")
             raise HTTPException(status_code=401, detail="X-API-Key header is required.")
        user = auth_db.users.find_one({"api_key": api_key})
        if not user:
            raise HTTPException(status_code=403, detail="Invalid API Key provided in X-API-Key header.")
        return user
    except OperationFailure as e:
        print(f"Database error finding user in auth_db (get_user_header): {e.details}")
        raise HTTPException(status_code=503, detail="Database operation failed while validating API key header.")
    except HTTPException as e:
         raise e
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
            print(f"Successfully inserted user ({supabase_user_id}) record into auth_db, db_name: {db_name}, MongoDB ObjectId: {result.inserted_id}")
            return {"api_key": api_key, "db_name": db_name}
        except OperationFailure as e:
            print(f"MongoDB Operation Failure creating user record for {supabase_user_id}: {e.details}")
            raise HTTPException(status_code=500, detail=f"Database operation failed during user registration: {e}")
        except Exception as e:
            print(f"Error inserting user {supabase_user_id} into auth_db: {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to create user record due to an unexpected error: {e}")

    @staticmethod
    def get_user_db(user: Dict):
        if mongo_client is None:
             print("Error in get_user_db: mongo_client is not available.")
             raise HTTPException(status_code=503, detail="Database service unavailable")
        db_name = user.get("db_name")
        if not db_name:
            print(f"Error in get_user_db: User object is missing 'db_name'. User Supabase ID: {user.get('supabase_user_id', 'N/A')}")
            raise HTTPException(status_code=500, detail="Internal server error: User data is inconsistent (missing database name).")
        try:
            return mongo_client[db_name]
        except Exception as e:
            print(f"Unexpected error accessing user database '{db_name}': {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=503, detail=f"Failed to access user database '{db_name}': {e}")

    @staticmethod
    def create_collection(db, name: str, user_id: str):
        if db is None:
             print(f"Error in create_collection: Database object is None for user {user_id}.")
             raise HTTPException(status_code=500, detail="Internal Server Error: Invalid database reference passed.")
        try:
            collection_names = db.list_collection_names()
            if name not in collection_names:
                 print(f"Collection '{name}' does not exist in database '{db.name}'. It will be created implicitly on first write operation.")
                 return db[name]
            else:
                 print(f"Collection '{name}' already exists in database '{db.name}'.")
                 return db[name]
        except OperationFailure as e:
            print(f"MongoDB Operation Failure accessing/listing collection '{name}' in database '{db.name}': {e.details}")
            raise HTTPException(status_code=500, detail=f"Database operation failed while checking for collection '{name}': {e.details}")
        except ConnectionFailure as e:
            print(f"MongoDB Connection Failure during collection check for '{name}' in database '{db.name}': {e}")
            raise HTTPException(status_code=503, detail=f"Database connection lost while checking for collection '{name}'.")
        except Exception as e:
            print(f"Error accessing or preparing collection '{name}' in database '{db.name}': {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to ensure collection '{name}' exists due to an unexpected error: {e}")

# --- Text splitting helper ---
def split_text_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 100) -> List[str]:
    if not text: return []
    words = text.split()
    if not words: return []
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
                 print(f"Warning: A single word is longer than the chunk size ({chunk_size}). Taking the full word: '{words[0][:50]}...'")
                 chunks.append(words[0])
                 current_pos += 1
                 continue
            elif current_pos < len(words):
                print(f"Warning: Word at index {current_pos} ('{words[current_pos][:50]}...') might exceed chunk size but forcing inclusion to progress.")
                last_valid_end_pos = current_pos + 1
            else:
                 break
        chunk_words = words[current_pos:last_valid_end_pos]
        chunks.append(" ".join(chunk_words))
        if last_valid_end_pos >= len(words):
             break
        overlap_start_index = last_valid_end_pos - 1
        overlap_char_count = 0
        while overlap_start_index > current_pos:
             overlap_char_count += len(words[overlap_start_index]) + 1
             if overlap_char_count >= overlap:
                 break
             overlap_start_index -= 1
        overlap_start_index = max(current_pos, overlap_start_index)
        if overlap_start_index > current_pos:
             current_pos = overlap_start_index
        else:
             current_pos = last_valid_end_pos
    return chunks

# --- OpenAI Embedding Function ---
def get_openai_embedding(text: str) -> List[float]:
    if not text or text.isspace():
        print("Warning: Attempted to get embedding for empty or whitespace-only text.")
        return []
    try:
        cleaned_text = ' '.join(text.split()).strip()
        if not cleaned_text:
             print("Warning: Text became empty after cleaning whitespace.")
             return []
        response = openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=cleaned_text)
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            print(f"Warning: OpenAI API returned success but no embedding data for text: {cleaned_text[:100]}...")
            raise HTTPException(status_code=500, detail="OpenAI API returned no embedding data unexpectedly.")
    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to OpenAI API: {e}")
    except openai.RateLimitError as e:
        print(f"OpenAI API Rate Limit Error: {e}")
        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. Please try again later.")
    except openai.AuthenticationError as e:
        print(f"OpenAI API Authentication Error: {e}")
        raise HTTPException(status_code=401, detail="OpenAI API authentication failed. Check your API key.")
    except openai.PermissionDeniedError as e:
         print(f"OpenAI API Permission Denied Error: {e}")
         raise HTTPException(status_code=403, detail="OpenAI API permission denied. Check your API key's permissions.")
    except openai.NotFoundError as e:
         print(f"OpenAI API Not Found Error (check model name?): {e}")
         raise HTTPException(status_code=404, detail=f"OpenAI resource not found. Check model name '{OPENAI_EMBEDDING_MODEL}'. Error: {e}")
    except openai.BadRequestError as e:
         print(f"OpenAI API Bad Request Error: {e}")
         raise HTTPException(status_code=400, detail=f"OpenAI API bad request. Check input data. Error: {e}")
    except openai.APIStatusError as e:
        print(f"OpenAI API Status Error: Status Code={e.status_code}, Response={e.response}")
        status_code = e.status_code
        detail = f"OpenAI API Error (Status {status_code}): {e.message or str(e.response)}"
        if status_code == 400: detail = f"OpenAI Bad Request: {e.message}"; raise HTTPException(status_code=400, detail=detail)
        elif status_code == 401: detail = "OpenAI Authentication Error. Check API Key."; raise HTTPException(status_code=401, detail=detail)
        elif status_code == 403: detail = "OpenAI Permission Denied. Check Key Permissions."; raise HTTPException(status_code=403, detail=detail)
        elif status_code == 404: detail = f"OpenAI Resource Not Found (Model?). Error: {e.message}"; raise HTTPException(status_code=404, detail=detail)
        elif status_code == 429: detail = "OpenAI Rate Limit Exceeded."; raise HTTPException(status_code=429, detail=detail)
        elif status_code >= 500: detail = f"OpenAI Server Error ({status_code}). {e.message}"; raise HTTPException(status_code=502, detail=detail)
        else: raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        print(f"Unexpected error during OpenAI embedding generation: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating text embedding: {e}")

# --- ★★★ Helper functions for /process endpoint ★★★ ---
ENCODING_TRY_ORDER = ['utf-8', 'shift_jis', 'euc_jp', 'cp932', 'latin-1']

def decode_bytes_with_fallbacks(file_bytes: bytes, try_order: List[str] = None) -> str:
    """Decodes bytes to string using a list of encodings, falling back on failure."""
    if try_order is None:
        try_order = ENCODING_TRY_ORDER
    
    decoded_text = None
    last_exception = None

    for enc in try_order:
        try:
            decoded_text = file_bytes.decode(enc)
            print(f"Successfully decoded using {enc}")
            return decoded_text # Return on first success
        except UnicodeDecodeError as e:
            last_exception = e
            print(f"Failed to decode with {enc}: {e}")
            continue
        except Exception as e: # Catch other potential errors like LookupError
            last_exception = e
            print(f"Error with encoding {enc} (likely not a UnicodeDecodeError): {e}")
            continue
    
    # If all attempts fail, raise an informative exception
    error_message = f"Failed to decode file content after trying encodings: {', '.join(try_order)}. Last error: {last_exception}"
    # Consider raising HTTPException directly if this is only used in request context
    raise ValueError(error_message)


def _recursive_json_extractor(data_node: Any, parent_url: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Recursively extracts text segments and associated metadata (especially URLs) from JSON.
    Yields a list of (text_content, metadata_dict_with_url).
    """
    extracted_items: List[Tuple[str, Dict[str, Any]]] = []

    if isinstance(data_node, dict):
        current_text_val = data_node.get("text")
        specific_url = data_node.get("url")
        
        # Determine the URL to be associated with text found at this level or passed to children
        effective_url_for_node = None
        if isinstance(specific_url, str) and specific_url.startswith(("http://", "https://")):
            effective_url_for_node = specific_url
        elif parent_url:
            effective_url_for_node = parent_url

        # If "text" key exists and is a string, treat it as a primary text segment
        if isinstance(current_text_val, str) and current_text_val.strip():
            meta = {}
            if effective_url_for_node:
                 meta["source_json_url"] = effective_url_for_node
            extracted_items.append((current_text_val, meta))
        
        # Recursively process other values in the dictionary
        for key, value in data_node.items():
            # Skip "text" if it was already processed as primary segment
            if key == "text" and isinstance(current_text_val, str):
                continue
            # Skip "url" string itself from being treated as text content
            if key == "url" and isinstance(value, str):
                continue
            
            # Pass the effective_url_for_node as the parent_url for children
            extracted_items.extend(_recursive_json_extractor(value, effective_url_for_node))

    elif isinstance(data_node, list):
        for item in data_node:
            # For items in a list, the parent_url context is inherited
            extracted_items.extend(_recursive_json_extractor(item, parent_url))
    elif isinstance(data_node, str) and data_node.strip():
        # This is a string not directly from a "text" key (or "text" was not primary)
        # Associate it with the parent_url from its context
        meta = {}
        if parent_url:
            meta["source_json_url"] = parent_url
        extracted_items.append((data_node, meta))
    
    return extracted_items


# --- API endpoints ---
@app.get("/health")
def health_check():
    db_status = "disconnected"; openai_status = "not_initialized"
    if mongo_client and auth_db:
        try: mongo_client.admin.command('ping'); db_status = "connected"
        except (ConnectionFailure, OperationFailure) as e: print(f"Health check: MongoDB ping failed: {e}"); db_status = f"error ({type(e).__name__})"
        except Exception as e: print(f"Health check: Unexpected error during MongoDB ping: {e}"); db_status = "error (unexpected)"
    if openai_client: openai_status = "initialized"
    all_ok = db_status == "connected" and openai_status.startswith("initialized")
    return JSONResponse(status_code=200 if all_ok else 503, content={"status": "ok" if all_ok else "error", "details": {"database": db_status, "openai_client": openai_status}})

@app.get("/auth-db", dependencies=[Depends(verify_admin)])
def get_auth_db_contents():
    if auth_db is None: raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        users = list(auth_db.users.find({}, {"_id": 0, "api_key": 0}))
        return {"users": users}
    except OperationFailure as e: print(f"Admin Error: Database error reading auth_db: {e.details}"); raise HTTPException(status_code=500, detail=f"Failed to retrieve user data: {e.details}")
    except Exception as e: print(f"Admin Error: Unexpected error reading auth_db: {type(e).__name__}"); traceback.print_exc(); raise HTTPException(status_code=500, detail="Unexpected error retrieving user data.")

@app.post("/register", response_model=RegisterResponse)
def register_user(request: UserRegister, response: Response):
    if auth_db is None: raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        existing_user = auth_db.users.find_one({"supabase_user_id": request.supabase_user_id})
        if existing_user:
            api_key, db_name = existing_user.get("api_key"), existing_user.get("db_name")
            if not api_key or not db_name: print(f"Error: Inconsistent data for existing user {request.supabase_user_id}."); raise HTTPException(status_code=500, detail="Internal server error: Inconsistent user data.")
            response.status_code = 200
            return RegisterResponse(api_key=api_key, db_name=db_name, database_exist=True)
        else:
            print(f"Registering new user: {request.supabase_user_id}")
            db_info = MongoDBManager.create_user_db(request.supabase_user_id)
            response.status_code = 201
            return RegisterResponse(api_key=db_info["api_key"], db_name=db_info["db_name"], database_exist=False)
    except HTTPException as e: raise e
    except OperationFailure as e: print(f"Database error during user registration for {request.supabase_user_id}: {e.details}"); raise HTTPException(status_code=500, detail=f"Database operation failed: {e.details}")
    except Exception as e: print(f"Unexpected error during user registration for {request.supabase_user_id}: {type(e).__name__} - {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/collections", status_code=201, response_model=ActionResponse)
def create_collection_endpoint(request: CollectionCreate, user: Dict = Depends(get_user_header)):
    try:
        db = MongoDBManager.get_user_db(user)
        collection = MongoDBManager.create_collection(db, request.name, user["supabase_user_id"])
        return ActionResponse(status="success", message=f"Collection '{collection.name}' is ready.")
    except HTTPException as e: raise e
    except Exception as e: print(f"Error creating collection '{request.name}' for user {user.get('supabase_user_id', 'N/A')}: {type(e).__name__} - {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Failed to prepare collection: {e}")


# ★★★ /process endpoint (Modified for TXT/JSON, advanced JSON parsing, and encoding fallbacks) ★★★
@app.post("/process")
async def process_pdf(request: ProcessRequest, user: Dict = Depends(get_user_header)):
    index_name = "vector_index"
    index_status = "not_checked"
    first_error = None
    duplicates_removed_count = 0
    inserted_count = 0
    processed_chunks_count = 0 # This will be the global chunk counter across all segments
    errors = []
    start_time_total = time.time()
    file_url_for_logging = str(request.pdf_url) # Original URL from request
    print(f"Process file request for user {user.get('supabase_user_id', 'N/A')}: URL={file_url_for_logging}, Collection='{request.collection_name}'")

    try:
        # --- 1. Download File ---
        try:
            start_time_download = time.time()
            # Use pdf_url from request, which can now be PDF, TXT, or JSON URL
            response = await asyncio.to_thread(requests.get, str(request.pdf_url), timeout=60)
            response.raise_for_status()
            end_time_download = time.time(); print(f"File downloaded ({end_time_download - start_time_download:.2f}s).")
            
            file_bytes = response.content
            file_size = len(file_bytes)
            if file_size > MAX_FILE_SIZE: raise HTTPException(status_code=413, detail=f"File size ({file_size/(1024*1024):.2f}MB) > limit ({MAX_FILE_SIZE/(1024*1024)}MB).")
            print(f"File size: {file_size/(1024*1024):.2f} MB.")
        except requests.exceptions.Timeout: raise HTTPException(status_code=408, detail="File download timeout.")
        except requests.exceptions.SSLError as ssl_err: raise HTTPException(status_code=502, detail=f"SSL error during file download: {ssl_err}.")
        except requests.exceptions.RequestException as req_error:
             status_code = 502 if isinstance(req_error, (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError)) else 400
             detail = f"File download failed: {req_error}" + (f" (Status: {req_error.response.status_code})" if hasattr(req_error, 'response') and req_error.response else "")
             raise HTTPException(status_code=status_code, detail=detail)

        # --- 2. Determine File Type and Prepare Texts for Processing ---
        start_time_extract = time.time()
        texts_to_process: List[Tuple[str, Dict[str, Any]]] = [] # List of (text_segment, segment_specific_metadata)
        detected_file_type = None

        content_type_header = response.headers.get('Content-Type', '').lower()
        file_url_str = str(request.pdf_url) 

        if 'application/pdf' in content_type_header or file_url_str.lower().endswith('.pdf'):
            detected_file_type = 'pdf'
        elif 'text/plain' in content_type_header or file_url_str.lower().endswith('.txt'):
            detected_file_type = 'txt'
        elif 'application/json' in content_type_header or \
             'application/ld+json' in content_type_header or \
             file_url_str.lower().endswith('.json') or \
             file_url_str.lower().endswith('.jsonl'):
            detected_file_type = 'json'
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type. URL: '{file_url_str}', Content-Type: '{content_type_header}'. Supported: PDF, TXT, JSON.")
        
        print(f"Detected file type: {detected_file_type.upper()}")

        # Extract text based on file type
        if detected_file_type == 'pdf':
            try:
                pdf_content_stream = io.BytesIO(file_bytes)
                def extract_text_from_pdf_sync(pdf_stream_sync):
                    reader = PyPDF2.PdfReader(pdf_stream_sync)
                    if reader.is_encrypted: raise ValueError("Encrypted PDFs not supported.")
                    return "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
                
                pdf_text = await asyncio.to_thread(extract_text_from_pdf_sync, pdf_content_stream)
                if pdf_text and not pdf_text.isspace():
                    texts_to_process.append((pdf_text, {})) # No segment-specific metadata for whole PDF text
            except PyPDF2.errors.PdfReadError as pdf_err: raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF: {pdf_err}")
            except ValueError as e: raise HTTPException(status_code=400, detail=str(e)) # For encryption error
            except Exception as e: print(f"PDF processing error: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")

        elif detected_file_type == 'txt':
            try:
                txt_text = decode_bytes_with_fallbacks(file_bytes)
                if txt_text and not txt_text.isspace():
                    texts_to_process.append((txt_text, {}))
            except ValueError as e: # From decode_bytes_with_fallbacks
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e: print(f"TXT processing error: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"TXT processing failed: {e}")

        elif detected_file_type == 'json':
            try:
                json_string_content = decode_bytes_with_fallbacks(file_bytes)
                json_data = json.loads(json_string_content)
                texts_to_process = _recursive_json_extractor(json_data) # Returns List[Tuple[str, Dict]]
            except ValueError as e: # From decode_bytes_with_fallbacks
                raise HTTPException(status_code=400, detail=str(e))
            except json.JSONDecodeError as e_json:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e_json}")
            except Exception as e: print(f"JSON processing error: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"JSON processing failed: {e}")
        
        end_time_extract = time.time()

        if not texts_to_process or all(not text_seg.strip() for text_seg, _ in texts_to_process):
            print(f"No processable text extracted from {detected_file_type.upper()} file.")
            return JSONResponse(content={
                "status": "success", "message": f"File ({detected_file_type.upper()}) processed, but no text extracted or all segments empty.",
                "chunks_processed": 0, "chunks_inserted": 0, "duplicates_removed": 0,
                "vector_index_name": index_name, "vector_index_status": "skipped_no_text",
                "processing_time_seconds": round(time.time() - start_time_total, 2)
            }, status_code=200)
        
        num_segments = len(texts_to_process)
        total_chars = sum(len(ts) for ts, _ in texts_to_process)
        print(f"Text extraction resulted in {num_segments} segment(s) from {detected_file_type.upper()} ({total_chars} total chars, {end_time_extract - start_time_extract:.2f}s).")

        # --- 3. DB Setup ---
        db = MongoDBManager.get_user_db(user); collection = MongoDBManager.create_collection(db, request.collection_name, user["supabase_user_id"])
        print(f"Using database '{db.name}' and collection '{collection.name}'.")

        # --- 4. Process Segments and Chunks (Split, Embed, Insert) ---
        start_time_chunk_processing_all = time.time()
        
        all_processed_chunks_for_dedup = [] # Store text of all chunks for this file to check for duplicates

        for segment_idx, (text_segment, segment_meta) in enumerate(texts_to_process):
            if not text_segment or text_segment.isspace():
                print(f"Skipping empty text segment {segment_idx + 1}/{num_segments}.")
                continue

            start_time_split = time.time()
            chunks = split_text_into_chunks(text_segment)
            end_time_split = time.time()
            print(f"Segment {segment_idx + 1}/{num_segments} split into {len(chunks)} chunks ({end_time_split - start_time_split:.2f}s).")

            if not chunks:
                print(f"Segment {segment_idx + 1}/{num_segments} resulted in zero chunks after splitting.")
                continue
            
            for chunk_idx_in_segment, chunk_text in enumerate(chunks):
                if not chunk_text or chunk_text.isspace():
                    print(f"Skipping empty chunk {chunk_idx_in_segment + 1} from segment {segment_idx + 1}.")
                    continue
                
                processed_chunks_count += 1 # Global chunk counter

                try:
                    start_time_embed = time.time()
                    embedding = await asyncio.to_thread(get_openai_embedding, chunk_text)
                    end_time_embed = time.time()

                    if not embedding:
                        msg = f"Skipping chunk (global index {processed_chunks_count}) due to empty embedding."
                        print(f"Warning: {msg}")
                        errors.append({"chunk_text_start": chunk_text[:50], "error": msg, "status_code": 500})
                        continue

                    # Construct metadata for this chunk
                    chunk_metadata = {
                        **request.metadata,          # Base metadata from request
                        **segment_meta,              # Metadata from JSON segment (e.g., source_json_url)
                        "chunk_index_in_segment": chunk_idx_in_segment,
                        "segment_index": segment_idx,
                        "original_url": file_url_for_logging, # URL the file was downloaded from
                        "processed_at": datetime.utcnow(),
                        "file_type": detected_file_type
                    }

                    doc_to_insert = {
                        "text": chunk_text, 
                        "embedding": embedding,
                        "metadata": chunk_metadata,
                        "created_at": datetime.utcnow()
                    }
                    all_processed_chunks_for_dedup.append(doc_to_insert) # Store for batch insert or later dedup logic

                    # Batch insertion could be more efficient here, but for now, one by one
                    start_time_insert = time.time()
                    insert_result = collection.insert_one(doc_to_insert)
                    end_time_insert = time.time()

                    if insert_result.inserted_id:
                        inserted_count += 1
                        if inserted_count % 20 == 0 or inserted_count == 1: # Log progress
                             print(f"Processed and inserted chunk {inserted_count} (global: {processed_chunks_count}). Embed: {end_time_embed-start_time_embed:.2f}s, Insert: {end_time_insert-start_time_insert:.2f}s")
                    else:
                        errors.append({"chunk_text_start": chunk_text[:50], "error": "MongoDB insert_one did not return an inserted_id", "status_code": 500})

                except HTTPException as e:
                    print(f"Error on chunk (global {processed_chunks_count}): HTTP {e.status_code} - {e.detail}")
                    errors.append({"chunk_text_start": chunk_text[:50], "error": e.detail, "status_code": e.status_code})
                    is_critical = e.status_code in [401, 403, 429, 500, 502, 503]
                    if is_critical and first_error is None: first_error = e; break
                except (OperationFailure, ConnectionFailure) as db_error:
                    err_detail = getattr(db_error, 'details', str(db_error))
                    errors.append({"chunk_text_start": chunk_text[:50], "error": f"DB error: {err_detail}", "status_code": 503})
                    if first_error is None: first_error = db_error; break
                except Exception as e:
                    errors.append({"chunk_text_start": chunk_text[:50], "error": f"Unexpected: {str(e)}", "status_code": 500})
                    if first_error is None: first_error = e; break
            
            if first_error: # If a critical error occurred in inner loop, break outer loop
                print(f"Stopping processing due to critical error: {type(first_error).__name__}")
                break 
        
        end_time_chunk_processing_all = time.time()
        print(f"Finished all chunk processing ({end_time_chunk_processing_all - start_time_chunk_processing_all:.2f}s). Total processed: {processed_chunks_count}, Inserted: {inserted_count}, Errors: {len(errors)}")

        # --- 5. Remove Duplicates (based on text content from this specific file processing job) ---
        if inserted_count > 0 and not first_error:
            start_time_dedup = time.time(); print(f"Starting duplicate check for documents from URL: {file_url_for_logging}...")
            duplicates_removed_count = 0
            # This duplicate check is simplified: it only checks for exact text duplicates *from the current processing job*
            # if `metadata.original_url` is used to scope, it works for re-processing the same URL.
            # More robust deduplication might involve checking against all existing text in the collection.
            try:
                pipeline = [
                    {"$match": {"metadata.original_url": file_url_for_logging}}, # Only consider docs from this source URL
                    {"$group": {"_id": "$text", "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
                    {"$match": {"count": {"$gt": 1}}}
                ]
                duplicate_groups = list(collection.aggregate(pipeline))
                ids_to_delete = [oid for group in duplicate_groups for oid in group['ids'][1:]] # Keep first, delete rest

                if ids_to_delete:
                    print(f"Found {len(ids_to_delete)} duplicate document(s) from this source to remove.")
                    delete_result = collection.delete_many({"_id": {"$in": ids_to_delete}})
                    duplicates_removed_count = delete_result.deleted_count
                    print(f"Successfully removed {duplicates_removed_count} duplicate document(s).")
                else: print("No duplicate documents found for this source URL.")
            except (OperationFailure, ConnectionFailure) as db_error: print(f"Database error during duplicate removal: {db_error}")
            except Exception as e: print(f"Unexpected error during duplicate removal: {e}"); traceback.print_exc()
            print(f"Duplicate check finished ({time.time() - start_time_dedup:.2f}s).")
        elif first_error: print("Skipping duplicate removal due to earlier error.")
        else: print("Skipping duplicate removal as no new documents were inserted.")

        # --- 6. Ensure Vector Search Index Exists ---
        # (This part remains largely the same)
        start_time_index_check = time.time()
        should_check_index = (inserted_count > 0) and not isinstance(first_error, (ConnectionFailure, OperationFailure))
        if should_check_index:
            print(f"Checking existence of vector search index '{index_name}'...")
            try:
                existing_indexes = list(collection.list_search_indexes(name=index_name))
                if existing_indexes: index_status = "exists"; print(f"Index '{index_name}' already exists.")
                else:
                    print(f"Index '{index_name}' not found. Attempting to create...")
                    index_definition = {"mappings": {"dynamic": False, "fields": {"embedding": { "type": "knnVector", "dimensions": 1536, "similarity": "cosine" }, "text": { "type": "string", "analyzer": "lucene.standard" }}}}
                    try: collection.create_search_index(model={"name": index_name, "definition": index_definition}); index_status = "created"; print(f"Index '{index_name}' creation initiated.")
                    except OperationFailure as create_err:
                        if getattr(create_err, 'codeName', '') == 'IndexAlreadyExists': index_status = "exists"; print("Index found during creation (race condition).")
                        else: print(f"MongoDB Error creating index: {create_err.details}"); index_status = f"failed_create_op_{getattr(create_err, 'codeName', 'Unknown')}"
                    except Exception as create_err: print(f"Unexpected error creating index: {create_err}"); index_status = "failed_create_unexpected"
            except (OperationFailure, ConnectionFailure) as list_err: print(f"DB error checking index: {list_err}"); index_status = f"failed_list_or_conn_{getattr(list_err, 'codeName', 'Unknown')}"
            except Exception as outer_idx_err: print(f"Unexpected error during index check: {outer_idx_err}"); index_status = "failed_check_unexpected"
            print(f"Index check/creation finished ({time.time() - start_time_index_check:.2f}s). Final Status: {index_status}")
        elif not inserted_count > 0 : index_status = "skipped_no_inserts"
        elif first_error: index_status = "skipped_due_to_db_error"
        else: index_status = "skipped_unknown_reason"


        # --- 7. Return Response ---
        processing_time = round(time.time() - start_time_total, 2)
        final_status_code = 200; response_status = "success"; message = "File processed successfully."

        if errors or first_error:
            if inserted_count > 0: # Partial success
                final_status_code = 207; response_status = "partial_success"
                message = f"Processed {processed_chunks_count} chunks with {len(errors)} errors. {inserted_count} chunks inserted."
            else: # Full failure for chunk processing
                response_status = "failed"
                message = f"Processing failed with {len(errors)} errors. No chunks were inserted."
                # Determine status code based on first_error
                if isinstance(first_error, HTTPException): final_status_code = first_error.status_code
                elif isinstance(first_error, (OperationFailure, ConnectionFailure)): final_status_code = 503
                else: final_status_code = 500 # Generic server error if unhandled
            
            if first_error: # Add critical error detail
                 critical_error_detail = f" Processing stopped early due to critical error: {type(first_error).__name__}."
                 if isinstance(first_error, HTTPException): critical_error_detail += f" (HTTP {first_error.status_code}: {first_error.detail})"
                 elif isinstance(first_error, (OperationFailure, ConnectionFailure)): critical_error_detail += f" (DB Error: {getattr(first_error, 'details', str(first_error))})"
                 else: critical_error_detail += f" (Error: {str(first_error)})"
                 message += critical_error_detail

        if index_status.startswith("failed_"): message += f" Index status: {index_status}."

        response_body = {
            "status": response_status, "message": message,
            "chunks_processed_total": processed_chunks_count, # Renamed for clarity
            "chunks_inserted": inserted_count,
            "duplicates_removed": duplicates_removed_count, 
            "vector_index_name": index_name,
            "vector_index_status": index_status, 
            "processing_time_seconds": processing_time,
        }
        if errors:
            response_body["errors_sample"] = [
                {"chunk_text_start": e.get("chunk_text_start", "N/A"), "status_code": e.get("status_code", 500), "error": str(e.get("error", "Unknown"))}
                for e in errors[:10] # Sample first 10 errors
            ]
        print(f"Responding with status code {final_status_code}. Final index status: {index_status}")
        return JSONResponse(content=response_body, status_code=final_status_code)

    except HTTPException as http_exc:
        print(f"Caught HTTPException: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc # Re-raise
    except Exception as e:
        print(f"Unexpected top-level error in '/process' endpoint: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during file processing: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, user: Dict = Depends(get_user_header)):
    index_name = "vector_index"
    print(f"Hybrid search: query='{request.query[:50]}...', collection='{request.collection_name}', limit={request.limit}, user={user.get('supabase_user_id')}")
    try:
        db = MongoDBManager.get_user_db(user)
        try:
            if request.collection_name not in db.list_collection_names():
                raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found.")
        except OperationFailure as e: raise HTTPException(status_code=503, detail=f"DB error checking collection: {e.details}")
        collection = db[request.collection_name]
        try:
            query_embedding = await asyncio.to_thread(get_openai_embedding, request.query)
            if not query_embedding: raise HTTPException(status_code=400, detail="Failed to generate query embedding.")
        except HTTPException as e: raise e
        except Exception as e: print(f"Query embedding error: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail="Unexpected error generating query embedding.")
        
        num_candidates = max(request.limit * 10, min(request.num_candidates, 1000)); rrf_k = 60
        pipeline = [
            {"$vectorSearch": {"index": index_name, "path": "embedding", "queryVector": query_embedding, "numCandidates": num_candidates, "limit": num_candidates}},
            {"$group": {"_id": None, "docs": {"$push": {"doc": "$$ROOT", "vector_score": {"$meta": "vectorSearchScore"}}}}},
            {"$unwind": {"path": "$docs", "includeArrayIndex": "vr_tmp"}},
            {"$replaceRoot": {"newRoot": {"$mergeObjects": ["$docs.doc", {"vr": {"$add": ["$vr_tmp", 1]}}]}}},
            {"$project": {"_id": 1, "text": 1, "metadata": 1, "vr": 1, "embedding": 0}},
            {"$unionWith": {"coll": request.collection_name, "pipeline": [
                {"$search": {"index": index_name, "text": {"query": request.query, "path": "text"}}},
                {"$limit": num_candidates},
                {"$group": {"_id": None, "docs": {"$push": {"doc": "$$ROOT", "text_score": {"$meta": "searchScore"}}}}},
                {"$unwind": {"path": "$docs", "includeArrayIndex": "tr_tmp"}},
                {"$replaceRoot": {"newRoot": {"$mergeObjects": ["$docs.doc", {"tr": {"$add": ["$tr_tmp", 1]}}]}}},
                {"$project": {"_id": 1, "text": 1, "metadata": 1, "tr": 1, "embedding": 0}}
            ]}},
            {"$group": {"_id": "$_id", "text": {"$first": "$text"}, "metadata": {"$first": "$metadata"}, "vr": {"$min": "$vr"}, "tr": {"$min": "$tr"}}},
            {"$addFields": {"rrf_score": {"$sum": [{"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$vr"]}]}, 0]}, {"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$tr"]}]}, 0]}]}}},
            {"$sort": {"rrf_score": -1}},
            {"$limit": request.limit},
            {"$project": {"_id": 0, "id": {"$toString": "$_id"}, "text": 1, "metadata": 1, "score": "$rrf_score"}}
        ]
        start_time = time.time(); results = list(collection.aggregate(pipeline)); end_time = time.time()
        print(f"Hybrid search in {end_time - start_time:.2f}s. Found {len(results)} results.")
        return SearchResponse(results=[SearchResultItem(**res) for res in results])
    except OperationFailure as e:
        detail = f"DB error during search: {e.details}"; status_code = 500
        if "index not found" in str(e.details).lower() or getattr(e, 'codeName', '') == 'IndexNotFound': status_code = 404; detail = f"Search index '{index_name}' not found in '{request.collection_name}'."
        elif e.code == 13: status_code = 403; detail = "Auth failed for search."
        elif 'vectorSearch' in str(e.details) and 'queryVector' in str(e.details): status_code = 400; detail = f"Invalid query vector. {e.details}"
        raise HTTPException(status_code=status_code, detail=detail)
    except HTTPException as e: raise e
    except Exception as e: print(f"Unexpected search error: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected error during search: {e}")

@app.post("/vector-search", response_model=VectorSearchResponse)
async def vector_search_documents(request: VectorSearchRequest, user: Dict = Depends(get_user_header)):
    index_name = "vector_index"; all_results = []; warnings = []
    print(f"Multi-collection vector search: query='{request.query[:50]}...', collections={request.collection_names}, filter={request.filter}, user={user.get('supabase_user_id')}")
    try:
        db = MongoDBManager.get_user_db(user); db_name = db.name
        try:
            query_embedding = await asyncio.to_thread(get_openai_embedding, request.query)
            if not query_embedding: raise HTTPException(status_code=400, detail="Failed to generate query embedding.")
        except HTTPException as e: raise e
        except Exception as e: print(f"Query embedding error: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail="Failed to generate query embedding.")
        
        vector_search_stage_base = {"$vectorSearch": {"index": index_name, "path": "embedding", "queryVector": query_embedding, "numCandidates": request.num_candidates, "limit": request.limit}}
        if request.filter: vector_search_stage_base["$vectorSearch"]["filter"] = request.filter
        project_stage = {"$project": {"_id": 0, "id": {"$toString": "$_id"}, "text": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}}
        
        available_collections = set(db.list_collection_names())
        for collection_name in request.collection_names:
            if collection_name not in available_collections: warnings.append(f"Collection '{collection_name}' not found in '{db_name}'."); continue
            print(f"Searching collection: '{collection_name}'...")
            collection = db[collection_name]; pipeline = [vector_search_stage_base, project_stage]
            try:
                results = list(collection.aggregate(pipeline))
                all_results.extend(results)
            except OperationFailure as e:
                warning_msg = f"Search failed for '{collection_name}': "
                if "index not found" in str(e.details).lower() or getattr(e, 'codeName', '') == 'IndexNotFound': warning_msg += f"Index '{index_name}' not found."
                else: warning_msg += f"OperationFailure ({e.details})."
                warnings.append(warning_msg); print(f"DB error in '{collection_name}': {e.details}")
            except ConnectionFailure as e: print(f"DB connection failure for '{collection_name}': {e}"); raise HTTPException(status_code=503, detail=f"DB connection lost for '{collection_name}'.")
            except Exception as e: warnings.append(f"Unexpected error in '{collection_name}': {str(e)}"); print(f"Unexpected error for '{collection_name}': {e}"); traceback.print_exc()
        
        if not all_results: return VectorSearchResponse(results=[], warnings=warnings if warnings else None)
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return VectorSearchResponse(results=[SearchResultItem(**res) for res in all_results[:request.limit]], warnings=warnings if warnings else None)
    except HTTPException as e: raise e
    except (OperationFailure, ConnectionFailure) as e: print(f"DB error in vector search setup: {e}"); raise HTTPException(status_code=503 if isinstance(e, ConnectionFailure) else 500, detail=f"DB error in search setup: {getattr(e, 'details', str(e))}")
    except Exception as e: print(f"Unexpected multi-collection vector search error: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected error during vector search: {e}")

@app.post("/user-collections", response_model=UserInfoResponse)
def get_user_collections_header_auth(request: UserCollectionsRequest, user: Dict = Depends(get_user_header)):
    auth_user_id = user.get("supabase_user_id"); req_user_id = request.supabase_user_id
    if auth_user_id != req_user_id: raise HTTPException(status_code=403, detail="Authenticated user does not match requested Supabase User ID.")
    if mongo_client is None: raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        db_name = user.get("db_name")
        if not db_name: raise HTTPException(status_code=500, detail="Internal error: User data incomplete.")
        try:
            user_db = mongo_client[db_name]
            return UserInfoResponse(db_name=db_name, collections=user_db.list_collection_names())
        except (OperationFailure, ConnectionFailure) as e: raise HTTPException(status_code=503, detail=f"DB access error: {getattr(e, 'details', str(e))}")
        except Exception as e: print(f"Unexpected error accessing user DB '{db_name}': {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected error retrieving collections: {e}")
    except HTTPException as e: raise e
    except Exception as e: print(f"Unexpected error in /user-collections for {auth_user_id}: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.delete("/collections/{collection_name}", response_model=ActionResponse)
async def delete_collection_endpoint(collection_name: str = Path(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$"), user: Dict = Depends(get_user_header)):
    print(f"Request to delete collection '{collection_name}' for user {user.get('supabase_user_id')}")
    try:
        db = MongoDBManager.get_user_db(user); db_name = db.name
        if collection_name not in db.list_collection_names(): raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found in '{db_name}'.")
        db.drop_collection(collection_name)
        return ActionResponse(status="success", message=f"Collection '{collection_name}' deleted.")
    except OperationFailure as e:
        detail = f"DB operation failed: {e.details}"; status_code = 500
        if e.code == 13: status_code = 403; detail = "Auth failed for delete."
        raise HTTPException(status_code=status_code, detail=detail)
    except ConnectionFailure as e: print(f"DB connection failure for delete: {e}"); raise HTTPException(status_code=503, detail="DB connection lost.")
    except HTTPException as e: raise e
    except Exception as e: print(f"Unexpected error deleting '{collection_name}': {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected error during deletion: {e}")

@app.put("/collections/{current_name}", response_model=ActionResponse)
async def rename_collection_endpoint(current_name: str = Path(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$"), request: RenameCollectionBody = Body(...), user: Dict = Depends(get_user_header)):
    new_name = request.new_name
    print(f"Request to rename '{current_name}' to '{new_name}' for user {user.get('supabase_user_id')}")
    if current_name == new_name: raise HTTPException(status_code=400, detail="New name cannot be same as current.")
    try:
        db = MongoDBManager.get_user_db(user); db_name = db.name
        collection_names = db.list_collection_names()
        if current_name not in collection_names: raise HTTPException(status_code=404, detail=f"Source collection '{current_name}' not found.")
        if new_name in collection_names: raise HTTPException(status_code=409, detail=f"Target collection '{new_name}' already exists.")
        try:
            if mongo_client is None: raise HTTPException(status_code=503, detail="DB client unavailable for rename.")
            mongo_client.admin.command('renameCollection', f'{db_name}.{current_name}', to=f'{db_name}.{new_name}')
            warning_detail = "Important: Atlas Search indexes must be manually recreated for the new collection name."
            return ActionResponse(status="success", message=f"Collection '{current_name}' renamed to '{new_name}'.", details=warning_detail)
        except OperationFailure as e:
            detail = f"DB rename failed: {e.details}"; status_code = 500
            if e.code == 13: status_code = 403; detail = "Auth error for rename."
            elif e.code == 10026: status_code = 409; detail = f"Target '{new_name}' already exists."
            elif e.code == 26: status_code = 404; detail = f"Source '{current_name}' not found."
            raise HTTPException(status_code=status_code, detail=detail)
    except ConnectionFailure as e: print(f"DB connection failure for rename: {e}"); raise HTTPException(status_code=503, detail="DB connection lost.")
    except HTTPException as e: raise e
    except Exception as e: print(f"Unexpected error renaming '{current_name}': {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected error during rename: {e}")

@app.get("/collections/{collection_name}/stats", response_model=CollectionStatsResponse)
def get_collection_stats(collection_name: str = Path(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$"), user: Dict = Depends(get_user_header)):
    print(f"Request stats for '{collection_name}' for user {user.get('supabase_user_id')}")
    distinct_urls_count = None; stats_message = None
    try:
        db = MongoDBManager.get_user_db(user); db_name = db.name
        try:
            if collection_name not in db.list_collection_names(): raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found in '{db_name}'.")
        except (OperationFailure, ConnectionFailure) as e: raise HTTPException(status_code=503, detail=f"DB error checking collection: {getattr(e, 'details', str(e))}")
        
        collection = db[collection_name]
        try:
            coll_stats_raw = db.command('collStats', collection_name)
            coll_stats = CollectionStats(**coll_stats_raw)
        except OperationFailure as e:
            detail = f"DB command 'collStats' failed: {e.details}"; status_code = 500
            if e.code == 13: status_code = 403; detail = "Auth failed for collStats."
            raise HTTPException(status_code=status_code, detail=detail)
        except ConnectionFailure as e: raise HTTPException(status_code=503, detail="DB connection lost during stats.")
        except Exception as e: print(f"Error processing collStats: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Failed to process stats: {e}")
        
        try:
            if collection.find_one({"metadata.original_url": {"$exists": True}}, {"_id": 1}):
                distinct_urls_count = len(collection.distinct("metadata.original_url"))
            else: distinct_urls_count = 0; stats_message = "No 'metadata.original_url' found."
        except (OperationFailure, ConnectionFailure) as e: stats_message = f"Could not count distinct URLs (DB error): {getattr(e, 'details', str(e))}"
        except Exception as e: stats_message = f"Could not count distinct URLs (unexpected error): {e}"
        
        return CollectionStatsResponse(collection_name=collection_name, stats=coll_stats, distinct_source_urls_count=distinct_urls_count, message=stats_message)
    except HTTPException as e: raise e
    except Exception as e: print(f"Unexpected error in get_collection_stats for '{collection_name}': {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected error getting stats: {e}")

# --- Application startup ---
if __name__ == "__main__":
    startup_errors = []
    if mongo_client is None: startup_errors.append("MongoDB Client is None")
    if auth_db is None: startup_errors.append("MongoDB Auth DB is None")
    if openai_client is None: startup_errors.append("OpenAI Client is None")
    if startup_errors:
         print("FATAL: Required clients not initialized. Server cannot start.")
         for error in startup_errors: print(f" - {error}")
         sys.exit("Server cannot start due to initialization failures.")
    else:
        print("All required clients initialized.")
        print("Starting FastAPI server on host 0.0.0.0, port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)