# -*- coding: utf-8 -*-
import os
import io
import secrets
from datetime import datetime
from typing import List, Optional, Dict, Any
import traceback # エラー詳細表示のため
import re # rename エンドポイントの検証用に追加
import sys # SystemExit用に追加

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
        serverSelectionTimeoutMS=5000, # 5 seconds timeout for server selection
        connectTimeoutMS=3000 # 3 seconds timeout for initial connection
    )
    # The ismaster command is cheap and does not require auth.
    mongo_client.admin.command('ismaster')
    print("Successfully connected to MongoDB.")
    # Get the authentication database
    auth_db = mongo_client["auth_db"]
except ConnectionFailure as e:
    print(f"FATAL: Failed to connect to MongoDB (ConnectionFailure): {e}")
    raise SystemExit(f"MongoDB connection failed: {e}")
except OperationFailure as e:
    # This might happen if 'ismaster' fails due to auth issues, though it usually doesn't require auth.
    # More likely during other operations if connection seems initially successful but isn't fully functional.
    print(f"FATAL: Failed to connect to MongoDB (OperationFailure): {e.details}")
    raise SystemExit(f"MongoDB operation failure during connection test: {e}")
except Exception as e:
    # Catch any other unexpected errors during connection.
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
    name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$") # Restrict collection names

class ProcessRequest(BaseModel):
    pdf_url: HttpUrl
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$") # Restrict collection names
    metadata: Dict[str, Any] = Field(default_factory=dict) # Allow Any type in metadata values

class UserCollectionsRequest(BaseModel):
    supabase_user_id: str = Field(..., min_length=1, description="The Supabase User ID")

class UserInfoResponse(BaseModel):
    db_name: str
    collections: List[str]

# コレクション名変更用リクエストモデル (Body用、APIキーなし)
class RenameCollectionBody(BaseModel):
    new_name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="New name for the collection")

# 汎用レスポンスモデル
class ActionResponse(BaseModel):
    status: str
    message: str
    details: Optional[str] = None

# Search models
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

# --- Dependencies ---
def verify_admin(api_key: str = Header(..., alias="X-API-Key")):
    """Verifies the admin API key provided in the header."""
    if not ADMIN_API_KEY:
         print("CRITICAL: ADMIN_API_KEY is not set in the environment.")
         raise HTTPException(status_code=500, detail="Server configuration error: Admin key missing.")
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required. Invalid API Key provided.")

# HeaderからAPIキーを取得して認証する依存関係
def get_user_header(api_key: str = Header(..., alias="X-API-Key")):
    """
    Retrieves the user based on the API key provided in the X-API-Key header.
    Used for endpoints requiring user authentication via header.
    """
    if auth_db is None:
         # This should ideally not happen if startup checks pass, but safeguard anyway.
         print("Error in get_user_header: auth_db is not available.")
         raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        if not api_key:
             print("Error in get_user_header: X-API-Key header is missing.")
             # Use 401 Unauthorized for missing credentials
             raise HTTPException(status_code=401, detail="X-API-Key header is required.")

        # Find user by API key
        user = auth_db.users.find_one({"api_key": api_key})
        if not user:
            # Use 403 Forbidden for invalid credentials
            raise HTTPException(status_code=403, detail="Invalid API Key provided in X-API-Key header.")
        # Convert ObjectId to string if necessary for later use, though often not needed for the user object itself.
        # user['_id'] = str(user['_id'])
        return user
    except OperationFailure as e:
        print(f"Database error finding user in auth_db (get_user_header): {e.details}")
        raise HTTPException(status_code=503, detail="Database operation failed while validating API key header.")
    except HTTPException as e:
         raise e # Re-raise already crafted HTTPExceptions
    except Exception as e:
        # Catch any other unexpected errors during user lookup
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
        # Generate a unique database name and API key
        db_name = f"userdb_{supabase_user_id[:8]}_{secrets.token_hex(4)}"
        api_key = secrets.token_urlsafe(32)
        try:
            # Insert user record into the central authentication database
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
            # Catch any other unexpected errors during insertion
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
            # This indicates inconsistent data in the auth_db.users collection
            print(f"Error in get_user_db: User object is missing 'db_name'. User Supabase ID: {user.get('supabase_user_id', 'N/A')}")
            raise HTTPException(status_code=500, detail="Internal server error: User data is inconsistent (missing database name).")
        try:
            # Return the specific user's database object
            return mongo_client[db_name]
        except Exception as e:
            # Catch potential errors accessing the database (though less common than accessing collections)
            print(f"Unexpected error accessing user database '{db_name}': {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=503, detail=f"Failed to access user database '{db_name}': {e}")

    @staticmethod
    def create_collection(db, name: str, user_id: str):
        # Note: MongoDB creates collections implicitly on first write.
        # This function mainly validates access and logs intention.
        if db is None:
             # Should be caught by caller, but double-check
             print(f"Error in create_collection: Database object is None for user {user_id}.")
             raise HTTPException(status_code=500, detail="Internal Server Error: Invalid database reference passed.")
        try:
            # Check if collection exists (optional, but good for logging/confirmation)
            collection_names = db.list_collection_names()
            if name not in collection_names:
                 # Log intention to create implicitly
                 print(f"Collection '{name}' does not exist in database '{db.name}'. It will be created implicitly on first write operation.")
                 # Return the collection object; actual creation happens on insert/index creation etc.
                 return db[name]
            else:
                 # Collection already exists
                 print(f"Collection '{name}' already exists in database '{db.name}'.")
                 return db[name]
        except OperationFailure as e:
            # Handle potential errors listing collections (e.g., permissions)
            print(f"MongoDB Operation Failure accessing/listing collection '{name}' in database '{db.name}': {e.details}")
            raise HTTPException(status_code=500, detail=f"Database operation failed while checking for collection '{name}': {e.details}")
        except ConnectionFailure as e:
            # Handle connection errors during the operation
            print(f"MongoDB Connection Failure during collection check for '{name}' in database '{db.name}': {e}")
            raise HTTPException(status_code=503, detail=f"Database connection lost while checking for collection '{name}'.")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"Error accessing or preparing collection '{name}' in database '{db.name}': {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to ensure collection '{name}' exists due to an unexpected error: {e}")

# --- Text splitting helper ---
def split_text_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 100) -> List[str]:
    """Splits text into chunks by words, respecting chunk_size and overlap."""
    if not text: return []
    words = text.split() # Split by whitespace
    if not words: return []

    chunks = []
    current_pos = 0
    while current_pos < len(words):
        # Calculate end position for the current chunk
        end_pos = current_pos
        current_length = 0
        last_valid_end_pos = current_pos # Fallback if a single word is too long

        while end_pos < len(words):
            word_len = len(words[end_pos])
            # Add 1 for space, except for the first word in the chunk
            length_to_add = word_len + (1 if end_pos > current_pos else 0)

            if current_length + length_to_add <= chunk_size:
                current_length += length_to_add
                last_valid_end_pos = end_pos + 1 # Point after the last included word
                end_pos += 1
            else:
                # Word would exceed chunk size, stop including more words
                break

        # Handle cases where a single word exceeds chunk size
        if last_valid_end_pos == current_pos:
            # If it's the very first word and it's too long, take it anyway (or truncate/error)
            if current_pos == 0 and len(words[0]) > chunk_size:
                 print(f"Warning: A single word is longer than the chunk size ({chunk_size}). Taking the full word: '{words[0][:50]}...'")
                 chunks.append(words[0])
                 current_pos += 1 # Move past this long word
                 continue # Start next chunk from the next word
            # If not the first word, or if the first word was handled, take at least one word
            elif current_pos < len(words):
                # This can happen if a word right after a previous chunk is itself larger than chunk_size.
                # To avoid infinite loop, force progress by taking at least this one word.
                print(f"Warning: Word at index {current_pos} ('{words[current_pos][:50]}...') might exceed chunk size but forcing inclusion to progress.")
                last_valid_end_pos = current_pos + 1
            else: # Reached end of words
                 break


        # Extract the chunk words and join them
        chunk_words = words[current_pos:last_valid_end_pos]
        chunks.append(" ".join(chunk_words))

        # Calculate the starting position for the next chunk, considering overlap
        if last_valid_end_pos >= len(words): # Reached the end
             break

        # Find the start index for overlap based on character count (approximate)
        overlap_start_index = last_valid_end_pos - 1 # Start from the last word of the current chunk
        overlap_char_count = 0
        while overlap_start_index > current_pos:
             # Add length of the word and 1 for the space before it (approx)
             overlap_char_count += len(words[overlap_start_index]) + 1
             if overlap_char_count >= overlap:
                 # Found enough overlap
                 break
             overlap_start_index -= 1

        # Ensure overlap doesn't go back beyond the start of the current chunk
        overlap_start_index = max(current_pos, overlap_start_index)

        # Determine next chunk's starting position
        # If overlap caused us to step back, start there. Otherwise, start after the current chunk.
        if overlap_start_index > current_pos:
             current_pos = overlap_start_index
        else:
             # No effective overlap achieved (e.g., chunk was short, or overlap small)
             # or overlap_start_index ended up back at current_pos
             current_pos = last_valid_end_pos # Start next chunk right after the current one

    return chunks

# --- OpenAI Embedding Function ---
# (Make sure OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL are set)
def get_openai_embedding(text: str) -> List[float]:
    """Generates embedding for the given text using OpenAI API."""
    # Handle empty or whitespace-only input
    if not text or text.isspace():
        print("Warning: Attempted to get embedding for empty or whitespace-only text.")
        return [] # Return empty list for empty input

    try:
        # Replace newlines and multiple spaces, trim whitespace
        cleaned_text = ' '.join(text.split()).strip()
        if not cleaned_text:
             print("Warning: Text became empty after cleaning whitespace.")
             return []

        # Call the OpenAI API
        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=cleaned_text
        )

        # Extract the embedding
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            # This case should be rare if the API call succeeds
            print(f"Warning: OpenAI API returned success but no embedding data for text: {cleaned_text[:100]}...")
            # Raise an exception as this indicates an unexpected API response format or issue
            raise HTTPException(status_code=500, detail="OpenAI API returned no embedding data unexpectedly.")

    # --- Specific OpenAI Error Handling ---
    except openai.APIConnectionError as e:
        # Handle connection errors (e.g., network issues)
        print(f"OpenAI API Connection Error: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to OpenAI API: {e}")
    except openai.RateLimitError as e:
        # Handle rate limit errors (429)
        print(f"OpenAI API Rate Limit Error: {e}")
        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. Please try again later.")
    except openai.AuthenticationError as e:
        # Handle authentication errors (401) - Invalid API key
        print(f"OpenAI API Authentication Error: {e}")
        raise HTTPException(status_code=401, detail="OpenAI API authentication failed. Check your API key.")
    except openai.PermissionDeniedError as e:
         # Handle permission errors (403) - Key might be valid but lacks permissions
         print(f"OpenAI API Permission Denied Error: {e}")
         raise HTTPException(status_code=403, detail="OpenAI API permission denied. Check your API key's permissions.")
    except openai.NotFoundError as e:
         # Handle not found errors (404) - Often means invalid model name
         print(f"OpenAI API Not Found Error (check model name?): {e}")
         raise HTTPException(status_code=404, detail=f"OpenAI resource not found. Check model name '{OPENAI_EMBEDDING_MODEL}'. Error: {e}")
    except openai.BadRequestError as e:
         # Handle bad request errors (400) - Often input validation issues (e.g., too long)
         print(f"OpenAI API Bad Request Error: {e}")
         raise HTTPException(status_code=400, detail=f"OpenAI API bad request. Check input data. Error: {e}")
    except openai.APIStatusError as e:
        # Handle other generic API errors based on status code
        print(f"OpenAI API Status Error: Status Code={e.status_code}, Response={e.response}")
        status_code = e.status_code
        detail = f"OpenAI API Error (Status {status_code}): {e.message or str(e.response)}"
        # Remap common errors to specific FastAPI status codes
        if status_code == 400: detail = f"OpenAI Bad Request: {e.message}"; raise HTTPException(status_code=400, detail=detail)
        elif status_code == 401: detail = "OpenAI Authentication Error. Check API Key."; raise HTTPException(status_code=401, detail=detail)
        elif status_code == 403: detail = "OpenAI Permission Denied. Check Key Permissions."; raise HTTPException(status_code=403, detail=detail)
        elif status_code == 404: detail = f"OpenAI Resource Not Found (Model?). Error: {e.message}"; raise HTTPException(status_code=404, detail=detail)
        elif status_code == 429: detail = "OpenAI Rate Limit Exceeded."; raise HTTPException(status_code=429, detail=detail)
        elif status_code >= 500: detail = f"OpenAI Server Error ({status_code}). {e.message}"; raise HTTPException(status_code=502, detail=detail) # 502 Bad Gateway for upstream errors
        else: raise HTTPException(status_code=status_code, detail=detail) # Use original status code if not specifically handled
    # --- General Error Handling ---
    except Exception as e:
        # Catch any other unexpected errors during embedding generation
        print(f"Unexpected error during OpenAI embedding generation: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating text embedding: {e}")

# --- API endpoints ---
@app.get("/health")
def health_check():
    """Provides basic health status of the service."""
    db_status = "disconnected"
    openai_status = "not_initialized"

    # Check MongoDB connection
    if mongo_client and auth_db:
        try:
            # Ping the MongoDB server to confirm connectivity
            mongo_client.admin.command('ping')
            db_status = "connected"
        except (ConnectionFailure, OperationFailure) as e:
            print(f"Health check: MongoDB ping failed: {e}")
            db_status = f"error ({type(e).__name__})"
        except Exception as e:
             print(f"Health check: Unexpected error during MongoDB ping: {e}")
             db_status = "error (unexpected)"


    # Check OpenAI client initialization
    if openai_client:
        openai_status = "initialized"
        # Optional: Add a simple test call to OpenAI API if needed, but be mindful of cost/rate limits
        # try:
        #     openai_client.models.list() # Example lightweight call
        #     openai_status = "initialized_and_responsive"
        # except Exception as e:
        #     print(f"Health check: OpenAI API test call failed: {e}")
        #     openai_status = "initialized_but_unresponsive"


    all_ok = db_status == "connected" and openai_status.startswith("initialized")

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ok" if all_ok else "error",
            "details": {
                 "database": db_status,
                 "openai_client": openai_status
            }
        }
     )

@app.get("/auth-db", dependencies=[Depends(verify_admin)])
def get_auth_db_contents():
    """(Admin Only) Retrieves user registration information (excluding API keys)."""
    if auth_db is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        # Fetch users, excluding sensitive _id and api_key fields
        users = list(auth_db.users.find({}, {"_id": 0, "api_key": 0}))
        return {"users": users}
    except OperationFailure as e:
        print(f"Admin Error: Database error reading auth_db: {e.details}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user data due to database operation failure: {e.details}")
    except Exception as e:
        print(f"Admin Error: Unexpected error reading auth_db: {type(e).__name__}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving user data.")

@app.post("/register", response_model=RegisterResponse)
def register_user(request: UserRegister, response: Response):
    """Registers a new user or retrieves existing user's API key and DB name."""
    if auth_db is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        # Check if user already exists based on Supabase User ID
        existing_user = auth_db.users.find_one({"supabase_user_id": request.supabase_user_id})

        if existing_user:
            # User exists, return existing details
            api_key = existing_user.get("api_key")
            db_name = existing_user.get("db_name")
            # Check for data consistency
            if not api_key or not db_name:
                 print(f"Error: Inconsistent data for existing user {request.supabase_user_id}. Missing key or db_name.")
                 raise HTTPException(status_code=500, detail="Internal server error: Inconsistent user data found.")

            response.status_code = 200 # OK
            return RegisterResponse(
                api_key=api_key,
                db_name=db_name,
                database_exist=True # Indicate that the user already existed
            )
        else:
            # User does not exist, create new user DB and record
            print(f"Registering new user: {request.supabase_user_id}")
            db_info = MongoDBManager.create_user_db(request.supabase_user_id)
            response.status_code = 201 # Created
            return RegisterResponse(
                api_key=db_info["api_key"],
                db_name=db_info["db_name"],
                database_exist=False # Indicate that a new user/DB was created
            )

    except HTTPException as e:
        # Re-raise HTTPExceptions raised by called functions (like create_user_db)
        raise e
    except OperationFailure as e:
        # Handle potential DB errors during find_one
        print(f"Database error during user registration check for {request.supabase_user_id}: {e.details}")
        raise HTTPException(status_code=500, detail=f"Database operation failed during registration check: {e.details}")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error during user registration for {request.supabase_user_id}: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during user registration: {e}")

# POST /collections uses Header auth
@app.post("/collections", status_code=201, response_model=ActionResponse)
def create_collection_endpoint(request: CollectionCreate, user: Dict = Depends(get_user_header)):
    """Creates a new collection (or ensures it exists) within the user's database. Requires X-API-Key header."""
    try:
        db = MongoDBManager.get_user_db(user)
        # Use the manager method which handles checking existence and returning the collection object
        collection = MongoDBManager.create_collection(db, request.name, user["supabase_user_id"])
        # MongoDB collections are created lazily, so this confirms the setup is ready.
        # Actual creation happens on first insert or index creation.
        return ActionResponse(
            status="success",
            message=f"Collection '{collection.name}' is ready for use (will be created on first write if new)."
        )
    except HTTPException as e:
        # Re-raise exceptions from dependencies or manager methods
        raise e
    except Exception as e:
        # Catch unexpected errors during the process
        print(f"Error creating/ensuring collection '{request.name}' for user {user.get('supabase_user_id', 'N/A')}: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to prepare collection '{request.name}': {e}")


# ★★★ Modified /process endpoint ★★★
@app.post("/process")
async def process_pdf(request: ProcessRequest, user: Dict = Depends(get_user_header)):
    """
    Downloads, processes PDF, stores embeddings, and ensures vector index exists.
    Requires X-API-Key header.
    Index management: Creates index 'vector_index' if it doesn't exist. Does not drop/recreate existing index.
    """
    index_name = "vector_index"
    index_status = "not_checked" # Initial status
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
            response = await asyncio.to_thread(requests.get, str(request.pdf_url), timeout=60)
            response.raise_for_status()
            end_time_download = time.time(); print(f"PDF downloaded ({end_time_download - start_time_download:.2f}s).")
            content_length = response.headers.get('Content-Length'); pdf_bytes = response.content; pdf_size = len(pdf_bytes)
            if pdf_size > MAX_FILE_SIZE: raise HTTPException(status_code=413, detail=f"PDF size ({pdf_size/(1024*1024):.2f}MB) > limit ({MAX_FILE_SIZE/(1024*1024)}MB).")
            pdf_content = io.BytesIO(pdf_bytes); size_source = f"header: {content_length}" if content_length else "checked post-download"
            print(f"PDF size: {pdf_size/(1024*1024):.2f} MB ({size_source}).")
        except requests.exceptions.Timeout: raise HTTPException(status_code=408, detail="PDF download timeout.")
        except requests.exceptions.SSLError as ssl_err: raise HTTPException(status_code=502, detail=f"SSL error during PDF download: {ssl_err}.")
        except requests.exceptions.RequestException as req_error:
             status_code = 502 if isinstance(req_error, (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError)) else 400
             detail = f"PDF download failed: {req_error}" + (f" (Status: {req_error.response.status_code})" if hasattr(req_error, 'response') and req_error.response else "")
             raise HTTPException(status_code=status_code, detail=detail)

        # --- 2. Extract text ---
        try:
            start_time_extract = time.time()
            # Use asyncio.to_thread for blocking PyPDF2 call
            def extract_text_sync(pdf_stream):
                 try:
                     reader = PyPDF2.PdfReader(pdf_stream)
                     if reader.is_encrypted:
                         raise ValueError("Processing encrypted PDFs is not supported.")
                     full_text = ""
                     for page in reader.pages:
                         page_text = page.extract_text()
                         if page_text: full_text += page_text + "\n"
                     return full_text
                 except PyPDF2.errors.PdfReadError as pdf_err:
                     raise ValueError(f"Invalid or corrupted PDF file: {pdf_err}") from pdf_err
                 except Exception as inner_e:
                     raise RuntimeError(f"Error during text extraction phase: {inner_e}") from inner_e

            text = await asyncio.to_thread(extract_text_sync, pdf_content)
            end_time_extract = time.time()

            if not text or text.isspace():
                print("No text could be extracted from the PDF or the PDF was empty.")
                return JSONResponse(
                    content={
                        "status": "success", "message": "PDF processed, but no text was extracted.",
                        "chunks_processed": 0, "chunks_inserted": 0, "duplicates_removed": 0,
                        "vector_index_name": index_name, "vector_index_status": "skipped_no_text",
                        "processing_time_seconds": round(time.time() - start_time_total, 2)
                    }, status_code=200
                )
            print(f"Text extracted successfully ({len(text)} characters, {end_time_extract - start_time_extract:.2f}s).")

        except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e: print(f"Text extraction runtime error: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Failed during text extraction phase: {e}")
        except Exception as e: print(f"Unexpected error during text extraction: {type(e).__name__}"); traceback.print_exc(); raise HTTPException(status_code=500, detail=f"An unexpected error occurred during text extraction: {e}")

        # --- 3. Split text ---
        start_time_split = time.time(); chunks = split_text_into_chunks(text); end_time_split = time.time()
        print(f"Text split into {len(chunks)} chunks ({end_time_split - start_time_split:.2f}s).")
        if not chunks:
            print("Text extracted but resulted in zero chunks after splitting.")
            return JSONResponse(
                content={
                    "status": "success", "message": "PDF processed, but no processable chunks generated.",
                    "chunks_processed": 0, "chunks_inserted": 0, "duplicates_removed": 0,
                    "vector_index_name": index_name, "vector_index_status": "skipped_no_chunks",
                    "processing_time_seconds": round(time.time() - start_time_total, 2)
                }, status_code=200
            )

        # --- 4. DB Setup ---
        db = MongoDBManager.get_user_db(user); collection = MongoDBManager.create_collection(db, request.collection_name, user["supabase_user_id"])
        print(f"Using database '{db.name}' and collection '{collection.name}'.")

        # --- 5. Process Chunks (Embedding and Insertion) ---
        start_time_chunks = time.time(); print(f"Starting processing for {len(chunks)} text chunks...")
        for i, chunk in enumerate(chunks):
            if not chunk or chunk.isspace():
                print(f"Skipping empty chunk at index {i}.")
                continue
            processed_chunks_count += 1

            try:
                start_time_embed = time.time()
                embedding = await asyncio.to_thread(get_openai_embedding, chunk)
                end_time_embed = time.time()

                if not embedding:
                    print(f"Warning: Skipping chunk {i+1}/{len(chunks)} due to empty embedding result.")
                    errors.append({"chunk_index": i, "error": "Received empty embedding without error", "status_code": 500})
                    continue

                doc = {
                    "text": chunk, "embedding": embedding,
                    "metadata": {**request.metadata, "chunk_index": i, "original_url": str(request.pdf_url), "processed_at": datetime.utcnow()},
                    "created_at": datetime.utcnow()
                }

                start_time_insert = time.time()
                insert_result = collection.insert_one(doc)
                end_time_insert = time.time()

                if insert_result.inserted_id:
                    inserted_count += 1
                    if (i + 1) % 50 == 0 or (i + 1) == len(chunks): # Log progress less frequently
                         print(f"Processed chunk {i+1}/{len(chunks)} (Insert ID: {insert_result.inserted_id}, Embed: {end_time_embed-start_time_embed:.2f}s, Insert: {end_time_insert-start_time_insert:.2f}s)")
                else:
                    print(f"Warning: Chunk {i+1}/{len(chunks)} insertion did not return an ID.")
                    errors.append({"chunk_index": i, "error": "MongoDB insert_one did not return an inserted_id", "status_code": 500})

            except HTTPException as e:
                print(f"Error processing chunk {i+1}/{len(chunks)}: HTTP {e.status_code} - {e.detail}")
                errors.append({"chunk_index": i, "error": e.detail, "status_code": e.status_code})
                is_critical = e.status_code in [401, 403, 429, 500, 502, 503]
                if is_critical and first_error is None:
                    print(f"Stopping processing due to critical error: {e.status_code}")
                    first_error = e; break

            except (OperationFailure, ConnectionFailure) as db_error:
                error_detail = getattr(db_error, 'details', str(db_error))
                print(f"Database error processing chunk {i+1}/{len(chunks)}: {type(db_error).__name__} - {error_detail}")
                errors.append({"chunk_index": i, "error": f"Database operation/connection failed: {error_detail}", "status_code": 503})
                if first_error is None:
                    print("Stopping processing due to critical database error.")
                    first_error = db_error; break

            except Exception as e:
                print(f"Unexpected error processing chunk {i+1}/{len(chunks)}: {type(e).__name__} - {e}")
                traceback.print_exc()
                errors.append({"chunk_index": i, "error": f"Unexpected error: {str(e)}", "status_code": 500})
                if first_error is None:
                    print("Stopping processing due to unexpected critical error.")
                    first_error = e; break

        end_time_chunks = time.time(); print(f"Finished chunk processing ({end_time_chunks - start_time_chunks:.2f}s). Attempted: {processed_chunks_count}, Inserted: {inserted_count}, Errors: {len(errors)}")

        # --- 6. Remove Duplicates ---
        if inserted_count > 0 and not first_error:
            start_time_dedup = time.time(); print(f"Starting duplicate check for documents from URL: {request.pdf_url}...")
            duplicates_removed_count = 0
            try:
                pipeline = [
                    {"$match": {"metadata.original_url": str(request.pdf_url)}},
                    {"$group": {"_id": "$text", "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
                    {"$match": {"count": {"$gt": 1}}}
                ]
                duplicate_groups = list(collection.aggregate(pipeline))
                ids_to_delete = [oid for group in duplicate_groups for oid in group['ids'][1:]]

                if ids_to_delete:
                    print(f"Found {len(ids_to_delete)} duplicate document(s) to remove.")
                    delete_result = collection.delete_many({"_id": {"$in": ids_to_delete}})
                    duplicates_removed_count = delete_result.deleted_count
                    print(f"Successfully removed {duplicates_removed_count} duplicate document(s).")
                else: print("No duplicate documents found for this URL.")
                print(f"Duplicate check and removal finished ({time.time() - start_time_dedup:.2f}s).")
            except (OperationFailure, ConnectionFailure) as db_error: print(f"Database error during duplicate removal: {db_error}") # Log error, continue
            except Exception as e: print(f"Unexpected error during duplicate removal: {e}"); traceback.print_exc() # Log error, continue
        elif first_error: print("Skipping duplicate removal because processing stopped due to an error.")
        else: print("Skipping duplicate removal as no new documents were inserted.")


        # --- ★ 7. Ensure Vector Search Index Exists ---
        start_time_index_check = time.time()
        index_exists = False
        # Only check/create index if chunks were processed and no critical DB error occurred earlier
        should_check_index = (processed_chunks_count > 0) and not isinstance(first_error, (ConnectionFailure, OperationFailure))

        if should_check_index:
            print(f"Checking existence of vector search index '{index_name}'...")
            try:
                # Check if the index already exists (try filtering directly by name)
                existing_indexes = list(collection.list_search_indexes(name=index_name))

                if existing_indexes:
                    index_exists = True
                    index_status = "exists"
                    print(f"Index '{index_name}' already exists. No action needed.")
                else:
                    # Index does not exist, attempt to create it
                    print(f"Index '{index_name}' not found. Attempting to create...")
                    index_definition = {
                        "mappings": {
                            "dynamic": False,
                            "fields": {
                                "embedding": { "type": "knnVector", "dimensions": 1536, "similarity": "cosine" },
                                "text": { "type": "string", "analyzer": "lucene.standard" }
                            }
                        }
                    }
                    search_index_model = {"name": index_name, "definition": index_definition}
                    try:
                        collection.create_search_index(model=search_index_model)
                        index_status = "created"
                        print(f"Index '{index_name}' creation initiated. It may take time to become queryable.")
                    except OperationFailure as create_err:
                        code_name = getattr(create_err, 'codeName', 'UnknownCode')
                        if code_name == 'IndexAlreadyExists':
                             index_status = "exists" # Race condition: treat as existing
                             print(f"Index '{index_name}' found during creation attempt (race condition). Treating as existing.")
                             index_exists = True
                        else:
                            print(f"MongoDB Operation Failure creating index '{index_name}': CodeName={code_name}, Details={create_err.details}")
                            index_status = f"failed_create_operation_{code_name}"
                    except Exception as create_err:
                        print(f"Unexpected error creating index '{index_name}': {type(create_err).__name__} - {create_err}")
                        traceback.print_exc(); index_status = f"failed_create_unexpected"

            except (OperationFailure, ConnectionFailure) as list_err:
                 code_name = getattr(list_err, 'codeName', 'UnknownCode')
                 print(f"Database error checking/listing search index '{index_name}': CodeName={code_name}, Details={getattr(list_err, 'details', str(list_err))}")
                 index_status = f"failed_list_or_connection_{code_name}"
            except Exception as outer_idx_err:
                print(f"Unexpected error during index existence check: {type(outer_idx_err).__name__} - {outer_idx_err}")
                traceback.print_exc(); index_status = f"failed_check_unexpected"

            print(f"Index check/creation finished ({time.time() - start_time_index_check:.2f}s). Final Status: {index_status}")

        # Handle cases where index check was skipped
        elif not processed_chunks_count > 0:
             index_status = "skipped_no_chunks_or_text"
             print("Skipping index check as no chunks were processed.")
        elif first_error:
             index_status = "skipped_due_to_db_error"
             print(f"Skipping index check due to earlier critical database error ({type(first_error).__name__}).")
        else:
             index_status = "skipped_unknown_reason"
             print("WARN: Index check skipped for an unknown reason.")


        # --- 8. Return Response ---
        processing_time = round(time.time() - start_time_total, 2)
        final_status_code = 200; response_status = "success"; message = "PDF processed successfully."

        if errors:
            if inserted_count > 0:
                final_status_code = 207; response_status = "partial_success"
                message = f"Processed {processed_chunks_count} chunks with {len(errors)} errors. {inserted_count} chunks inserted."
            else:
                if first_error and isinstance(first_error, HTTPException) and 400 <= first_error.status_code < 500: final_status_code = first_error.status_code
                elif first_error: final_status_code = 500;
                if isinstance(first_error, (OperationFailure, ConnectionFailure)): final_status_code = 503
                elif isinstance(first_error, HTTPException) and first_error.status_code >= 500: final_status_code = first_error.status_code
                else: final_status_code = 400 # Default if no inserts and non-5xx critical error
                response_status = "failed"
                message = f"Processing failed with {len(errors)} errors. No chunks were inserted."

            if first_error:
                 critical_error_message = f" Processing stopped early due to critical error: {type(first_error).__name__}."
                 if isinstance(first_error, HTTPException):
                     if first_error.status_code in [401, 403, 413, 429] or first_error.status_code >= 500: final_status_code = first_error.status_code
                     elif final_status_code < 400: final_status_code = first_error.status_code
                     critical_error_message += f" (HTTP {first_error.status_code}: {first_error.detail})"
                 elif isinstance(first_error, (OperationFailure, ConnectionFailure)):
                     final_status_code = 503
                     critical_error_message += f" (DB Error: {getattr(first_error, 'details', str(first_error))})"
                 else:
                     if final_status_code < 500: final_status_code = 500
                     critical_error_message += f" (Error: {str(first_error)})"
                 message += critical_error_message
                 if final_status_code >= 500 or final_status_code == 429 or final_status_code == 503: response_status = "failed"
                 elif final_status_code == 207: response_status = "partial_success"
                 elif final_status_code >= 400: response_status = "failed"

        if index_status.startswith("failed_"): message += f" Index status: {index_status}."

        response_body = {
            "status": response_status, "message": message,
            "chunks_processed": processed_chunks_count, "chunks_inserted": inserted_count,
            "duplicates_removed": duplicates_removed_count, "vector_index_name": index_name,
            "vector_index_status": index_status, "processing_time_seconds": processing_time,
        }
        if errors:
            response_body["errors_sample"] = [
                {"chunk_index": e.get("chunk_index", "N/A"), "status_code": e.get("status_code", 500), "error": str(e.get("error", "Unknown"))}
                for e in errors[:10]
            ]

        print(f"Responding with status code {final_status_code}. Final index status: {index_status}")
        return JSONResponse(content=response_body, status_code=final_status_code)

    except HTTPException as http_exc:
        print(f"Caught HTTPException: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        print(f"Unexpected top-level error in '/process' endpoint: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during PDF processing: {str(e)}")

# /search uses Header auth
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, user: Dict = Depends(get_user_header)):
    """Performs hybrid search using RRF. Requires X-API-Key header."""
    index_name = "vector_index" # The $vectorSearch and $search stages use the same index name
    print(f"Hybrid search request: query='{request.query[:50]}...', collection='{request.collection_name}', limit={request.limit}, user={user.get('supabase_user_id')}")

    try:
        db = MongoDBManager.get_user_db(user)
        # Check if collection exists before querying
        try:
            if request.collection_name not in db.list_collection_names():
                raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found.")
        except OperationFailure as e:
            print(f"Database error listing collections during search: {e.details}")
            raise HTTPException(status_code=503, detail=f"Database error checking collection existence: {e.details}")

        collection = db[request.collection_name]

        # 1. Get query embedding (asynchronously)
        try:
            query_embedding = await asyncio.to_thread(get_openai_embedding, request.query)
            if not query_embedding:
                raise HTTPException(status_code=400, detail="Failed to generate embedding for the query.")
        except HTTPException as e:
            # Re-raise embedding errors (like OpenAI API issues)
            raise e
        except Exception as e:
            print(f"Unexpected error during query embedding: {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Failed to generate query embedding due to an unexpected error.")

        # 2. Define the aggregation pipeline for Hybrid Search with RRF
        num_candidates = max(request.limit * 10, min(request.num_candidates, 1000)) # Adjust numCandidates based on limit
        rrf_k = 60 # RRF constant (can be tuned)
        print(f"Using numCandidates: {num_candidates} for vector and text search phases.")

        pipeline = [
            # --- Vector Search Branch ---
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": num_candidates, # Fetch more candidates for ranking
                    "limit": num_candidates # Return num_candidates results from this stage
                }
            },
            # Add vector rank (vr)
            {
                "$group": {
                    "_id": None,
                    "docs": {"$push": {"doc": "$$ROOT", "vector_score": {"$meta": "vectorSearchScore"}}}
                }
            },
            {"$unwind": {"path": "$docs", "includeArrayIndex": "vr_tmp"}}, # vr_tmp is 0-based index
            {"$replaceRoot": {"newRoot": {"$mergeObjects": ["$docs.doc", {"vr": {"$add": ["$vr_tmp", 1]}}]}}}, # Make vr 1-based rank
            {"$project": {"_id": 1, "text": 1, "metadata": 1, "vr": 1, "embedding": 0}}, # Exclude embedding

            # --- Union with Text Search Branch ---
            {
                "$unionWith": {
                    "coll": request.collection_name, # Search the same collection
                    "pipeline": [
                        # $search stage for text search
                        {
                            "$search": {
                                "index": index_name, # Use the same index containing the text field mapping
                                "text": {
                                    "query": request.query,
                                    "path": "text" # Field defined in index mapping
                                }
                                # Add "fuzzy": {} for typo tolerance if needed
                                # Add highlight, etc. if needed
                            }
                        },
                        {"$limit": num_candidates}, # Limit text search results too
                        # Add text rank (tr)
                        {
                            "$group": {
                                "_id": None,
                                "docs": {"$push": {"doc": "$$ROOT", "text_score": {"$meta": "searchScore"}}}
                            }
                        },
                        {"$unwind": {"path": "$docs", "includeArrayIndex": "tr_tmp"}}, # tr_tmp is 0-based index
                        {"$replaceRoot": {"newRoot": {"$mergeObjects": ["$docs.doc", {"tr": {"$add": ["$tr_tmp", 1]}}]}}}, # Make tr 1-based rank
                        {"$project": {"_id": 1, "text": 1, "metadata": 1, "tr": 1, "embedding": 0}} # Exclude embedding
                    ]
                }
            },

            # --- Combine and Rank using RRF ---
            {
                "$group": {
                    "_id": "$_id", # Group by original document ID
                    "text": {"$first": "$text"},
                    "metadata": {"$first": "$metadata"},
                    "vr": {"$min": "$vr"}, # Get the best vector rank (if duplicated)
                    "tr": {"$min": "$tr"}  # Get the best text rank (if duplicated)
                }
            },
            {
                "$addFields": {
                    # Calculate RRF score: 1/(k + rank) for each, sum them
                    # Use $ifNull to handle cases where a doc appears in only one result set (rank would be null)
                    "rrf_score": {
                        "$sum": [
                            {"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$vr"]}]}, 0]}, # Score from vector rank
                            {"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$tr"]}]}, 0]}  # Score from text rank
                        ]
                    }
                }
            },

            # --- Final Sorting and Formatting ---
            {"$sort": {"rrf_score": -1}}, # Sort by descending RRF score
            {"$limit": request.limit}, # Apply the final limit
            {
                "$project": {
                    "_id": 0, # Exclude MongoDB ObjectId
                    "id": {"$toString": "$_id"}, # Return ID as string
                    "text": 1,
                    "metadata": 1,
                    "score": "$rrf_score" # Return the final RRF score
                }
            }
        ]

        # 3. Execute the aggregation pipeline
        start_time = time.time()
        results = list(collection.aggregate(pipeline))
        end_time = time.time()

        print(f"Hybrid search executed in {end_time - start_time:.2f}s. Found {len(results)} results.")

        # Format results according to SearchResponse model
        formatted_results = [SearchResultItem(**res) for res in results]

        return SearchResponse(results=formatted_results)

    except OperationFailure as e:
        print(f"Database error during hybrid search: {e.details} (Code: {e.code})")
        detail = f"Database error during search: {e.details}"
        status_code = 500 # Default to server error
        # Check for specific error codes
        if "index not found" in str(e.details).lower() or getattr(e, 'codeName', '') == 'IndexNotFound':
            status_code = 404
            detail = f"Search index '{index_name}' not found or not ready in collection '{request.collection_name}'. Please ensure processing was successful."
        elif e.code == 13: # Authorization failure
            status_code = 403
            detail = "Authorization failed for search operation."
        elif 'vectorSearch' in str(e.details) and 'queryVector' in str(e.details):
             status_code = 400
             detail = f"Invalid query vector format or dimension. {e.details}"

        raise HTTPException(status_code=status_code, detail=detail)
    except HTTPException as e:
        # Re-raise specific HTTP exceptions (e.g., from embedding)
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error during hybrid search: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during search: {e}")

# /vector-search uses Header auth
@app.post("/vector-search", response_model=VectorSearchResponse)
async def vector_search_documents(request: VectorSearchRequest, user: Dict = Depends(get_user_header)):
    """Performs vector-only search with optional metadata filtering. Requires X-API-Key header."""
    index_name = "vector_index"
    print(f"Vector search request: query='{request.query[:50]}...', collection='{request.collection_name}', limit={request.limit}, filter={request.filter}, user={user.get('supabase_user_id')}")

    try:
        db = MongoDBManager.get_user_db(user)
        # Check if collection exists
        try:
            if request.collection_name not in db.list_collection_names():
                raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found.")
        except OperationFailure as e:
            print(f"Database error listing collections during vector search: {e.details}")
            raise HTTPException(status_code=503, detail=f"Database error checking collection existence: {e.details}")

        collection = db[request.collection_name]

        # 1. Get query embedding
        try:
            query_embedding = await asyncio.to_thread(get_openai_embedding, request.query)
            if not query_embedding:
                raise HTTPException(status_code=400, detail="Failed to generate embedding for the query.")
        except HTTPException as e: raise e
        except Exception as e: print(f"Unexpected query embed err: {e}"); traceback.print_exc(); raise HTTPException(status_code=500, detail="Query embed fail.")

        # 2. Define the $vectorSearch stage
        vector_search_stage = {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": request.num_candidates, # Number of candidates for initial search
                "limit": request.limit # Final number of results to return
            }
        }

        # Add filter if provided
        if request.filter:
            print(f"Applying metadata filter to vector search: {request.filter}")
            vector_search_stage["$vectorSearch"]["filter"] = request.filter

        # 3. Define the aggregation pipeline
        pipeline = [
            vector_search_stage,
            {
                "$project": {
                    "_id": 0, # Exclude MongoDB ObjectId
                    "id": {"$toString": "$_id"}, # Return ID as string
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"} # Get the similarity score
                }
            }
        ]

        # 4. Execute the pipeline
        start_time = time.time()
        results = list(collection.aggregate(pipeline))
        end_time = time.time()

        print(f"Vector search executed in {end_time - start_time:.2f}s. Found {len(results)} results.")

        # Format results (Pydantic model validation happens implicitly here)
        formatted_results = [SearchResultItem(**res) for res in results]

        return VectorSearchResponse(results=formatted_results)

    except OperationFailure as e:
        print(f"Database error during vector search: {e.details} (Code: {e.code})")
        detail = f"Database error during vector search: {e.details}"
        status_code = 500
        if "index not found" in str(e.details).lower() or getattr(e, 'codeName', '') == 'IndexNotFound':
            status_code = 404
            detail = f"Search index '{index_name}' not found or not ready in collection '{request.collection_name}'."
        elif e.code == 13: status_code = 403; detail = "Authorization failed for search operation."
        elif '$vectorSearch' in str(e.details):
             # Try to provide more specific feedback for common vector search issues
             if 'queryVector' in str(e.details): status_code=400; detail = f"Invalid query vector format or dimension. {e.details}"
             elif 'filter' in str(e.details): status_code=400; detail = f"Invalid filter syntax or field name. {e.details}"

        raise HTTPException(status_code=status_code, detail=detail)
    except HTTPException as e: raise e
    except Exception as e:
        print(f"Unexpected error during vector search: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during vector search: {e}")


# /user-collections uses Header auth
@app.post("/user-collections", response_model=UserInfoResponse)
def get_user_collections_header_auth(request: UserCollectionsRequest, user: Dict = Depends(get_user_header)):
    """Retrieves user DB name and collection list. Requires X-API-Key header and matching Supabase User ID in body."""
    authenticated_user_id = user.get("supabase_user_id")
    request_user_id = request.supabase_user_id

    print(f"Request for collections for Supabase ID: {request_user_id}, Authenticated user via API Key corresponds to Supabase ID: {authenticated_user_id}")

    # Security check: Ensure the API key used corresponds to the Supabase ID requested in the body
    if authenticated_user_id != request_user_id:
        print(f"Mismatch: API key user ({authenticated_user_id}) != Request body user ({request_user_id})")
        raise HTTPException(status_code=403, detail="Authenticated user via API key does not match the requested Supabase User ID.")

    if mongo_client is None: # Should be caught by startup check, but safeguard
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        db_name = user.get("db_name")
        if not db_name:
            # Data consistency issue in auth_db
            print(f"Error: User data for {authenticated_user_id} is incomplete (missing db_name).")
            raise HTTPException(status_code=500, detail="Internal server error: User data incomplete.")

        print(f"Accessing database: {db_name} for user {authenticated_user_id}")
        try:
            # Get the user's specific database
            user_db = mongo_client[db_name]
            # List the collections within that database
            collection_names = user_db.list_collection_names()
            print(f"Found collections for user {authenticated_user_id}: {collection_names}")
            return UserInfoResponse(db_name=db_name, collections=collection_names)
        except (OperationFailure, ConnectionFailure) as e:
             # Handle errors accessing the user's specific DB or listing collections
             print(f"Database access error for db '{db_name}': {getattr(e, 'details', str(e))}")
             raise HTTPException(status_code=503, detail=f"Database access error: {getattr(e, 'details', str(e))}")
        except Exception as e:
             # Catch unexpected errors during DB access
             print(f"Unexpected error accessing user DB '{db_name}': {type(e).__name__} - {e}")
             traceback.print_exc()
             raise HTTPException(status_code=500, detail=f"Unexpected error retrieving collection list: {e}")

    except HTTPException as e:
        # Re-raise specific HTTP exceptions
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error in /user-collections endpoint for user {authenticated_user_id}: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# --- Collection Management Endpoints (Header Auth) ---

@app.delete("/collections/{collection_name}", response_model=ActionResponse)
async def delete_collection_endpoint(
    collection_name: str = Path(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Name of the collection to delete"),
    user: Dict = Depends(get_user_header) # Dependency Injection last
):
    """
    Deletes the specified collection from the user's database.
    Requires X-API-Key header. WARNING: This operation is irreversible.
    """
    print(f"Received request to delete collection '{collection_name}' for user {user.get('supabase_user_id')}")
    try:
        db = MongoDBManager.get_user_db(user)
        db_name = db.name # Get the actual database name for logging/errors

        # Check if the collection exists before attempting to drop
        collection_names = db.list_collection_names()
        if collection_name not in collection_names:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found in database '{db_name}'.")

        # Attempt to drop the collection
        print(f"Attempting to drop collection '{collection_name}' from database '{db_name}'...")
        db.drop_collection(collection_name)
        # Note: drop_collection doesn't raise an error if the collection doesn't exist after the check,
        # but our check above handles the 404 case. It returns None or raises on actual failure.
        print(f"Successfully initiated drop for collection '{collection_name}'.")

        return ActionResponse(status="success", message=f"Collection '{collection_name}' deleted successfully.")

    except OperationFailure as e:
        # Handle database errors during list_collection_names or drop_collection
        print(f"Database error during delete operation for collection '{collection_name}': {e.details} (Code: {e.code})")
        detail = f"Database operation failed during deletion: {e.details}"
        status_code = 500
        if e.code == 13: status_code = 403; detail = "Authorization failed for delete operation."
        # Add more specific error handling based on codes if needed
        raise HTTPException(status_code=status_code, detail=detail)
    except ConnectionFailure as e:
        # Handle connection errors
        print(f"Database connection failure during delete operation for collection '{collection_name}': {e}")
        raise HTTPException(status_code=503, detail="Database connection lost during delete operation.")
    except HTTPException as e:
        # Re-raise exceptions from dependencies or checks
        raise e
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error deleting collection '{collection_name}': {type(e).__name__}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during collection deletion: {e}")

@app.put("/collections/{current_name}", response_model=ActionResponse)
async def rename_collection_endpoint(
    # Path parameter first
    current_name: str = Path(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Current name of the collection"),
    # Body parameter next
    request: RenameCollectionBody = Body(...), # Use Body(...) for the request body model
    # Dependency Injection last
    user: Dict = Depends(get_user_header)
):
    """
    Renames a collection within the user's database.
    Requires X-API-Key header and the new name in the request body.
    Important: Atlas Search indexes associated with the old name must be manually recreated for the new name.
    """
    new_name = request.new_name # Get new name from request body
    print(f"Received request to rename collection '{current_name}' to '{new_name}' for user {user.get('supabase_user_id')}")

    # Basic validation
    if current_name == new_name:
        raise HTTPException(status_code=400, detail="The new collection name cannot be the same as the current name.")
    # Pattern validation for new_name is handled by the RenameCollectionBody model

    try:
        db = MongoDBManager.get_user_db(user)
        db_name = db.name

        # Check existence of source and target names
        collection_names = db.list_collection_names()
        if current_name not in collection_names:
            raise HTTPException(status_code=404, detail=f"Source collection '{current_name}' not found.")
        if new_name in collection_names:
            raise HTTPException(status_code=409, detail=f"Target collection name '{new_name}' already exists.")

        # Attempt the rename operation using the admin command
        print(f"Attempting to rename '{current_name}' to '{new_name}' in database '{db_name}'...")
        try:
            if mongo_client is None: # Safeguard, should be available if get_user_db worked
                 raise HTTPException(status_code=503, detail="Database client unavailable for rename operation.")
            # The renameCollection command requires admin privileges on the database.
            # Format: db.adminCommand({ renameCollection: "<db>.<oldName>", to: "<db>.<newName>" })
            mongo_client.admin.command('renameCollection', f'{db_name}.{current_name}', to=f'{db_name}.{new_name}')
            print(f"Rename operation successful.")
            # Add a warning about search indexes
            warning_detail = "Important: If you were using Atlas Search on the collection, you must manually recreate the search index for the new collection name."
            return ActionResponse(status="success", message=f"Collection '{current_name}' successfully renamed to '{new_name}'.", details=warning_detail)
        except OperationFailure as e:
            # Handle specific errors from the rename command
            print(f"Database error during rename operation: {e.details} (Code: {e.code})")
            detail = f"Database rename failed: {e.details}"
            status_code = 500
            if e.code == 13: status_code = 403; detail = "Authorization error: Insufficient permissions to rename collection."
            # Common codes for renameCollection:
            elif e.code == 72: status_code = 400; detail = f"Invalid target namespace '{db_name}.{new_name}'." # NamespaceContainsDollar, NamespaceContainsNull, etc.
            elif e.code == 10026: status_code = 409; detail = f"Target collection '{new_name}' already exists (OperationFailure code 10026)." # NamespaceExists
            elif e.code == 26: status_code = 404; detail = f"Source collection '{current_name}' not found (OperationFailure code 26)." # NamespaceNotFound
            raise HTTPException(status_code=status_code, detail=detail)

    except ConnectionFailure as e:
        print(f"Database connection failure during rename operation: {e}")
        raise HTTPException(status_code=503, detail="Database connection lost during rename operation.")
    except HTTPException as e: raise e
    except Exception as e:
        print(f"Unexpected error renaming collection '{current_name}': {type(e).__name__}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during collection rename: {e}")


# --- Application startup ---
if __name__ == "__main__":
    # Perform crucial client initialization checks before starting the server
    startup_errors = []
    if mongo_client is None:
         startup_errors.append("MongoDB Client (mongo_client) is None")
    if auth_db is None:
         startup_errors.append("MongoDB Auth DB (auth_db) is None")
    if openai_client is None:
         startup_errors.append("OpenAI Client (openai_client) is None")

    if startup_errors:
         print("FATAL: Required clients not initialized properly. Server cannot start.")
         for error in startup_errors:
             print(f" - {error}")
         # Use sys.exit for a cleaner exit than raise SystemExit directly in some contexts
         sys.exit("Server cannot start due to initialization failures.")
    else:
        print("All required clients initialized.")
        print("Starting FastAPI server on host 0.0.0.0, port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)