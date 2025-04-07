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
    # Optional: Test call to verify connection (might incur cost)
    # openai_client.models.list()
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"FATAL: Failed to initialize OpenAI client: {e}")
    raise SystemExit(f"Failed to initialize OpenAI client: {e}")

# --- MongoDB Connection ---
mongo_client = None
auth_db = None
try:
    # Set connectTimeoutMS for faster timeout if server is unavailable
    # serverSelectionTimeoutMS determines how long to try finding a suitable server
    mongo_client = MongoClient(
        MONGO_MASTER_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=3000 # Added connection timeout
    )
    # The ismaster command is cheap and does not require auth.
    mongo_client.admin.command('ismaster')
    print("Successfully connected to MongoDB.")
    auth_db = mongo_client["auth_db"]
except ConnectionFailure as e:
    print(f"FATAL: Failed to connect to MongoDB (ConnectionFailure): {e}")
    raise SystemExit(f"MongoDB connection failed: {e}")
except OperationFailure as e: # e.g., Authentication failed if credentials required for ismaster (unlikely)
    print(f"FATAL: Failed to connect to MongoDB (OperationFailure): {e.details}")
    raise SystemExit(f"MongoDB operation failure during connection test: {e}")
except Exception as e: # Other unexpected connection errors
    print(f"FATAL: An unexpected error occurred during MongoDB connection: {type(e).__name__} - {e}")
    raise SystemExit(f"Unexpected MongoDB connection error: {e}")


# --- Data models ---
class UserRegister(BaseModel):
    supabase_user_id: str = Field(..., min_length=1)

# ★★★ /register エンドポイント用の新しいレスポンスモデル ★★★
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

# ★ 新しいエンドポイント用のリクエストモデル
class UserInfoRequest(BaseModel):
    supabase_user_id: str = Field(..., min_length=1, description="The Supabase User ID")
    api_key: str = Field(..., description="The user's API key") # 空チェックはエンドポイント内で行う

# ★ 新しいエンドポイント用のレスポンスモデル
class UserInfoResponse(BaseModel):
    db_name: str
    collections: List[str]


# --- Dependencies ---
def verify_admin(api_key: str = Header(..., alias="X-API-Key")):
    """Verifies the admin API key provided in the header."""
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required")

def get_user(api_key: str = Body(..., description="User's API Key")):
    """
    Retrieves the user based on the API key provided in the request body.
    Checks if auth_db is available.
    """
    if auth_db is None:
         print("Error in get_user: auth_db is not available.")
         raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        user = auth_db.users.find_one({"api_key": api_key})
        if not user:
            raise HTTPException(status_code=403, detail="Invalid API Key")
        return user
    except OperationFailure as e:
        print(f"Error finding user in auth_db: {e.details}")
        raise HTTPException(status_code=503, detail="Database operation failed while validating API key.")
    except Exception as e:
        print(f"Unexpected error in get_user: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected error occurred during API key validation.")

# --- MongoDB manager ---
class MongoDBManager:
    @staticmethod
    def create_user_db(supabase_user_id: str):
        if auth_db is None: # ★ Check if auth_db is initialized
             print("Error in create_user_db: auth_db is not available.")
             raise HTTPException(status_code=503, detail="Database service unavailable")
        # Generate a somewhat unique DB name based on user ID prefix and random hex
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
        if mongo_client is None: # ★ Check if mongo_client is initialized
             print("Error in get_user_db: mongo_client is not available.")
             raise HTTPException(status_code=503, detail="Database service unavailable")
        db_name = user.get("db_name")
        if not db_name:
            # This should not happen if get_user works correctly
            print(f"Error in get_user_db: User object is missing 'db_name'. User: {user.get('supabase_user_id', 'N/A')}")
            raise HTTPException(status_code=500, detail="Internal server error: User data is inconsistent.")
        try:
            # Accessing the database does not throw an error immediately if it doesn't exist
            # Errors occur upon actual operations.
            return mongo_client[db_name]
        except Exception as e: # Catch potential unexpected errors during client access
            print(f"Unexpected error accessing user DB '{db_name}': {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=503, detail=f"Failed to access user database '{db_name}': {e}")


    @staticmethod
    def create_collection(db, name: str, user_id: str):
        # db can be None if get_user_db fails or returns None unexpectedly, though the latter is less likely now.
        if db is None:
             # This indicates a problem likely before this function was called.
             print(f"Error in create_collection: Database object is None for user {user_id}.")
             raise HTTPException(status_code=500, detail="Internal Server Error: Invalid database reference.")

        try:
            # Check if collection exists
            collection_names = db.list_collection_names()
            if name not in collection_names:
                 # Create collection implicitly by inserting and deleting a dummy document
                 # This ensures the collection exists even if empty initially.
                 # Note: MongoDB creates collections automatically on first insert.
                 # This step might be redundant but ensures explicit creation intent.
                 collection = db[name]
                 init_result = collection.insert_one({"__init__": True, "user_id": user_id, "timestamp": datetime.utcnow()})
                 delete_result = collection.delete_one({"_id": init_result.inserted_id})
                 if delete_result.deleted_count == 1:
                     print(f"Collection '{name}' created in database '{db.name}' (init doc method).")
                 else:
                     # This case might happen if the document insertion succeeded but deletion failed immediately (unlikely).
                     print(f"Warning: Collection '{name}' created in '{db.name}', but dummy init doc deletion failed.")
                 return collection # Return the collection object
            else:
                 print(f"Collection '{name}' already exists in database '{db.name}'.")
                 return db[name] # Return the existing collection object

        except OperationFailure as e:
            print(f"MongoDB Operation Failure accessing/creating collection '{name}' in '{db.name}': {e.details}")
            # Provide more context if possible
            detail = f"Database operation failed for collection '{name}': {e.details}"
            # Check for specific auth errors if applicable
            # if e.code == 13: # Example: Authorization failed error code
            #     detail = f"Authorization failed for collection '{name}'. Check permissions."
            raise HTTPException(status_code=500, detail=detail)
        except ConnectionFailure as e:
            # Handle potential connection loss during operation
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

        # Find the furthest possible end position within chunk_size limit
        while end_pos < len(words):
            word_len = len(words[end_pos])
            # Add 1 for space if not the first word
            length_to_add = word_len + (1 if end_pos > current_pos else 0)

            if current_length + length_to_add <= chunk_size:
                current_length += length_to_add
                last_valid_end_pos = end_pos + 1 # Include the current word
                end_pos += 1
            else:
                # Current word makes it too long
                break

        # If a single word is longer than chunk_size, take just that word
        if last_valid_end_pos == current_pos:
             # Check if we are at the very start and the first word is too long
             if current_pos == 0 and len(words[0]) > chunk_size:
                 print(f"Warning: Single word exceeds chunk size: '{words[0][:50]}...'")
                 # Take the oversized word as its own chunk
                 chunks.append(words[0])
                 current_pos += 1
                 continue # Move to the next word
             # Otherwise, if it's not the first word or the first word fits, proceed normally
             # However, we must advance, so take at least one word if possible
             if current_pos < len(words):
                 last_valid_end_pos = current_pos + 1
             else: # Reached end of words
                 break


        # Slice words for the current chunk
        chunk_words = words[current_pos:last_valid_end_pos]
        chunks.append(" ".join(chunk_words))

        # Move current_pos for the next chunk, considering overlap
        # Find a suitable overlap start point (go back roughly `overlap` chars)
        overlap_start_index = last_valid_end_pos - 1 # Start from the last word of the current chunk
        overlap_char_count = 0
        # Iterate backwards from the end of the chunk
        while overlap_start_index > current_pos:
             # Count characters of the word at overlap_start_index
             overlap_char_count += len(words[overlap_start_index]) + 1 # +1 for the space
             # If we've accumulated enough overlap characters, stop
             if overlap_char_count >= overlap:
                  break
             overlap_start_index -= 1

        # Ensure overlap_start_index doesn't go before current_pos
        overlap_start_index = max(current_pos, overlap_start_index)

        # Determine the next starting position
        # Move start position forward, ensuring progress. If overlap calculation results
        # in the same position or earlier, force it to move forward by at least one word.
        next_pos = overlap_start_index
        # If the overlap didn't move us forward enough (e.g., very short words, large overlap),
        # ensure we advance by at least one position relative to the *start* of the current chunk.
        # Also handle the case where last_valid_end_pos is the end of the list.
        # If next_pos <= current_pos, it means overlap didn't move us forward, so advance by 1.
        # Otherwise, use the calculated overlap_start_index.
        if next_pos <= current_pos:
            current_pos += 1
        else:
            current_pos = next_pos

        # Final check: Ensure we always move forward from the start of the last chunk processed
        # This prevents getting stuck if overlap logic leads to a non-advancing position.
        if current_pos < last_valid_end_pos:
             current_pos = last_valid_end_pos # Move past the chunk we just processed


    return chunks

# --- ★ OpenAI Embedding Function ---
def get_openai_embedding(text: str) -> List[float]:
    """Generates embedding for the given text using OpenAI API."""
    if not text or text.isspace():
         print("Warning: Attempted to get embedding for empty or whitespace text.")
         # OpenAI API will error on empty input, so handle it here.
         # Returning empty list might cause issues later if not handled.
         # Consider raising ValueError or returning None and checking later.
         # For now, return empty list as per original structure.
         return []

    try:
        # Simple cleaning: replace multiple whitespace chars with a single space
        cleaned_text = ' '.join(text.split())
        if not cleaned_text: # Check again after cleaning
             print("Warning: Text became empty after cleaning whitespace.")
             return []

        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=cleaned_text,
            # Optional: specify dimensions if the model supports it and you need shorter vectors
            # dimensions=1024
        )
        # The response structure is OpenAIObject containing a list of Embedding objects in `data`
        if response.data and len(response.data) > 0 and response.data[0].embedding:
            return response.data[0].embedding
        else:
             # This case should be rare if the API call succeeds
             print(f"Warning: OpenAI API returned no embedding data for text: {cleaned_text[:100]}...")
             # Raise an internal server error as this is unexpected behavior from OpenAI API
             raise HTTPException(status_code=500, detail="OpenAI API returned unexpected empty data.")

    # Specific OpenAI errors
    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to OpenAI API: {e}")
    except openai.APIStatusError as e: # Covers non-200 responses (like 4xx, 5xx from OpenAI)
        print(f"OpenAI API Status Error: Status Code {e.status_code}, Response: {e.response}")
        status_code = e.status_code
        detail = f"OpenAI Service Error (Status {status_code}): {e.message or e.response.text}"
        # Map specific OpenAI status codes to appropriate HTTP status codes
        if status_code == 400: # Bad Request (e.g., input too long after chunking - shouldn't happen often)
            detail = f"OpenAI Bad Request: Input may be invalid. {e.message}"
            raise HTTPException(status_code=400, detail=detail)
        elif status_code == 401: # Authentication Error
            # This should ideally be caught at startup, but key might expire/be revoked
            detail = "OpenAI Authentication Error. Check API Key configuration."
            raise HTTPException(status_code=401, detail=detail)
        elif status_code == 429: # Rate Limit Error
            detail = "OpenAI Rate Limit Exceeded. Please wait and retry."
            raise HTTPException(status_code=429, detail=detail)
        elif status_code >= 500: # Server error on OpenAI's side
            detail = f"OpenAI Server Error (Status {status_code}). Please retry later. {e.message}"
            raise HTTPException(status_code=502, detail=detail) # 502 Bad Gateway seems appropriate
        else: # Other 4xx errors
             raise HTTPException(status_code=status_code, detail=detail)
    # Catching the base openai.APIError might be sufficient if fine-grained handling isn't needed
    # except openai.APIError as e:
    #     print(f"OpenAI API Error: {e}")
    #     raise HTTPException(status_code=502, detail=f"OpenAI Service Error: {e}")
    except Exception as e: # Catch any other unexpected errors (network issues, library bugs)
        print(f"An unexpected error occurred while getting OpenAI embedding: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding generation failed due to an unexpected error: {e}")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Name of the collection to search within")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    num_candidates: int = Field(100, ge=10, le=1000, description="Number of candidates to consider for vector search (higher value increases recall but may impact performance)")

class SearchResultItem(BaseModel):
    id: str # Use string representation of ObjectId
    text: str
    metadata: Dict[str, Any] # Allow any type in metadata values
    score: float

# Vector Search専用のリクエストモデル
class VectorSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Name of the collection to search within")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    num_candidates: int = Field(100, ge=10, le=1000, description="Number of candidates for initial vector search phase")
    # Optional filter field
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filter criteria for metadata (e.g., {\"metadata.category\": \"news\"})")

# Vector Search専用のレスポンスモデル
class VectorSearchResponse(BaseModel):
    results: List[SearchResultItem] # Reuse SearchResultItem

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

# --- API endpoints ---
@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    # Improve by checking DB/OpenAI client status if possible without making costly calls
    db_status = "connected" if mongo_client and auth_db else "disconnected"
    openai_status = "initialized" if openai_client else "not_initialized"
    if db_status == "connected":
        try:
            # Quick, non-blocking check
            mongo_client.admin.command('ping')
        except (ConnectionFailure, OperationFailure) as e:
            db_status = f"error ({type(e).__name__})"

    return {"status": "ok", "database": db_status, "openai_client": openai_status}

@app.get("/auth-db", dependencies=[Depends(verify_admin)])
def get_auth_db_contents():
    """(Admin Only) Retrieve non-sensitive user info from the auth database."""
    if auth_db is None: # ★ Check auth_db availability
         raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        # Exclude sensitive fields like _id and api_key
        users = list(auth_db.users.find({}, {"_id": 0, "api_key": 0}))
        return {"users": users}
    except OperationFailure as e:
        print(f"Error reading from auth_db: {e.details}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user data due to database operation error: {e.details}")
    except Exception as e:
        print(f"Unexpected error reading from auth_db: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving user data.")


# ★★★ /register エンドポイントの修正 ★★★
@app.post("/register", response_model=RegisterResponse)
def register_user(request: UserRegister, response: Response): # Inject Response object to set status code
    """
    Registers a new user or retrieves existing user info based on Supabase ID.
    Returns API key, DB name, and whether the user/database already existed.
    - Status 201 Created: New user and database created.
    - Status 200 OK: User and database already existed, returning existing info.
    """
    if auth_db is None:
        print("Error in register_user: auth_db is not available.")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Check if user already exists
        print(f"Checking existence for supabase_user_id: {request.supabase_user_id}")
        existing_user = auth_db.users.find_one({"supabase_user_id": request.supabase_user_id})

        if existing_user:
            # User exists, retrieve info
            print(f"User {request.supabase_user_id} already exists. Returning existing credentials.")
            api_key = existing_user.get("api_key")
            db_name = existing_user.get("db_name")

            # Validate retrieved data (should always exist if record is valid)
            if not api_key or not db_name:
                print(f"Error: Existing user {request.supabase_user_id} record in auth_db is incomplete (missing api_key or db_name).")
                raise HTTPException(status_code=500, detail="Internal Server Error: Existing user data is inconsistent.")

            # Set status code to 200 OK for existing user
            response.status_code = 200
            return RegisterResponse(
                api_key=api_key,
                db_name=db_name,
                database_exist=True
            )
        else:
            # User does not exist, create new user record and DB info
            print(f"User {request.supabase_user_id} not found. Creating new user and database info...")
            # MongoDBManager.create_user_db handles insertion into auth_db
            db_info = MongoDBManager.create_user_db(request.supabase_user_id)

            # Set status code to 201 Created for new user
            response.status_code = 201
            return RegisterResponse(
                api_key=db_info["api_key"],
                db_name=db_info["db_name"],
                database_exist=False
            )

    except HTTPException as e:
         # Re-raise HTTPExceptions from create_user_db or the internal validation checks
         # Ensure status code is preserved if already set by the exception
         raise e
    except OperationFailure as e: # Catch potential DB errors during find_one or insert_one (via create_user_db)
        print(f"Database operation error during user registration check/creation: {e.details}")
        raise HTTPException(status_code=500, detail=f"Database error during user registration: {e.details}")
    except Exception as e:
        print(f"Unexpected error during user registration: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during user registration: {e}")


@app.post("/collections", status_code=201)
def create_collection_endpoint(
    request: CollectionCreate,
    user: Dict = Depends(get_user) # get_user handles API key validation and retrieves user data
):
    """Creates a new collection within the user's dedicated database."""
    try:
        # Get the database object for the validated user
        db = MongoDBManager.get_user_db(user)
        # Create the collection (or ensure it exists)
        collection = MongoDBManager.create_collection(db, request.name, user["supabase_user_id"])
        return {"status": "created", "collection_name": collection.name} # Use collection.name for certainty
    except HTTPException as e:
         # Re-raise HTTPExceptions from get_user_db or create_collection
         raise e
    except Exception as e:
        # Catch unexpected errors not already converted to HTTPException
        print(f"Unexpected error creating collection '{request.name}' for user {user.get('supabase_user_id', 'N/A')}: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create collection due to an unexpected error: {e}")

@app.post("/process")
async def process_pdf(
    request: ProcessRequest,
    user: Dict = Depends(get_user) # Validates API key and provides user context
):
    """Downloads a PDF, extracts text, chunks, embeds, and stores it in the user's collection."""
    index_name = "vector_index" # Fixed name for the Atlas Search index
    index_status = "not_checked" # Tracks index management outcome
    first_error = None # Stores the first critical error encountered during chunk processing
    duplicates_removed_count = 0 # Tracks removed duplicate documents
    inserted_count = 0 # Tracks successfully inserted documents
    processed_chunks_count = 0 # Tracks chunks attempted (excluding empty ones)
    errors = [] # List to store errors encountered per chunk

    start_time_total = time.time()
    print(f"Processing PDF from URL for user {user.get('supabase_user_id', 'N/A')}: {request.pdf_url}")

    try:
        # --- 1. Download PDF ---
        pdf_content: io.BytesIO
        try:
            print(f"Downloading PDF from {request.pdf_url}...")
            start_time_download = time.time()
            # Run blocking requests.get in a separate thread
            response = await asyncio.to_thread(requests.get, str(request.pdf_url), timeout=60)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            end_time_download = time.time()
            print(f"PDF downloaded successfully in {end_time_download - start_time_download:.2f}s.")

            # Check file size based on Content-Length header (if available)
            content_length = response.headers.get('Content-Length')
            if content_length:
                pdf_size = int(content_length)
                if pdf_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413, # Payload Too Large
                        detail=f"PDF file size ({pdf_size / (1024*1024):.2f} MB) exceeds the limit of {MAX_FILE_SIZE / (1024*1024)} MB."
                    )
                print(f"PDF size: {pdf_size / (1024*1024):.2f} MB.")
            else:
                # If Content-Length is missing, check after reading into memory (less ideal)
                pdf_bytes = response.content
                pdf_size = len(pdf_bytes)
                if pdf_size > MAX_FILE_SIZE:
                     raise HTTPException(
                         status_code=413,
                         detail=f"PDF file size ({pdf_size / (1024*1024):.2f} MB) exceeds the limit of {MAX_FILE_SIZE / (1024*1024)} MB (checked after download)."
                     )
                pdf_content = io.BytesIO(pdf_bytes)
                print(f"PDF size: {pdf_size / (1024*1024):.2f} MB (checked after download).")


            if 'pdf_content' not in locals(): # If Content-Length was present and checked
                 pdf_content = io.BytesIO(response.content)


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
            # PyPDF2 is CPU/IO bound, running in thread might help slightly on multi-core
            # but benefits might be limited by GIL for pure Python parts.
            def extract_text_sync(pdf_stream):
                 reader = PyPDF2.PdfReader(pdf_stream)
                 if reader.is_encrypted:
                      # Note: is_encrypted is basic; might not catch all encryption types.
                      # Password-protected PDFs are not supported.
                      raise ValueError("Encrypted PDF files are not supported.")
                 extracted_text = ""
                 for page_num, page in enumerate(reader.pages):
                      try:
                          page_text = page.extract_text()
                          if page_text:
                               extracted_text += page_text + "\n" # Add newline between pages
                      except Exception as page_error:
                           print(f"Warning: Could not extract text from page {page_num + 1}: {page_error}")
                           continue # Skip problematic pages
                 return extracted_text

            text = await asyncio.to_thread(extract_text_sync, pdf_content)
            end_time_extract = time.time()

            if not text.strip():
                 print("Warning: No text could be extracted from the PDF.")
                 # Return a success response indicating no content was found
                 return JSONResponse(
                      content={
                           "status": "success",
                           "message": "No text content found in PDF.",
                           "chunks_processed": 0,
                           "chunks_inserted": 0,
                           "duplicates_removed": 0,
                           "vector_index_name": index_name,
                           "vector_index_status": "skipped_no_text",
                           "processing_time_seconds": round(time.time() - start_time_total, 2)
                           },
                      status_code=200 # Operation successful, but no data yielded
                 )
            print(f"Text extracted successfully ({len(text)} chars) in {end_time_extract - start_time_extract:.2f}s.")

        except PyPDF2.errors.PdfReadError as pdf_error:
            print(f"Error reading PDF structure (PyPDF2): {pdf_error}")
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF file: {pdf_error}")
        except ValueError as val_err: # Catch the encrypted PDF error
             print(f"ValueError during PDF processing: {val_err}")
             raise HTTPException(status_code=400, detail=str(val_err))
        except Exception as e: # Catch other unexpected PyPDF2 errors
            print(f"Unexpected error during PDF text extraction: {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF due to an unexpected error: {e}")

        # --- 3. Split text ---
        print("Splitting text into chunks...")
        start_time_split = time.time()
        # Use a chunk size appropriate for the embedding model context window
        # OpenAI ada-002 has 8191 tokens. 1500 chars is conservative.
        # Using tiktoken library would be more accurate for token counting.
        chunks = split_text_into_chunks(text, chunk_size=1500, overlap=100)
        end_time_split = time.time()
        print(f"Text split into {len(chunks)} chunks in {end_time_split - start_time_split:.2f}s.")

        if not chunks:
             # This might happen if the extracted text was minimal and only whitespace after splitting
             print("Warning: No text chunks were generated after splitting.")
             return JSONResponse(
                  content={
                      "status": "success",
                      "message": "No processable text chunks generated after splitting.",
                      "chunks_processed": 0,
                      "chunks_inserted": 0,
                      "duplicates_removed": 0,
                      "vector_index_name": index_name,
                      "vector_index_status": "skipped_no_chunks",
                      "processing_time_seconds": round(time.time() - start_time_total, 2)
                  },
                  status_code=200
             )

        # --- 4. Database operations setup ---
        db = MongoDBManager.get_user_db(user)
        collection = MongoDBManager.create_collection(db, request.collection_name, user["supabase_user_id"])
        print(f"Using collection '{request.collection_name}' in database '{db.name}'.")

        # --- 5. Process Chunks (Embed and Insert) ---
        print(f"Starting processing of {len(chunks)} chunks...")
        start_time_chunks = time.time()

        # Consider using asyncio.gather for parallel embedding generation if rate limits allow
        # tasks = []
        # for i, chunk in enumerate(chunks):
        #     if chunk and not chunk.isspace():
        #          tasks.append(process_single_chunk(i, chunk, collection, request, user)) # Define helper async func
        # results = await asyncio.gather(*tasks, return_exceptions=True)
        # Process results...

        # Sequential processing (simpler, safer for rate limits initially)
        for i, chunk in enumerate(chunks):
            if not chunk or chunk.isspace():
                 print(f"Skipping empty chunk {i+1}/{len(chunks)}.")
                 continue

            processed_chunks_count += 1
            try:
                print(f"Processing chunk {i+1}/{len(chunks)} (Length: {len(chunk)})...")
                # Generate embedding (run blocking OpenAI call in thread)
                start_time_embed = time.time()
                embedding = await asyncio.to_thread(get_openai_embedding, chunk)
                end_time_embed = time.time()

                if not embedding:
                     # This can happen if get_openai_embedding returns [] for empty/whitespace
                     print(f"Skipping chunk {i+1} due to empty embedding result (original chunk might have been whitespace).")
                     errors.append({"chunk_index": i, "error": "Skipped due to empty embedding result", "status_code": 400})
                     continue # Skip insertion if embedding failed or was empty
                print(f"Embedding generated for chunk {i+1} in {end_time_embed - start_time_embed:.2f}s.")

                # Prepare document for insertion
                doc_to_insert = {
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {
                        **request.metadata, # Merge provided metadata
                        "chunk_index": i,
                        "original_url": str(request.pdf_url), # Store source URL
                        "processed_at": datetime.utcnow()
                        },
                    "created_at": datetime.utcnow() # Document creation timestamp
                }

                # Insert document into MongoDB
                start_time_insert = time.time()
                insert_result = collection.insert_one(doc_to_insert)
                end_time_insert = time.time()
                if insert_result.inserted_id:
                    inserted_count += 1
                    print(f"Chunk {i+1} inserted successfully (ID: {insert_result.inserted_id}) in {end_time_insert - start_time_insert:.2f}s.")
                else:
                    # Should not happen with insert_one unless exception occurs
                    print(f"Warning: Chunk {i+1} insertion reported no ID, but no exception was raised.")
                    errors.append({"chunk_index": i, "error": "Insertion failed silently", "status_code": 500})


            except HTTPException as http_exc:
                # Handle errors from get_openai_embedding or other potential HTTPExceptions
                print(f"HTTP Error processing chunk {i+1}: Status {http_exc.status_code} - {http_exc.detail}")
                errors.append({"chunk_index": i, "error": http_exc.detail, "status_code": http_exc.status_code})
                # Decide whether to stop based on the error type
                is_critical = http_exc.status_code == 429 or http_exc.status_code >= 500
                if is_critical and first_error is None:
                    first_error = http_exc
                    print(f"Stopping chunk processing due to critical error (Status: {http_exc.status_code}).")
                    break # Stop processing further chunks

            except (OperationFailure, ConnectionFailure) as db_error:
                # Handle MongoDB specific errors during insert
                error_detail = getattr(db_error, 'details', str(db_error))
                print(f"Database Error processing chunk {i+1}: {type(db_error).__name__} - {error_detail}")
                errors.append({"chunk_index": i, "error": f"Database error: {error_detail}", "status_code": 503}) # Service Unavailable seems appropriate
                if first_error is None:
                    first_error = db_error
                    print("Stopping chunk processing due to critical database error.")
                    break # Stop processing

            except Exception as chunk_error:
                # Catch any other unexpected error during chunk processing
                print(f"Unexpected Error processing chunk {i+1}: {type(chunk_error).__name__} - {chunk_error}")
                traceback.print_exc()
                error_detail = f"Unexpected error: {chunk_error}"
                errors.append({"chunk_index": i, "error": error_detail, "status_code": 500})
                if first_error is None:
                     first_error = chunk_error
                     print("Stopping chunk processing due to unexpected critical error.")
                     break # Stop processing


        end_time_chunks = time.time()
        print(f"Chunk processing finished in {end_time_chunks - start_time_chunks:.2f}s. "
              f"Processed: {processed_chunks_count}, Inserted: {inserted_count}, Errors: {len(errors)}")

        # --- 6. Remove Duplicate Chunks (Optional but Recommended) ---
        # Run this only if new data was inserted and no critical error stopped processing early
        if inserted_count > 0 and not first_error:
            print(f"Checking for and removing duplicate chunks based on text content for URL: {request.pdf_url}...")
            start_time_dedup = time.time()
            try:
                # Pipeline to find duplicate text content originating from the *same URL*
                pipeline = [
                    # Match documents from the specific URL processed in this run
                    {"$match": {"metadata.original_url": str(request.pdf_url)}},
                    # Group by the text content
                    {"$group": {
                        "_id": "$text", # Group by the text field
                        "ids": {"$push": "$_id"}, # Collect all ObjectIds for each unique text
                        "count": {"$sum": 1} # Count occurrences of each text
                    }},
                    # Filter groups where the count is greater than 1 (duplicates)
                    {"$match": {"count": {"$gt": 1}}}
                ]
                duplicate_groups = list(collection.aggregate(pipeline))

                ids_to_delete = []
                if duplicate_groups:
                    print(f"Found {len(duplicate_groups)} text groups with duplicates.")
                    for group in duplicate_groups:
                        # Keep the first ID (arbitrarily), delete the rest
                        ids_to_delete.extend(group['ids'][1:]) # Add all IDs except the first one to the deletion list

                    if ids_to_delete:
                        print(f"Attempting to delete {len(ids_to_delete)} duplicate documents...")
                        delete_result = collection.delete_many({"_id": {"$in": ids_to_delete}})
                        duplicates_removed_count = delete_result.deleted_count
                        print(f"Successfully deleted {duplicates_removed_count} duplicate documents.")
                        # Adjust inserted_count conceptually, though it represents initial insertions
                        # final_inserted_count = inserted_count - duplicates_removed_count
                    else:
                        print("No duplicate documents needed deletion (logic error?).") # Should not happen if ids_to_delete has items
                else:
                    print("No duplicate text content found for this URL.")

            except OperationFailure as agg_error:
                # Log error but don't necessarily stop the whole process; index creation might still be valuable
                print(f"MongoDB Operation Failure during duplicate check/removal: {agg_error.details}")
                index_status = "duplicates_check_failed_operation" # Mark that deduplication failed
            except Exception as agg_error:
                print(f"Unexpected error during duplicate check/removal: {type(agg_error).__name__} - {agg_error}")
                traceback.print_exc()
                index_status = "duplicates_check_failed_unexpected" # Mark that deduplication failed

            end_time_dedup = time.time()
            print(f"Duplicate check/removal finished in {end_time_dedup - start_time_dedup:.2f}s.")
        elif first_error:
            print("Skipping duplicate removal due to critical errors during chunk processing.")
            index_status = "duplicates_skipped_due_to_error"
        else:
            print("Skipping duplicate removal as no new chunks were successfully inserted.")
            index_status = "duplicates_skipped_no_inserts"


        # --- 7. Manage Vector Search Index ---
        # Recreate index if data changed (inserted or duplicates removed) and no critical errors occurred.
        data_changed = inserted_count > 0 or duplicates_removed_count > 0
        if data_changed and not first_error:
            attempt_creation = True
            index_dropped = False
            print(f"Data changed (Inserted: {inserted_count}, Removed: {duplicates_removed_count}). Managing vector search index '{index_name}'...")
            start_time_index = time.time()
            try:
                # Check for existing index
                print(f"Checking for existing index '{index_name}'...")
                existing_indexes = list(collection.list_search_indexes(index_name)) # More efficient check
                index_exists = bool(existing_indexes) # True if the list is not empty

                if index_exists:
                    print(f"Index '{index_name}' found. Attempting to drop it before recreation...")
                    try:
                        collection.drop_search_index(index_name)
                        print(f"Successfully initiated drop for index '{index_name}'.")
                        # IMPORTANT: Index drop is asynchronous. A short wait increases chances
                        # the subsequent create call won't conflict, but it's not guaranteed.
                        # Production systems might need polling or webhooks to confirm drop completion.
                        wait_time = 20 # seconds
                        print(f"Waiting {wait_time}s for index drop to propagate...")
                        await asyncio.sleep(wait_time) # Use asyncio.sleep in async context
                        index_dropped = True
                        print("Wait finished. Proceeding to create new index.")
                    except OperationFailure as drop_err:
                        print(f"MongoDB Operation Failure dropping index '{index_name}': {drop_err.details}")
                        # Depending on the error, maybe creation shouldn't be attempted.
                        # If it's e.g. "index not found" (race condition?), we can proceed.
                        # If it's auth error, we should stop.
                        # For simplicity here, log and prevent creation attempt.
                        index_status = f"failed_drop_operation: {drop_err.codeName or drop_err.details}"
                        attempt_creation = False
                    except Exception as drop_err:
                        print(f"Unexpected error dropping index '{index_name}': {type(drop_err).__name__} - {drop_err}")
                        traceback.print_exc()
                        index_status = f"failed_drop_unexpected: {str(drop_err)}"
                        attempt_creation = False
                else:
                    print(f"Index '{index_name}' not found. Will create a new one.")

                # Define the index for hybrid search (vector + text)
                # Ensure dimensions match the OpenAI model being used.
                # text-embedding-ada-002 = 1536 dimensions.
                # Check OpenAI documentation if using a different model.
                openai_dimensions = 1536 # Default for ada-002
                # Add logic here if OPENAI_EMBEDDING_MODEL could vary and have different dimensions

                if attempt_creation:
                    print(f"Defining index definition for '{index_name}'...")
                    index_definition = {
                        "mappings": {
                            "dynamic": False, # Explicitly define fields to index
                            "fields": {
                                "embedding": {
                                    "type": "knnVector",
                                    "dimensions": openai_dimensions,
                                    "similarity": "cosine" # Cosine similarity is common for OpenAI embeddings
                                },
                                "text": {
                                    "type": "string",
                                    "analyzer": "lucene.standard", # Standard analyzer works well for English/many languages
                                    # For multi-language or specific language needs, consider:
                                    # "analyzer": "lucene.standard", # General purpose
                                    # "analyzer": "lucene.english", # English specific stemming, stop words
                                    # "analyzer": "lucene.kuromoji", # Japanese
                                    # "multi": {"myAnalyzer": {"type": "multiLanguage", "languages": ["en", "es"]}} # Multi-language
                                    # Add other options if needed, e.g., for phrase search accuracy:
                                    # "indexOptions": "positions"
                                }
                                # Optionally index metadata fields if needed for filtering/searching
                                # "metadata.category": {
                                #     "type": "string",
                                #     "analyzer": "lucene.keyword" # Good for exact matches on categories/tags
                                # },
                                # "metadata.original_url": {
                                #     "type": "string",
                                #     "analyzer": "lucene.keyword" # Index URL for potential filtering
                                # }
                            }
                        }
                        # Synonyms can be added here if needed
                        # "synonyms": [
                        #    { "name": "mySynonyms", "source": { "collection": "synonym_collection_name" } }
                        # ]
                    }
                    search_index_model = {"name": index_name, "definition": index_definition}

                    print(f"Attempting to create vector search index '{index_name}'...")
                    try:
                        collection.create_search_index(model=search_index_model)
                        # Status reflects whether it was newly created or recreated after dropping
                        index_status = f"recreated (name: {index_name})" if index_dropped else f"created (name: {index_name})"
                        print(f"Index creation/recreation for '{index_name}' initiated successfully. "
                              "It may take some time for the index to become fully queryable.")
                    except OperationFailure as create_err:
                        print(f"MongoDB Operation Failure creating index '{index_name}': {create_err.details}")
                        index_status = f"failed_create_operation: {create_err.codeName or create_err.details}"
                    except Exception as create_err:
                        print(f"Unexpected error creating index '{index_name}': {type(create_err).__name__} - {create_err}")
                        traceback.print_exc()
                        index_status = f"failed_create_unexpected: {str(create_err)}"

            except Exception as outer_idx_err:
                # Catch errors during the index management setup phase (e.g., list_search_indexes failure)
                print(f"Error during index management setup: {type(outer_idx_err).__name__} - {outer_idx_err}")
                traceback.print_exc()
                if index_status == "not_checked": # Ensure status reflects failure if it happened early
                    index_status = f"failed_management_setup: {str(outer_idx_err)}"

            end_time_index = time.time()
            print(f"Index management finished in {end_time_index - start_time_index:.2f}s. Status: {index_status}")

        elif first_error:
            index_status = "skipped_due_to_processing_error"
            print(f"Skipping index management due to critical errors during chunk processing. First error: {type(first_error).__name__}")
        else: # No data changed
            index_status = "skipped_no_data_change"
            print("Skipping index management as no data was inserted or removed in this run.")


        # --- 8. Return Response ---
        end_time_total = time.time()
        processing_time = round(end_time_total - start_time_total, 2)
        print(f"Total processing time: {processing_time}s")

        final_status_code = 200
        response_message = "PDF processed successfully."

        if errors:
            if inserted_count > 0:
                final_status_code = 207 # Multi-Status: Partial success
                response_message = f"PDF processed with {len(errors)} errors out of {processed_chunks_count} chunks attempted."
            else:
                final_status_code = 400 # Bad Request or potentially 500 if all errors were server-side
                response_message = f"PDF processing failed. Encountered {len(errors)} errors on {processed_chunks_count} chunks."
                # If a critical error caused stoppage, reflect that status code if appropriate
                if first_error and hasattr(first_error, 'status_code') and first_error.status_code:
                      # Use the status code of the first critical error if it's more specific
                      if first_error.status_code in [429, 500, 502, 503]:
                           final_status_code = first_error.status_code
                elif first_error and isinstance(first_error, (OperationFailure, ConnectionFailure)):
                    final_status_code = 503 # Service Unavailable for DB errors

        response_body = {
            "status": "success" if final_status_code == 200 else ("partial_success" if final_status_code == 207 else "failed"),
            "message": response_message,
            "chunks_processed": processed_chunks_count,
            "chunks_inserted": inserted_count,
            "duplicates_removed": duplicates_removed_count,
            "vector_index_name": index_name,
            "vector_index_status": index_status,
            "processing_time_seconds": processing_time,
        }
        if errors:
            # Include a sample of errors in the response for debugging
            response_body["errors_sample"] = errors[:10] # Limit number of errors shown

        return JSONResponse(content=response_body, status_code=final_status_code)

    except HTTPException as http_exc:
        # Catch HTTPExceptions raised explicitly (e.g., download error, pdf read error, auth error)
        print(f"Caught HTTPException: Status {http_exc.status_code}, Detail: {http_exc.detail}")
        raise http_exc # Re-raise to let FastAPI handle it
    except Exception as e:
        # Catch any unexpected errors during the setup or orchestration phase
        print(f"Unexpected top-level error during PDF processing: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        # Return a generic 500 error
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during processing: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    user: Dict = Depends(get_user) # Handles API key validation
):
    """
    Performs hybrid search (vector + full-text with RRF) using Atlas Search.
    Requires a vector search index named 'vector_index' with mappings for 'embedding' and 'text'.
    """
    vector_search_index_name = "vector_index" # Name of the Atlas Search index
    print(f"Received hybrid search request: query='{request.query[:50]}...', collection='{request.collection_name}', limit={request.limit}, user={user.get('supabase_user_id')}")

    try:
        # 1. Get User DB and Collection
        db = MongoDBManager.get_user_db(user)
        # Ensure collection exists before querying
        try:
            # list_collection_names can be slow on dbs with many collections.
            # A quicker check might be trying a cheap operation like estimated_document_count()
            # and catching the specific error if the collection doesn't exist, but this is simpler.
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
            # Run blocking OpenAI call in thread
            query_vector = await asyncio.to_thread(get_openai_embedding, request.query)
            end_time_embed = time.time()
            if not query_vector:
                 # Handle case where query is empty or embedding fails silently
                 raise HTTPException(status_code=400, detail="Could not generate embedding for the provided query.")
            print(f"Query embedding generated in {end_time_embed - start_time_embed:.2f}s.")
        except HTTPException as embed_exc:
             # Re-raise HTTPExceptions from get_openai_embedding (e.g., rate limit, auth error)
             raise embed_exc
        except Exception as e:
             # Catch unexpected errors during embedding
             print(f"Unexpected error during query embedding: {type(e).__name__} - {e}")
             traceback.print_exc()
             raise HTTPException(status_code=500, detail="Failed to generate query embedding due to an unexpected error.")

        # 3. Construct Atlas Search Aggregation Pipeline (Hybrid with RRF)
        # numCandidates should be >= limit, higher values improve recall at cost of performance.
        # Rule of thumb: 10-20x limit, but capped by num_candidates parameter.
        num_candidates = max(request.limit * 10, min(request.num_candidates, 1000))
        print(f"Using numCandidates: {num_candidates}")

        # RRF constant k, MongoDB recommends k=60. Adjust based on experimentation.
        rrf_k = 60

        # Define the aggregation pipeline
        pipeline = [
            # Stage 1: Vector Search
            {
                "$vectorSearch": {
                    "index": vector_search_index_name,
                    "path": "embedding",       # Field containing the vectors
                    "queryVector": query_vector, # The generated embedding for the query
                    "numCandidates": num_candidates, # Number of candidates to consider
                    "limit": num_candidates,        # Return top N candidates for RRF ranking later
                }
            },
            # Stage 2: Add vector search rank (Note: $vectorSearch score is distance, lower is better, but rank matters for RRF)
            # Group all results to calculate rank based on order returned by $vectorSearch
            {
                "$group": {
                    "_id": None, # Group all documents together
                    # Store original document and its vector search score
                    "docs": {"$push": {"doc": "$$ROOT", "vector_score": {"$meta": "vectorSearchScore"}}}
                }
            },
            # Unwind the documents and add the rank (index in the array + 1)
            {"$unwind": {"path": "$docs", "includeArrayIndex": "vector_rank_temp"}}, # 0-based index
            # Replace the root with the original document, preserving rank
            {"$replaceRoot": {"newRoot": { "$mergeObjects": [ "$docs.doc", { "vector_rank": { "$add": [ "$vector_rank_temp", 1 ] } } ] } } },
            # Project only necessary fields plus the rank
            {
                "$project": {
                    "_id": 1, "text": 1, "metadata": 1, "vector_rank": 1, "embedding": 0 # Exclude embedding
                }
            },

            # Stage 3: Full-Text Search (using $unionWith for RRF)
            {
                "$unionWith": {
                    "coll": request.collection_name, # Search the same collection
                    "pipeline": [
                        {
                            "$search": {
                                "index": vector_search_index_name, # Use the same index (must have text mapping)
                                "text": {
                                    "query": request.query,
                                    "path": "text" # Field containing the text to search
                                    # Optional: Add fuzzy matching, path synonyms etc.
                                    # "fuzzy": { "maxEdits": 1, "prefixLength": 3 }
                                },
                                # Optional: Highlight matching terms in the text
                                # "highlight": { "path": "text" }
                            }
                        },
                        # Limit the number of text search results to consider for RRF
                        {"$limit": num_candidates},
                        # Stage 3b: Add text search rank
                        {
                            "$group": {
                                "_id": None,
                                "docs": {"$push": {"doc": "$$ROOT", "text_score": {"$meta": "searchScore"}}}
                            }
                        },
                        {"$unwind": {"path": "$docs", "includeArrayIndex": "text_rank_temp"}},
                        {"$replaceRoot": {"newRoot": { "$mergeObjects": [ "$docs.doc", { "text_rank": { "$add": [ "$text_rank_temp", 1 ] } } ] } } },
                         {
                            "$project": {
                                "_id": 1, "text": 1, "metadata": 1, "text_rank": 1, "embedding": 0 # Exclude embedding
                            }
                        }
                    ]
                }
            },

            # Stage 4: Reciprocal Rank Fusion (RRF) Calculation
            {
                # Group documents by their _id to combine results from vector and text search
                "$group": {
                    "_id": "$_id",
                    # Keep the fields from the first occurrence (they should be the same)
                    "text": {"$first": "$text"},
                    "metadata": {"$first": "$metadata"},
                    # Capture the rank from each search type if present (use $min as placeholder if grouped)
                    "vector_rank": {"$min": "$vector_rank"}, # Will be rank or null if not found by vector search
                    "text_rank": {"$min": "$text_rank"}      # Will be rank or null if not found by text search
                }
            },
            {
                # Calculate the RRF score
                "$addFields": {
                    "rrf_score": {
                        "$sum": [
                            # Add score component for vector rank (1 / (k + rank)) if rank exists, else 0
                            {"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$vector_rank"]}]}, 0]},
                            # Add score component for text rank (1 / (k + rank)) if rank exists, else 0
                            {"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$text_rank"]}]}, 0]}
                        ]
                    }
                }
            },

            # Stage 5: Final Sorting, Limiting, and Formatting
            {"$sort": {"rrf_score": -1}}, # Sort by the combined RRF score in descending order
            {"$limit": request.limit},      # Apply the final limit on the number of results
            {
                # Project the final output format
                "$project": {
                    "_id": 0, # Exclude MongoDB's ObjectId
                    "id": {"$toString": "$_id"}, # Convert ObjectId to string for the response
                    "text": 1,
                    "metadata": 1,
                    "score": "$rrf_score" # Use the calculated RRF score as the final score
                }
            }
        ]


        # 4. Execute Aggregation Pipeline
        print("Executing hybrid search pipeline...")
        start_time_search = time.time()
        # Run the aggregation query
        search_results = list(collection.aggregate(pipeline))
        end_time_search = time.time()
        print(f"Hybrid search completed in {end_time_search - start_time_search:.2f} seconds. Found {len(search_results)} results.")

        # 5. Return Formatted Results
        return SearchResponse(results=search_results)

    except OperationFailure as mongo_error:
        # Handle MongoDB errors (e.g., index not found, auth issues)
        print(f"MongoDB Operation Failure during search: {mongo_error.details}")
        error_detail = f"Database operation failed during search: {mongo_error.details}"
        status_code = 500 # Default to internal server error
        # Check if the error indicates the index wasn't found (error code might vary)
        # Example check (needs verification with actual error message/code):
        if "index not found" in str(mongo_error.details).lower() or \
           (hasattr(mongo_error, 'codeName') and mongo_error.codeName == 'IndexNotFound'): # Example code name
            status_code = 404 # Not Found is more specific
            error_detail = f"Search index '{vector_search_index_name}' not found or not ready in collection '{request.collection_name}'. Ensure it has been created via the /process endpoint and is active."
        elif hasattr(mongo_error, 'code') and mongo_error.code == 13: # Example: Authorization failure code
             status_code = 403
             error_detail = "Authorization failed for search operation. Check database permissions."

        raise HTTPException(status_code=status_code, detail=error_detail)

    except HTTPException as http_exc:
         # Re-raise exceptions from dependencies (get_user, embedding) or collection check
         raise http_exc

    except Exception as e:
        # Catch any other unexpected errors during pipeline construction or execution
        print(f"An unexpected error occurred during hybrid search: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during search: {e}")


@app.post("/vector-search", response_model=VectorSearchResponse)
async def vector_search_documents(
    request: VectorSearchRequest,
    user: Dict = Depends(get_user) # Handles API key validation
):
    """
    Performs vector-only search using Atlas $vectorSearch.
    Requires a vector search index named 'vector_index' with a mapping for 'embedding'.
    Supports optional metadata filtering within the $vectorSearch stage.
    """
    vector_search_index_name = "vector_index" # Name of the Atlas Search index
    print(f"Received vector search request: query='{request.query[:50]}...', collection='{request.collection_name}', limit={request.limit}, filter={request.filter}, user={user.get('supabase_user_id')}")

    try:
        # 1. Get User DB and Collection (Similar to hybrid search)
        db = MongoDBManager.get_user_db(user)
        try:
            if request.collection_name not in db.list_collection_names():
                 raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found in database '{db.name}'.")
        except OperationFailure as e:
             print(f"Database error checking collection existence: {e.details}")
             raise HTTPException(status_code=503, detail=f"Database error accessing collection list: {e.details}")

        collection = db[request.collection_name]
        print(f"Vector searching in collection '{request.collection_name}' of database '{db.name}'")

        # 2. Generate Query Embedding (Similar to hybrid search)
        try:
            print("Generating embedding for the vector search query...")
            start_time_embed = time.time()
            query_vector = await asyncio.to_thread(get_openai_embedding, request.query)
            end_time_embed = time.time()
            if not query_vector:
                 raise HTTPException(status_code=400, detail="Could not generate embedding for the provided query.")
            print(f"Query embedding generated in {end_time_embed - start_time_embed:.2f}s.")
        except HTTPException as embed_exc:
             raise embed_exc
        except Exception as e:
             print(f"Unexpected error during query embedding: {type(e).__name__} - {e}")
             traceback.print_exc()
             raise HTTPException(status_code=500, detail="Failed to generate query embedding due to an unexpected error.")

        # 3. Construct Atlas Vector Search Aggregation Pipeline
        print(f"Using numCandidates: {request.num_candidates}")
        vector_search_stage = {
            "$vectorSearch": {
                "index": vector_search_index_name,
                "path": "embedding",               # Field with the vectors
                "queryVector": query_vector,       # Query embedding
                "numCandidates": request.num_candidates, # Number of candidates to inspect
                "limit": request.limit             # Max results to return
                # --- Add filter if provided in the request ---
                # Filters are applied *before* the vector similarity search, making it efficient.
                # Ensure the 'filter' field in the request uses correct MongoDB query syntax
                # and refers to fields mapped in the index if using 'string' type fields for filtering.
                # Keyword or other exact match types are generally better for filtering.
                # Example filter request: {"filter": {"metadata.category": "news"}}
                # Example filter request: {"filter": {"metadata.year": {"$gte": 2022}}}
            }
        }
        # Add the filter to the $vectorSearch stage if it exists in the request
        if request.filter:
            print(f"Applying filter to vector search: {request.filter}")
            vector_search_stage["$vectorSearch"]["filter"] = request.filter

        pipeline = [
            vector_search_stage, # The $vectorSearch stage with optional filter
            {
                # Project the results into the desired format
                "$project": {
                    "_id": 0, # Exclude MongoDB ObjectId
                    "id": {"$toString": "$_id"}, # Convert ObjectId to string ID
                    "text": 1,                   # Include the text content
                    "metadata": 1,               # Include the metadata
                    # Retrieve the similarity score from the vector search
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
            # Optional: Add a $match stage *after* $vectorSearch if filtering needs to happen
            # *after* the initial vector similarity ranking (less common, less efficient for filtering).
            # Example: if request.filter: pipeline.append({"$match": request.filter})
        ]

        # 4. Execute Aggregation Pipeline
        print("Executing vector search pipeline...")
        start_time_search = time.time()
        search_results = list(collection.aggregate(pipeline))
        end_time_search = time.time()
        print(f"Vector search completed in {end_time_search - start_time_search:.2f} seconds. Found {len(search_results)} results.")

        # 5. Return Formatted Results
        return VectorSearchResponse(results=search_results)

    # Error handling similar to hybrid search
    except OperationFailure as mongo_error:
        print(f"MongoDB Operation Failure during vector search: {mongo_error.details}")
        error_detail = f"Database operation failed during vector search: {mongo_error.details}"
        status_code = 500
        if "index not found" in str(mongo_error.details).lower() or \
           (hasattr(mongo_error, 'codeName') and mongo_error.codeName == 'IndexNotFound'):
            status_code = 404
            error_detail = f"Search index '{vector_search_index_name}' not found or not ready in collection '{request.collection_name}'. Ensure it exists and is active."
        elif hasattr(mongo_error, 'code') and mongo_error.code == 13:
             status_code = 403
             error_detail = "Authorization failed for vector search operation."
        # Handle potential errors from the filter syntax if possible
        # Example: if 'failed to parse query' in str(mongo_error.details).lower(): status_code = 400; error_detail = f"Invalid filter syntax provided: {request.filter}"

        raise HTTPException(status_code=status_code, detail=error_detail)
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred during vector search: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during vector search: {e}")


# ★★★ 新しいエンドポイント: ユーザーのDB名とコレクションリストを取得 ★★★
@app.post("/user-collections", response_model=UserInfoResponse)
def get_user_collections(request: UserInfoRequest):
    """
    Retrieves the database name and list of collection names associated with a user,
    verified by their Supabase User ID and API key.
    """
    print(f"Received request for user collections: supabase_user_id='{request.supabase_user_id}'")

    # --- Input Validation ---
    # 1. Check if API key is provided (not null or empty string)
    if not request.api_key:
        print("API key is missing in the request.")
        # 400 Bad Request might be more appropriate than 404 Not Found for missing required input
        raise HTTPException(status_code=400, detail="API key is required.")
        # Original requirement: return {"message": "No database found"} with 404
        # return JSONResponse(content={"message": "No database found"}, status_code=404)

    # --- Database Availability Check ---
    if auth_db is None or mongo_client is None:
        print("Error: Database services (auth_db or mongo_client) are unavailable.")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # --- 2. Find user in auth_db by supabase_user_id ---
        print(f"Searching for user with supabase_user_id: {request.supabase_user_id} in auth_db...")
        user = auth_db.users.find_one({"supabase_user_id": request.supabase_user_id})

        # --- 3. Handle User Not Found ---
        if not user:
            print(f"User not found for supabase_user_id: {request.supabase_user_id}")
            raise HTTPException(status_code=404, detail="User not found for the provided Supabase User ID.")

        # --- 4. Verify API Key ---
        print("User found. Verifying API key...")
        # Use secure comparison (secrets.compare_digest) to prevent timing attacks
        if not secrets.compare_digest(user.get("api_key", ""), request.api_key):
            print(f"API key mismatch for user {request.supabase_user_id}.")
            # Original requirement: return {"message": "api_key could be wrong..."} with 403
            raise HTTPException(
                status_code=403, # Forbidden - user exists, but key is wrong
                detail="API key could be wrong. Please check API key again."
            )
            # return JSONResponse(content={"message": "api_key could be wrong. Please check api_key again."}, status_code=403)

        # --- 5. API Key Verified - Get DB Name ---
        db_name = user.get("db_name")
        if not db_name:
            # This indicates an inconsistency in the auth_db user record
            print(f"Error: User {request.supabase_user_id} record in auth_db is missing 'db_name'.")
            raise HTTPException(status_code=500, detail="Internal Server Error: User data is incomplete.")

        print(f"API key verified. Accessing user database: {db_name}")

        # --- 6. Get Collections from User's Database ---
        try:
            user_db = mongo_client[db_name]
            # list_collection_names retrieves only user-created collections (excluding system.*)
            collection_names = user_db.list_collection_names()
            print(f"Found collections in {db_name}: {collection_names}")

            # --- 7. Return Success Response ---
            return UserInfoResponse(db_name=db_name, collections=collection_names)

        except OperationFailure as op_fail:
            print(f"MongoDB Operation Failure accessing user DB '{db_name}' or listing collections: {op_fail.details}")
            # Could be auth issue on the user DB, or other DB problem
            raise HTTPException(status_code=503, detail=f"Database operation failed accessing user data: {op_fail.details}")
        except ConnectionFailure as conn_fail:
            print(f"MongoDB Connection Failure accessing user DB '{db_name}': {conn_fail}")
            raise HTTPException(status_code=503, detail=f"Could not connect to user database: {conn_fail}")
        except Exception as db_err:
            # Catch unexpected errors during user DB operations
            print(f"Unexpected error accessing user DB '{db_name}': {type(db_err).__name__} - {db_err}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred accessing user database: {db_err}")

    except OperationFailure as auth_op_fail:
        # Catch errors during the initial find_one in auth_db
        print(f"MongoDB Operation Failure searching auth_db: {auth_op_fail.details}")
        raise HTTPException(status_code=503, detail=f"Database operation failed searching for user: {auth_op_fail.details}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred in /user-collections endpoint: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Avoid raising the raw exception; raise HTTPException
        if isinstance(e, HTTPException): # Re-raise if it's already an HTTPException
             raise e
        else:
             raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")


# --- Application startup ---
if __name__ == "__main__":
    # Final check for essential clients before starting server
    if mongo_client is None or auth_db is None or openai_client is None:
         print("FATAL: Required clients (MongoDB Auth DB, MongoDB Client, OpenAI Client) not initialized properly.")
         # Log details about which client failed if possible
         if mongo_client is None: print(" - MongoDB Client is None")
         if auth_db is None: print(" - MongoDB Auth DB is None")
         if openai_client is None: print(" - OpenAI Client is None")
         raise SystemExit("Server cannot start due to initialization failures.")

    print("Starting FastAPI server on host 0.0.0.0, port 8000...")
    # Consider adding reload=True for development, but remove for production
    uvicorn.run(app, host="0.0.0.0", port=8000) #, reload=True)