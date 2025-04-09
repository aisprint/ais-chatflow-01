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
    index_name = "vector_index"; index_status = "not_checked"; first_error = None
    duplicates_removed_count = 0; inserted_count = 0; processed_chunks_count = 0; errors = []
    start_time_total = time.time()
    print(f"Process PDF for {user.get('supabase_user_id', 'N/A')}: {request.pdf_url}")
    try:
        # --- 1. Download PDF ---
        try:
            start_time_download = time.time()
            response = await asyncio.to_thread(requests.get, str(request.pdf_url), timeout=60); response.raise_for_status()
            end_time_download = time.time(); print(f"PDF downloaded ({end_time_download - start_time_download:.2f}s).")
            content_length = response.headers.get('Content-Length'); pdf_bytes = response.content; pdf_size = len(pdf_bytes)
            if pdf_size > MAX_FILE_SIZE: raise HTTPException(status_code=413, detail=f"PDF size ({pdf_size/(1024*1024):.2f}MB) > limit ({MAX_FILE_SIZE/(1024*1024)}MB).")
            pdf_content = io.BytesIO(pdf_bytes); size_source = f"header: {content_length}" if content_length else "checked post-download"
            print(f"PDF size: {pdf_size/(1024*1024):.2f} MB ({size_source}).")
        except requests.exceptions.Timeout: raise HTTPException(status_code=408, detail="PDF download timeout.")
        except requests.exceptions.SSLError as ssl_err: raise HTTPException(status_code=502, detail=f"SSL error: {ssl_err}.")
        except requests.exceptions.RequestException as req_error:
             status_code = 502 if isinstance(req_error, requests.exceptions.ConnectionError) else 400
             detail = f"PDF download fail: {req_error}" + (f" (Status: {req_error.response.status_code})" if hasattr(req_error, 'response') and req_error.response else "")
             raise HTTPException(status_code=status_code, detail=detail)
        # --- 2. Extract text ---
        try:
            start_time_extract = time.time()
            def extract_text_sync(pdf_stream):
                 reader = PyPDF2.PdfReader(pdf_stream);
                 if reader.is_encrypted: raise ValueError("Encrypted PDF unsupported.")
                 return "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
            text = await asyncio.to_thread(extract_text_sync, pdf_content); end_time_extract = time.time()
            if not text.strip(): return JSONResponse(content={"status": "success", "message": "No text.", "chunks_processed": 0, "chunks_inserted": 0, "duplicates_removed": 0, "vector_index_name": index_name, "vector_index_status": "skipped_no_text", "processing_time_seconds": round(time.time() - start_time_total, 2)}, status_code=200)
            print(f"Text extracted ({len(text)} chars, {end_time_extract - start_time_extract:.2f}s).")
        except PyPDF2.errors.PdfReadError as e: raise HTTPException(status_code=400, detail=f"Invalid PDF: {e}")
        except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
        except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Text extract fail: {e}")
        # --- 3. Split text ---
        start_time_split = time.time(); chunks = split_text_into_chunks(text); end_time_split = time.time()
        print(f"Split {len(chunks)} chunks ({end_time_split - start_time_split:.2f}s).")
        if not chunks: return JSONResponse(content={"status": "success", "message": "No chunks.", "chunks_processed": 0, "chunks_inserted": 0, "duplicates_removed": 0, "vector_index_name": index_name, "vector_index_status": "skipped_no_chunks", "processing_time_seconds": round(time.time() - start_time_total, 2)}, status_code=200)
        # --- 4. DB Setup ---
        db = MongoDBManager.get_user_db(user); collection = MongoDBManager.create_collection(db, request.collection_name, user["supabase_user_id"])
        # --- 5. Process Chunks ---
        start_time_chunks = time.time(); print(f"Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            if not chunk or chunk.isspace(): continue; processed_chunks_count += 1
            try:
                start_time_embed = time.time(); embedding = await asyncio.to_thread(get_openai_embedding, chunk); end_time_embed = time.time()
                if not embedding: errors.append({"chunk_index": i, "error": "Empty embedding", "status_code": 400}); continue
                doc = {"text": chunk, "embedding": embedding, "metadata": {**request.metadata, "chunk_index": i, "original_url": str(request.pdf_url), "processed_at": datetime.utcnow()}, "created_at": datetime.utcnow()}
                start_time_insert = time.time(); res = collection.insert_one(doc); end_time_insert = time.time()
                if res.inserted_id: inserted_count += 1; print(f"Chunk {i+1}/{len(chunks)} processed (embed: {end_time_embed-start_time_embed:.2f}s, insert: {end_time_insert-start_time_insert:.2f}s)")
                else: errors.append({"chunk_index": i, "error": "Insert fail", "status_code": 500})
            except HTTPException as e:
                errors.append({"chunk_index": i, "error": e.detail, "status_code": e.status_code})
                is_critical = e.status_code in [429, 500, 502, 503]
                if is_critical and first_error is None: first_error = e; print(f"Stop: HTTP {e.status_code}"); break
            except (OperationFailure, ConnectionFailure) as e:
                error_detail = getattr(e, 'details', str(e)); errors.append({"chunk_index": i, "error": f"DB error: {error_detail}", "status_code": 503})
                if first_error is None: first_error = e; print("Stop: DB error"); break
            except Exception as e:
                traceback.print_exc(); errors.append({"chunk_index": i, "error": f"Unexpected: {e}", "status_code": 500})
                if first_error is None: first_error = e; print("Stop: Unexpected error"); break
        end_time_chunks = time.time(); print(f"Chunk processing done ({end_time_chunks - start_time_chunks:.2f}s). Processed: {processed_chunks_count}, Inserted: {inserted_count}, Errors: {len(errors)}")
        # --- 6. Remove Duplicates ---
        if inserted_count > 0 and not first_error:
            start_time_dedup = time.time(); print(f"Checking duplicates for {request.pdf_url}...")
            try:
                pipeline = [{"$match": {"metadata.original_url": str(request.pdf_url)}},{"$group": {"_id": "$text", "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},{"$match": {"count": {"$gt": 1}}}]
                ids_to_delete = [oid for group in collection.aggregate(pipeline) for oid in group['ids'][1:]]
                if ids_to_delete: res = collection.delete_many({"_id": {"$in": ids_to_delete}}); duplicates_removed_count = res.deleted_count; print(f"Deleted {duplicates_removed_count} duplicates.")
                else: print("No duplicates.")
                print(f"Deduplication done ({time.time() - start_time_dedup:.2f}s).")
            except Exception as e: print(f"Deduplication error: {e}"); index_status = "duplicates_check_failed"
        elif first_error: index_status = "duplicates_skipped_error"; print("Skip duplicate removal (errors).")
        else: index_status = "duplicates_skipped_no_inserts"; print("Skip duplicate removal (no inserts).")
        # --- 7. Manage Vector Index ---
        data_changed = inserted_count > 0 or duplicates_removed_count > 0
        if data_changed and not first_error:
            attempt_creation, index_dropped = True, False; print(f"Data changed. Manage index '{index_name}'...")
            start_time_index = time.time()
            try:
                if list(collection.list_search_indexes(index_name)):
                    print(f"Index '{index_name}' exists. Dropping...");
                    try: collection.drop_search_index(index_name); wait = 20; print(f"Wait {wait}s..."); await asyncio.sleep(wait); index_dropped = True; print("Drop ok.")
                    except Exception as e: print(f"Index drop error: {e}"); index_status = f"failed_drop:{getattr(e,'codeName','')}"; attempt_creation = False
                if attempt_creation:
                    print(f"Creating index '{index_name}'..."); dims = 1536; index_def = {"mappings": {"dynamic": False, "fields": {"embedding": {"type": "knnVector", "dimensions": dims, "similarity": "cosine"}, "text": {"type": "string", "analyzer": "lucene.standard"}}}}
                    try: collection.create_search_index({"name": index_name, "definition": index_def}); index_status = f"{'re' if index_dropped else ''}created"; print(f"Index creation initiated.")
                    except Exception as e: print(f"Index create error: {e}"); index_status = f"failed_create:{getattr(e,'codeName','')}"
            except Exception as e: print(f"Index mgmt setup error: {e}"); 
            if index_status == "not_checked": index_status = f"failed_mgmt_setup"
            print(f"Index management done ({time.time() - start_time_index:.2f}s). Status: {index_status}")
        elif first_error: index_status = "skipped_processing_error"; print(f"Skip index management (error: {type(first_error).__name__})")
        else: index_status = "skipped_no_data_change"; print("Skip index management (no data change).")
        # --- 8. Return Response ---
        processing_time = round(time.time() - start_time_total, 2); final_status_code = 200; msg = "Processed successfully."
        if errors:
            if inserted_count > 0: final_status_code = 207; msg = f"Processed with {len(errors)} errors/{processed_chunks_count} chunks."
            else: final_status_code = 400; msg = f"Processing failed ({len(errors)} errors/{processed_chunks_count} chunks)."
            if first_error and hasattr(first_error, 'status_code') and first_error.status_code in [429, 500, 502, 503]: final_status_code = first_error.status_code
            elif first_error and isinstance(first_error, (OperationFailure, ConnectionFailure)): final_status_code = 503
        resp_body = {"status": "success" if final_status_code == 200 else ("partial_success" if final_status_code == 207 else "failed"), "message": msg, "chunks_processed": processed_chunks_count, "chunks_inserted": inserted_count, "duplicates_removed": duplicates_removed_count, "vector_index_name": index_name, "vector_index_status": index_status, "processing_time_seconds": processing_time, }
        if errors: resp_body["errors_sample"] = errors[:10]
        return JSONResponse(content=resp_body, status_code=final_status_code)
    except HTTPException as e: raise e
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Unexpected processing error: {e}")

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