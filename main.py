import os
import io
import secrets
from datetime import datetime
from typing import List, Optional, Dict, Any
import traceback # エラー詳細表示のため

import requests
import PyPDF2
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends, Body
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # ★ OpenAI APIキー
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL") # ★ 使用するモデル

# --- Validate required environment variables on startup ---
required_env_vars = ["MONGO_MASTER_URI", "ADMIN_API_KEY", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    print(f"FATAL: Missing required environment variables: {', '.join(missing_vars)}")
    raise SystemExit(f"Missing required environment variables: {', '.join(missing_vars)}")

# --- OpenAI Client Initialization ---
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    # 簡単なテストAPIコール (オプション、コストがかかる可能性あり)
    # openai_client.models.list()
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"FATAL: Failed to initialize OpenAI client: {e}")
    raise SystemExit(f"Failed to initialize OpenAI client: {e}")

# --- MongoDB Connection ---
mongo_client = None
auth_db = None
try:
    mongo_client = MongoClient(MONGO_MASTER_URI, serverSelectionTimeoutMS=5000)
    # The ismaster command is cheap and does not require auth.
    mongo_client.admin.command('ismaster')
    print("Successfully connected to MongoDB.")
    auth_db = mongo_client["auth_db"]
except ConnectionFailure as e:
    print(f"FATAL: Failed to connect to MongoDB: {e}")
    raise SystemExit(f"MongoDB connection failed: {e}")
except Exception as e: # その他の予期せぬ接続エラー
    print(f"FATAL: An unexpected error occurred during MongoDB connection: {e}")
    raise SystemExit(f"Unexpected MongoDB connection error: {e}")


# --- Data models ---
class UserRegister(BaseModel):
    supabase_user_id: str = Field(..., min_length=1)

class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$") # コレクション名の制限を追加

class ProcessRequest(BaseModel):
    pdf_url: HttpUrl
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$") # コレクション名の制限を追加
    metadata: Dict[str, str] = Field(default_factory=dict)

# --- Dependencies ---
def verify_admin(api_key: str = Header(..., alias="X-API-Key")):
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required")

def get_user(api_key: str = Body(..., description="User's API Key")):
    """
    Retrieves the user based on the API key provided in the request body.
    """
    if auth_db is None:
         raise HTTPException(status_code=503, detail="Database service unavailable")
    user = auth_db.users.find_one({"api_key": api_key})
    if not user:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return user

# --- MongoDB manager ---
class MongoDBManager:
    @staticmethod
    def create_user_db(supabase_user_id: str):
        # if not auth_db: # 修正前
        if auth_db is None: # ★ 修正後
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
            print(f"Inserted user ({supabase_user_id}) record into auth_db, db_name: {db_name}")
            return {"api_key": api_key, "db_name": db_name}
        except OperationFailure as e:
            print(f"MongoDB Operation Failure creating user record: {e.details}")
            raise HTTPException(status_code=500, detail=f"Database operation failed: {e}")
        except Exception as e:
            print(f"Error inserting user into auth_db: {type(e).__name__} - {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to create user record: {e}")

    @staticmethod
    def get_user_db(user: Dict):
        # if not mongo_client: # 修正前
        if mongo_client is None: # ★ 修正後
             raise HTTPException(status_code=503, detail="Database service unavailable")
        return mongo_client[user["db_name"]]

    @staticmethod
    def create_collection(db, name: str, user_id: str):
        try:
            # if not db: # 修正前
            if db is None: # ★ 修正後
                raise ConnectionError("Database object is not valid")

            # (以降のコードは変更なし)
            if name not in db.list_collection_names():
                 collection = db[name]
                 init_result = collection.insert_one({"__init__": True, "user_id": user_id, "timestamp": datetime.utcnow()})
                 delete_result = collection.delete_one({"_id": init_result.inserted_id})
                 print(f"Collection '{name}' created in database '{db.name}' (deleted init doc: {delete_result.deleted_count})")
            else:
                 print(f"Collection '{name}' already exists in database '{db.name}'.")
            return db[name]

        except OperationFailure as e:
            print(f"MongoDB Operation Failure accessing/creating collection '{name}': {e.details}")
            raise HTTPException(status_code=500, detail=f"Database operation failed for collection '{name}': {e}")
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
            last_valid_end_pos = current_pos + 1

        chunk_words = words[current_pos:last_valid_end_pos]
        chunks.append(" ".join(chunk_words))

        # Move current_pos for the next chunk, considering overlap
        # Find a suitable overlap start point (go back roughly `overlap` chars)
        overlap_start_index = last_valid_end_pos - 1
        overlap_char_count = 0
        while overlap_start_index > current_pos:
             overlap_char_count += len(words[overlap_start_index]) + 1
             if overlap_char_count >= overlap:
                  break
             overlap_start_index -= 1

        # Ensure we don't get stuck if overlap is too large or chunks too small
        current_pos = max(current_pos + 1, overlap_start_index)
        # Prevent infinite loop if stuck
        if current_pos >= last_valid_end_pos:
            current_pos = last_valid_end_pos

    return chunks

# --- ★ OpenAI Embedding Function ---
def get_openai_embedding(text: str) -> List[float]:
    """Generates embedding for the given text using OpenAI API."""
    if not text or text.isspace():
         print("Warning: Attempted to get embedding for empty or whitespace text.")
         # 空のテキストに対して空リストを返すかエラーにするか選択
         # OpenAI APIは空文字列でエラーになるため、ここでハンドリング
         return [] # または raise ValueError("Cannot embed empty text")

    try:
        # textをクリーニング (例: 過剰な空白の削除)
        cleaned_text = ' '.join(text.split())
        if not cleaned_text:
             return []

        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=cleaned_text # クリーニングされたテキストを使用
        )
        # response.data[0].embedding に埋め込みベクトル(リスト)が入っている
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
             # 通常は起こらないはずだが念のため
             print(f"Warning: OpenAI API returned no embedding data for text: {cleaned_text[:100]}...")
             raise HTTPException(status_code=500, detail="OpenAI API returned unexpected empty data.")

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI Service Error: {e}")
    except openai.AuthenticationError as e:
        print(f"OpenAI Authentication Error: {e}")
        # これは起動時に検知されるべきだが、キーが途中で無効になる可能性も考慮
        raise HTTPException(status_code=401, detail=f"OpenAI Authentication Error. Check API Key.")
    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Exceeded: {e}")
        raise HTTPException(status_code=429, detail=f"OpenAI Rate Limit Exceeded. Please wait and retry.")
    except openai.BadRequestError as e:
         # 入力が長すぎるなどのリクエスト自体の問題
         print(f"OpenAI Bad Request Error: {e}")
         # チャンク分割がうまくいっていない可能性がある
         raise HTTPException(status_code=400, detail=f"OpenAI Bad Request: Input may be too long or invalid. {e}")
    except Exception as e:
        print(f"An unexpected error occurred while getting OpenAI embedding: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Name of the collection to search within")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    num_candidates: int = Field(100, ge=10, le=1000, description="Number of candidates to consider for vector search (higher value increases recall but may impact performance)")

class SearchResultItem(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] # ★ 変更後: 値として任意の型を許可
    score: float

# Vector Search専用のリクエストモデル (num_candidatesの用途が明確になる)
class VectorSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    collection_name: str = Field("documents", min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$", description="Name of the collection to search within")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    num_candidates: int = Field(100, ge=10, le=1000, description="Number of candidates to consider for vector search (higher value increases recall but may impact performance)")
    # 必要であれば、フィルター条件を追加するためのフィールドも定義可能
    # filter: Optional[Dict[str, Any]] = Field(None, description="Optional filter criteria for metadata")

# Vector Search専用のレスポンスモデル (SearchResponseを流用しても良いが、区別のため作成)
class VectorSearchResponse(BaseModel):
    results: List[SearchResultItem] # SearchResultItem は流用

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

# --- API endpoints ---
@app.get("/health")
def health_check():
    # 簡単なヘルスチェックエンドポイント
    # DBやOpenAIクライアントの状態もチェックするとより良い
    return {"status": "ok"}

@app.get("/auth-db", dependencies=[Depends(verify_admin)])
def get_auth_db_contents():
    # if not auth_db: # 修正前 (エラー発生箇所)
    if auth_db is None: # ★ 修正後
         raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        users = list(auth_db.users.find({}, {"_id": 0, "api_key": 0}))
        return {"users": users}
    except Exception as e:
        print(f"Error reading from auth_db: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user data")


@app.post("/register", status_code=201)
def register_user(request: UserRegister):
    # if not auth_db: # 修正前
    if auth_db is None: # ★ 修正後
         raise HTTPException(status_code=503, detail="Database service unavailable")
    try:
        if auth_db.users.find_one({"supabase_user_id": request.supabase_user_id}):
            raise HTTPException(status_code=409, detail="User already exists") # 409 Conflict

        db_info = MongoDBManager.create_user_db(request.supabase_user_id)
        return {"api_key": db_info["api_key"]}
    except HTTPException as e:
         raise e
    except Exception as e:
        print(f"Unexpected error during user registration: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to register user")


@app.post("/collections", status_code=201)
def create_collection_endpoint(
    request: CollectionCreate,
    user: Dict = Depends(get_user)
):
    try:
        db = MongoDBManager.get_user_db(user)
        MongoDBManager.create_collection(db, request.name, user["supabase_user_id"])
        return {"status": "created", "collection_name": request.name}
    except HTTPException as e:
         raise e
    except Exception as e:
        print(f"Unexpected error creating collection: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to create collection")

@app.post("/process")
async def process_pdf( # asyncにするとバックグラウンドタスク等で有利だが、必須ではない
    request: ProcessRequest,
    user: Dict = Depends(get_user)
):
    # インデックス作成に関する変数を初期化
    index_name = "vector_index" # ★ 固定のインデックス名を使用
    index_status = "not_checked" # インデックスの状態を追跡
    first_error = None
    duplicates_removed_count = 0
    
    try:
        # 1. Download PDF
        print(f"Processing PDF from URL for user {user.get('supabase_user_id', 'N/A')}: {request.pdf_url}")
        try:
            response = await asyncio.to_thread(requests.get, str(request.pdf_url), timeout=60) # asyncで実行
            # response = requests.get(str(request.pdf_url), timeout=60) # 同期の場合
            response.raise_for_status()
        except requests.exceptions.Timeout:
             print(f"Timeout downloading PDF: {request.pdf_url}")
             raise HTTPException(status_code=408, detail="PDF download timed out.")
        except requests.exceptions.RequestException as req_error:
             print(f"PDF download failed: {str(req_error)}")
             status_code = 502 if isinstance(req_error, requests.exceptions.ConnectionError) else 400
             raise HTTPException(status_code=status_code, detail=f"PDF download failed: {str(req_error)}")

        # Check file size before reading content
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > MAX_FILE_SIZE:
             raise HTTPException(
                 status_code=413, # Payload Too Large
                 detail=f"PDF file size ({int(content_length) / (1024*1024):.2f} MB) exceeds the limit of {MAX_FILE_SIZE / (1024*1024)} MB"
             )
        pdf_content = io.BytesIO(response.content)
        print("PDF downloaded successfully.")

        # 2. Extract text
        try:
            # PyPDF2はIO負荷が高い可能性があるので、asyncにするメリットは薄いかもしれない
            pdf_reader = PyPDF2.PdfReader(pdf_content)
            if pdf_reader.is_encrypted:
                 # is_encrypted は限定的なチェック。より確実に判断するには他の方法も必要かも。
                 # パスワード解除はサポート外とする
                 raise HTTPException(status_code=400, detail="Encrypted PDF files are not supported.")
            text = ""
            for page in pdf_reader.pages:
                 extracted = page.extract_text()
                 if extracted:
                      text += extracted + "\n" # ページ間に改行を入れる

            if not text.strip():
                 print("Warning: No text could be extracted from the PDF.")
                 # テキストが空でもエラーにせず、処理結果を返す
                 return JSONResponse(
                      content={"status": "success", "message": "No text content found in PDF.", "chunks_processed": 0, "chunks_inserted": 0},
                      status_code=200 # 処理は成功したが、内容は空
                 )
            print(f"Text extracted, length: {len(text)} characters.")
        except PyPDF2.errors.PdfReadError as pdf_error:
            print(f"Error reading PDF structure: {pdf_error}")
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF file: {pdf_error}")
        except Exception as e: # PyPDF2の予期せぬエラー
            print(f"Unexpected error during PDF text extraction: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")

        # 3. Split text
        # OpenAIのada-002は8191トークン制限。chunk_sizeは文字数なので、
        # 安全マージンを見て1500文字程度に設定。より正確にはtiktokenを使う。
        chunks = split_text_into_chunks(text, chunk_size=1500, overlap=100)
        print(f"Text split into {len(chunks)} chunks.")
        if not chunks:
             # split_text_into_chunksが空リストを返すケース (元のテキストが非常に短いなど)
             print("No text chunks generated after splitting.")
             return JSONResponse(content={"status": "success", "message": "No processable text chunks generated.", "chunks_processed": 0, "chunks_inserted": 0})

        # 4. Database operations setup
        db = MongoDBManager.get_user_db(user)
        collection = MongoDBManager.create_collection(db, request.collection_name, user["supabase_user_id"])
        print(f"Using collection '{request.collection_name}' in database '{db.name}'.")

        # 5. Process chunks (consider running embeddings in parallel if needed)
        inserted_count = 0
        errors = []
        processed_chunks_count = 0
        first_error = None # 最初に発生した重大なエラーを記録

        for i, chunk in enumerate(chunks):
            if not chunk or chunk.isspace():
                 print(f"Skipping empty chunk {i}.")
                 continue
            processed_chunks_count += 1
            try:
                print(f"Generating embedding for chunk {i+1}/{len(chunks)}...")
                embedding = await asyncio.to_thread(get_openai_embedding, chunk) # ★ asyncで実行
                if not embedding:
                     print(f"Skipping chunk {i+1} due to empty embedding result.")
                     continue

                doc_to_insert = {
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {**request.metadata, "chunk_index": i, "original_url": str(request.pdf_url)},
                    "created_at": datetime.utcnow()
                }
                collection.insert_one(doc_to_insert)
                inserted_count += 1
            except (HTTPException, OperationFailure, Exception) as chunk_error:
                 error_detail = getattr(chunk_error, 'detail', str(chunk_error))
                 status_code = getattr(chunk_error, 'status_code', 500)
                 print(f"Error processing chunk {i+1}: Status {status_code} - {error_detail}")
                 errors.append({"chunk_index": i, "error": error_detail, "status_code": status_code})
                 # レート制限やDBエラーなど、継続困難なエラーを判定
                 is_critical = status_code == 429 or status_code >= 500 or isinstance(chunk_error, (OperationFailure, ConnectionFailure))
                 if is_critical and first_error is None:
                      first_error = chunk_error # 最初のエラーを保持
                      print(f"Stopping chunk processing due to critical error: {error_detail}")
                      break # ループ中断
                 # 軽微なエラー（例：Bad Request 400）は記録して継続（設定による）
                 # else: continue

        print(f"Chunk processing finished. Processed: {processed_chunks_count}, Inserted: {inserted_count}, Errors: {len(errors)}")

        # --- ★ 6. Remove Duplicate Chunks (based on text and URL) ---
        if inserted_count > 0 and not first_error: # データが挿入され、重大エラーがない場合のみ実行
            try:
                print(f"Checking for and removing duplicate chunks for URL: {request.pdf_url}...")
                pipeline = [
                    {"$match": {"metadata.original_url": str(request.pdf_url)}},
                    {"$group": {"_id": "$text", "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
                    {"$match": {"count": {"$gt": 1}}}
                ]
                duplicate_groups = list(collection.aggregate(pipeline))
                ids_to_delete = []
                if duplicate_groups:
                    print(f"Found {len(duplicate_groups)} text groups with duplicates.")
                    for group in duplicate_groups:
                        ids_to_delete.extend(group['ids'][1:])
                    if ids_to_delete:
                        print(f"Attempting to delete {len(ids_to_delete)} duplicate documents...")
                        delete_result = collection.delete_many({"_id": {"$in": ids_to_delete}})
                        duplicates_removed_count = delete_result.deleted_count
                        print(f"Successfully deleted {duplicates_removed_count} duplicate documents.")
                    else: print("No duplicate documents needed deletion.")
                else: print("No duplicate text content found for this URL.")
            except OperationFailure as agg_error: print(f"MongoDB Operation Failure during duplicate check/removal: {agg_error.details}")
            except Exception as agg_error: print(f"Unexpected error during duplicate check/removal: {type(agg_error).__name__} - {agg_error}"); traceback.print_exc()
        elif first_error: print("Skipping duplicate removal due to errors during chunk processing.")
        else: print("Skipping duplicate removal as no new data was inserted.")


        # --- ★ 7. Drop Existing Index (if found) and Create New Vector Search Index ---
        # このステップは重複削除の後に行う
        if (inserted_count > 0 or duplicates_removed_count > 0) and not first_error: # データ挿入or削除があり、重大エラーがない場合
            # (注意: duplicates_removed_count > 0 の条件は、既存データに対して重複削除のみ行い、
            #  インデックスを再作成したい場合に意味があります。挿入が必ず伴うなら inserted_count > 0 だけで良い)
            attempt_creation = True
            index_dropped = False
            try:
                print(f"Checking for existing vector search index '{index_name}'...")
                existing_indexes = list(collection.list_search_indexes())
                index_exists = any(idx['name'] == index_name for idx in existing_indexes)

                if index_exists:
                    print(f"Index '{index_name}' found. Attempting to drop it...")
                    try:
                        collection.drop_search_index(index_name)
                        print(f"Successfully initiated drop for index '{index_name}'. Waiting briefly...")
                        time.sleep(20) # 待機
                        index_dropped = True
                        print("Proceeding to create new index.")
                    except OperationFailure as drop_err: print(f"MongoDB Operation Failure dropping index '{index_name}': {drop_err.details}"); index_status = f"failed_drop_operation: {drop_err.details}"; attempt_creation = False
                    except Exception as drop_err: print(f"Unexpected error dropping index '{index_name}': {type(drop_err).__name__} - {drop_err}"); traceback.print_exc(); index_status = f"failed_drop_unexpected: {str(drop_err)}"; attempt_creation = False
                else: print(f"Index '{index_name}' not found. Will create a new one.")

                if attempt_creation:
                    print(f"Attempting to create vector search index '{index_name}' for hybrid search...") # ログメッセージ変更 (任意)
                    index_definition = {
                        "mappings": {
                            # dynamic を False に設定し、インデックス対象フィールドを明示的に指定します。
                            "dynamic": False,
                            "fields": {
                                # 1. Vectorフィールド (既存)
                                "embedding": {
                                    "type": "knnVector",     # ベクトル検索用タイプ
                                    "dimensions": 1536,      # OpenAI text-embedding-ada-002 の次元数
                                    "similarity": "cosine"   # 類似度計算方法
                                },
                                # 2. Text フィールド (全文検索用に新規追加)
                                "text": {
                                    "type": "string",          # テキスト型であることを指定
                                    "analyzer": "lucene.standard", # テキスト分析方法 (標準)
                                    # 多言語対応や特定の言語（例: 日本語）に最適化する場合はアナライザーを変更:
                                    # "analyzer": "lucene.kuromoji", # 日本語の場合
                                    # 必要に応じて他のオプションを追加:
                                    # "indexOptions": "positions" # フレーズ検索などの精度向上
                                }
                                # 3. Metadata フィールド (オプション: メタデータも検索対象にする場合)
                                # "metadata": {
                                #    "type": "document", # ネストされたオブジェクト用
                                #    "dynamic": True     # metadata内のフィールドは動的にマッピング
                                # }
                            }
                        }
                    }
                    search_index_model = {"name": index_name, "definition": index_definition}
                    try:
                        collection.create_search_index(model=search_index_model)
                        index_status = f"recreated (name: {index_name})" if index_dropped else f"created (name: {index_name})"
                        print(f"Index creation/recreation for '{index_name}' initiated. May take time to become queryable.")
                    except OperationFailure as create_err: print(f"MongoDB Operation Failure creating index '{index_name}': {create_err.details}"); index_status = f"failed_create_operation: {create_err.details}"
                    except Exception as create_err: print(f"Unexpected error creating index '{index_name}': {type(create_err).__name__} - {create_err}"); traceback.print_exc(); index_status = f"failed_create_unexpected: {str(create_err)}"

            except Exception as outer_idx_err: print(f"Error during index management setup: {type(outer_idx_err).__name__} - {outer_idx_err}"); traceback.print_exc(); index_status = f"failed_management_setup: {str(outer_idx_err)}"
        elif first_error: index_status = "skipped_due_to_processing_error"; print("Skipping index management due to errors during chunk processing.")
        else: index_status = "skipped_no_data_change"; print("Skipping index management as no data was inserted or removed.") # inserted/removed がない場合


        # --- 8. Return Response ---
        final_status_code = 200
        response_body = {
            "status": "success",
            "message": "PDF processed.",
            "chunks_processed": processed_chunks_count,
            "chunks_inserted": inserted_count,
            "duplicates_removed": duplicates_removed_count,
            "vector_index_status": index_status, # ★ 正しく設定されたステータス
        }
        # (エラー時のレスポンス調整)
        if errors:
            response_body["status"] = "partial_success" if inserted_count > 0 else "failed"
            response_body["message"] = f"Processed {processed_chunks_count} chunks with {len(errors)} errors."
            response_body["errors"] = errors[:10]
            final_status_code = 207 if inserted_count > 0 else 400
        if first_error:
             error_status = getattr(first_error, 'status_code', 500)
             if error_status >= 500: final_status_code = 503
             elif error_status == 429: final_status_code = 429
             elif final_status_code == 200 or final_status_code == 207: final_status_code = 500

        return JSONResponse(content=response_body, status_code=final_status_code)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error during PDF processing setup: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during processing: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    user: Dict = Depends(get_user)
):
    """
    Performs hybrid search (vector + full-text with RRF) on a specified collection.
    Requires a properly configured Atlas Search index named 'vector_index'
    with mappings for both 'embedding' (knnVector) and 'text' (string).
    """
    print(f"Received search request: query='{request.query}', collection='{request.collection_name}', limit={request.limit}")

    try:
        # 1. Get User DB and Collection
        db = MongoDBManager.get_user_db(user)
        # コレクションが存在するか確認 (create_collectionは作成しようとするので不適切)
        if request.collection_name not in db.list_collection_names():
             raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found.")
        collection = db[request.collection_name]
        print(f"Searching in collection '{request.collection_name}' of database '{db.name}'")

        # 2. Generate Query Embedding
        try:
            print("Generating embedding for the query...")
            # query_vector = await asyncio.to_thread(get_openai_embedding, request.query) # async
            query_vector = get_openai_embedding(request.query) # sync
            if not query_vector:
                 raise HTTPException(status_code=400, detail="Could not generate embedding for the query (empty query or OpenAI issue).")
            print("Query embedding generated.")
        except HTTPException as embed_exc:
             # get_openai_embedding 内で発生した HTTP エラーをそのまま re-raise
             raise embed_exc
        except Exception as e:
             print(f"Unexpected error during query embedding: {e}")
             traceback.print_exc()
             raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        # 3. Construct Atlas Search Aggregation Pipeline (using RRF)
        num_candidates = max(request.num_candidates, request.limit * 5) # 候補数はlimitの最低5倍程度推奨
        rrf_k = 60  # RRFの定数 (MongoDB推奨値)
        vector_search_index_name = "vector_index" # /processで作成したインデックス名

        # --- RRF パイプライン ---
        pipeline = [
            # --- Vector Search Stage ---
            {
                "$vectorSearch": {
                    "index": vector_search_index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates, # 探索候補数
                    "limit": num_candidates,        # vectorSearchから返す上限
                }
            },
            { # vector search の結果にランク付け
                "$group": {
                    "_id": None,
                    "docs": {"$push": {"doc": "$$ROOT", "vector_score": {"$meta": "vectorSearchScore"}}}
                }
            },
            {"$unwind": {"path": "$docs", "includeArrayIndex": "vector_rank"}},
            {"$replaceRoot": {"newRoot": "$docs.doc"}},
            {"$set": {"vector_rank": {"$add": ["$vector_rank", 1]}}}, # ランクを1から開始
            {
                "$project": { # 必要なフィールドとランクを保持
                    "_id": 1, "text": 1, "metadata": 1, "vector_rank": 1
                }
            },
            # --- Full-Text Search Stage (unionWith) ---
            {
                "$unionWith": {
                    "coll": request.collection_name,
                    "pipeline": [
                        { # 全文検索
                            "$search": {
                                "index": vector_search_index_name, # ★ 同じインデックスを使用 (textマッピングが必要)
                                "text": {
                                    "query": request.query,
                                    "path": "text"
                                },
                                # "highlight": { "path": "text" } # ハイライトが必要な場合
                            }
                        },
                        {"$limit": num_candidates}, # text searchの結果も制限
                        { # text search の結果にランク付け
                           "$group": {
                               "_id": None,
                               "docs": {"$push": {"doc": "$$ROOT", "text_score": {"$meta": "searchScore"}}}
                           }
                        },
                        {"$unwind": {"path": "$docs", "includeArrayIndex": "text_rank"}},
                        {"$replaceRoot": {"newRoot": "$docs.doc"}},
                        {"$set": {"text_rank": {"$add": ["$text_rank", 1]}}}, # ランクを1から開始
                        {
                            "$project": { # 必要なフィールドとランクを保持
                                "_id": 1, "text": 1, "metadata": 1, "text_rank": 1
                            }
                        }
                    ]
                }
            },
            # --- RRF Calculation Stage ---
            { # 同じドキュメントをグループ化し、ランク情報を集約
                "$group": {
                    "_id": "$_id",
                    "text": {"$first": "$text"},
                    "metadata": {"$first": "$metadata"},
                    "vector_rank": {"$min": "$vector_rank"}, # 存在すればランク、なければnull
                    "text_rank": {"$min": "$text_rank"}     # 存在すればランク、なければnull
                }
            },
            { # RRFスコアを計算
                "$addFields": {
                    "rrf_score": {
                        "$sum": [
                            # vector_rankが存在すればスコア寄与、なければ0
                            {"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$vector_rank"]}]}, 0]},
                            # text_rankが存在すればスコア寄与、なければ0
                            {"$ifNull": [{"$divide": [1, {"$add": [rrf_k, "$text_rank"]}]}, 0]}
                        ]
                    }
                }
            },
            # --- Final Sorting and Projection ---
            {"$sort": {"rrf_score": -1}}, # RRFスコアで降順ソート
            {"$limit": request.limit},      # 最終結果数を制限
            { # レスポンス形式に整形
                "$project": {
                    "_id": 0, # MongoDBのObjectIdは除外
                    "id": {"$toString": "$_id"}, # 文字列形式のID
                    "text": 1,
                    "metadata": 1,
                    "score": "$rrf_score"
                }
            }
        ]

        # 4. Execute Aggregation Pipeline
        print("Executing search pipeline...")
        start_time = time.time()
        search_results = list(collection.aggregate(pipeline))
        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.2f} seconds. Found {len(search_results)} results.")

        # 5. Return Formatted Results
        return SearchResponse(results=search_results)

    except OperationFailure as mongo_error:
        # MongoDBの操作エラー（インデックスが見つからない、権限不足など）
        print(f"MongoDB Operation Failure during search: {mongo_error.details}")
        error_detail = f"Database operation failed: {mongo_error.details}"
        status_code = 500
        # 特定のエラーコードに基づいてステータスを変更することも可能
        # 例: インデックスが見つからない場合 (エラーコードを確認する必要あり)
        # if "index not found" in str(mongo_error.details).lower():
        #     status_code = 404
        #     error_detail = f"Search index '{vector_search_index_name}' not found or not ready in collection '{request.collection_name}'. Please ensure it is created and active."
        raise HTTPException(status_code=status_code, detail=error_detail)

    except HTTPException as http_exc:
         # 既に発生したHTTPException（get_user, Embedding生成, コレクションNotFoundなど）
         raise http_exc

    except Exception as e:
        # その他の予期せぬエラー
        print(f"An unexpected error occurred during search: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during search: {e}")

@app.post("/vector-search", response_model=VectorSearchResponse)
async def vector_search_documents(
    request: VectorSearchRequest,
    user: Dict = Depends(get_user)
):
    """
    Performs vector search ONLY on a specified collection using $vectorSearch.
    Requires a properly configured Atlas Search index named 'vector_index'
    with a mapping for the 'embedding' field (knnVector).
    """
    print(f"Received vector search request: query='{request.query}', collection='{request.collection_name}', limit={request.limit}")
    vector_search_index_name = "vector_index" # /processで作成したインデックス名

    try:
        # 1. Get User DB and Collection (同上)
        db = MongoDBManager.get_user_db(user)
        if request.collection_name not in db.list_collection_names():
             raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found.")
        collection = db[request.collection_name]
        print(f"Vector searching in collection '{request.collection_name}' of database '{db.name}'")

        # 2. Generate Query Embedding (同上)
        try:
            print("Generating embedding for the query...")
            # query_vector = await asyncio.to_thread(get_openai_embedding, request.query) # async
            query_vector = get_openai_embedding(request.query) # sync
            if not query_vector:
                 raise HTTPException(status_code=400, detail="Could not generate embedding for the query.")
            print("Query embedding generated.")
        except HTTPException as embed_exc:
             raise embed_exc
        except Exception as e:
             print(f"Unexpected error during query embedding: {e}"); traceback.print_exc()
             raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        # 3. Construct Atlas Vector Search Aggregation Pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": vector_search_index_name,
                    "path": "embedding", # Embeddingベクトルが格納されているフィールド
                    "queryVector": query_vector,
                    "numCandidates": request.num_candidates, # 検索候補数
                    "limit": request.limit # 返す結果の上限数
                    # --- オプション: メタデータによるフィルタリング ---
                    # "$vectorSearch" ステージ内でフィルタリングを行うと効率が良い
                    # "filter": {
                    #     "metadata.category": "technology" # 例: categoryがtechnologyのものに絞る
                    # }
                    # filter フィールドをリクエストに追加し、それを使う場合:
                    # **({"filter": request.filter} if request.filter else {})
                }
            },
            {
                # 結果の整形
                "$project": {
                    "_id": 0, # MongoDBのObjectIdを除外
                    "id": {"$toString": "$_id"}, # 文字列IDに変換
                    "text": 1,
                    "metadata": 1, # メタデータを含める (Anyを許容するモデルが必要)
                    "score": {"$meta": "vectorSearchScore"} # ベクトル検索のスコアを取得
                }
            }
            # --- オプション: vectorSearch の後にメタデータでフィルタリングする場合 ---
            # (vectorSearch内でfilterする方が通常は効率的)
            # {
            #    "$match": {
            #        "metadata.some_field": "some_value"
            #    }
            # }
        ]

        # リクエストにフィルターが含まれていれば、パイプラインに追加（$vectorSearch内でやる方が良い）
        # if request.filter:
        #     pipeline.insert(1, {"$match": request.filter}) # $vectorSearch の後に追加

        # 4. Execute Aggregation Pipeline (同上)
        print("Executing vector search pipeline...")
        start_time = time.time()
        search_results = list(collection.aggregate(pipeline))
        end_time = time.time()
        print(f"Vector search completed in {end_time - start_time:.2f} seconds. Found {len(search_results)} results.")

        # 5. Return Formatted Results
        return VectorSearchResponse(results=search_results)

    except OperationFailure as mongo_error:
        # (エラーハンドリングは /search と同様)
        print(f"MongoDB Operation Failure during vector search: {mongo_error.details}")
        error_detail = f"Database operation failed: {mongo_error.details}"
        status_code = 500
        # if "index not found" in str(mongo_error.details).lower(): ... (インデックスエラー判定)
        raise HTTPException(status_code=status_code, detail=error_detail)
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred during vector search: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during vector search: {e}")


# --- Application startup ---
if __name__ == "__main__":
    # if not mongo_client or not auth_db or not openai_client: # 修正前
    if mongo_client is None or auth_db is None or openai_client is None: # ★ 修正後
         print("FATAL: Required clients (MongoDB, OpenAI) not initialized.")
         raise SystemExit("Client initialization failed.")

    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)