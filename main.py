from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import PyPDF2
import io
import os
from pydantic import BaseModel, HttpUrl
import httpx
import uvicorn
from typing import Optional

app = FastAPI()

# 環境変数から設定を読み込み
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 10 * 1024 * 1024))  # デフォルト10MB
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 30))  # デフォルト30秒

class PdfRequest(BaseModel):
    pdf_url: HttpUrl  # URLのバリデーションを含む

@app.post("/extract-text-from-url/")
async def extract_text_from_url(request: PdfRequest):
    """
    PDFからテキストを抽出するエンドポイント
    - pdf_url: 有効なPDFファイルの公開URL
    """
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            # ヘッダーのみ先に取得してファイルサイズを確認
            head_response = await client.head(str(request.pdf_url))
            head_response.raise_for_status()
            
            content_length = int(head_response.headers.get('content-length', 0))
            if content_length > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDFファイルが大きすぎます（最大 {MAX_FILE_SIZE//(1024*1024)}MB まで）"
                )
            
            # ファイルをストリームでダウンロード
            async with client.stream('GET', str(request.pdf_url)) as response:
                response.raise_for_status()
                
                # メモリに一気に読み込まず、チャンクで処理
                pdf_content = bytearray()
                async for chunk in response.aiter_bytes():
                    pdf_content.extend(chunk)
                    if len(pdf_content) > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="PDFファイルがストリーム処理中にサイズ制限を超えました"
                        )

        # PDFを解析
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            if not pdf_reader.pages:
                raise HTTPException(status_code=400, detail="PDFに有効なページがありません")
            
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
            return {"extracted_text": text.strip()}
            
        except PyPDF2.PdfReadError as e:
            raise HTTPException(status_code=400, detail=f"PDFの解析に失敗しました: {str(e)}")
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502 if e.response.status_code >= 500 else 400,
            detail=f"PDFのダウンロードに失敗しました（ステータス {e.response.status_code}）"
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="PDFのダウンロードがタイムアウトしました")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"予期せぬエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    workers = int(os.environ.get("UVICORN_WORKERS", 1))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=workers,
        timeout_keep_alive=60
    )
