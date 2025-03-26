from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import PyPDF2
import io
import requests  # 追加

app = FastAPI()

@app.post("/extract-text-from-url/")
async def extract_text_from_pdf_url(pdf_url: str):
    try:
        # URLからPDFをダウンロード
        response = requests.get(pdf_url)
        response.raise_for_status()  # エラーチェック

        # PDFを読み込む
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
        
        # テキストを抽出
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        return JSONResponse(content={"extracted_text": text})
    
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"PDFのダウンロードに失敗しました: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDFの処理中にエラーが発生しました: {str(e)}")
