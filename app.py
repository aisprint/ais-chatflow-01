from fastapi import FastAPI, HTTPException
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import fitz  # PyMuPDF
from typing import List

app = FastAPI()

@app.get("/")
def hello_world():
    return 'Hello, World!'

@app.get("/extract_pdf")
def extract_pdf(pdf_url: str):
    """
    指定されたPDFのURLからテキストを抽出して返すAPIエンドポイント
    """
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # PDFの内容を読み込む
        pdf_bytes = BytesIO(response.content)
        pdf_reader = PdfReader(pdf_bytes)
        extracted_text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")
        
        return {"pdf_text": extracted_text}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching PDF: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/PyMuPDF")
def extract_text_from_pdf(pdf_bytes: bytes):
    """
    PyMuPDFを使ってPDFからテキストを抽出するエンドポイント
    """
    doc = fitz.open("pdf", pdf_bytes)  # PyMuPDFでPDFを開く
    extracted_text = "\n".join([page.get_text() for page in doc])
    return extracted_text if extracted_text else "No text extracted"


@app.get("/display-list/")
def display_list(my_list: List[int]):
    print(my_list)
    return {"list": my_list}


# FastAPIアプリケーションを起動するためのstart command
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
