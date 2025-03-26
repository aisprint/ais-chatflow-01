from flask import Flask, request, jsonify
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import fitz  # PyMuPDF

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():
    return "Hello, World!"

@app.route("/extract_pdf", methods=["GET"])
def extract_pdf():
    """
    指定されたPDFのURLからテキストを抽出して返すAPIエンドポイント
    """
    pdf_url = request.args.get("pdf_url")
    
    if not pdf_url:
        return jsonify({"error": "PDF URL is required"}), 400
    
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # PDFの内容を読み込む
        pdf_bytes = BytesIO(response.content)
        pdf_reader = PdfReader(pdf_bytes)
        extracted_text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        
        if not extracted_text:
            return jsonify({"error": "No text could be extracted from the PDF."}), 400
        
        return jsonify({"pdf_text": extracted_text})
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching PDF: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/PyMuPDF", methods=["POST"])
def extract_text_from_pdf():
    """
    PyMuPDFを使ってPDFからテキストを抽出するエンドポイント
    """
    pdf_file = request.files.get("pdf_file")
    
    if not pdf_file:
        return jsonify({"error": "PDF file is required"}), 400

    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open("pdf", pdf_bytes)  # PyMuPDFでPDFを開く
        extracted_text = "\n".join([page.get_text() for page in doc])
        
        return jsonify({"extracted_text": extracted_text if extracted_text else "No text extracted"})
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/display-list/", methods=["GET"])
def display_list():
    """
    クエリパラメータで渡された整数のリストを表示するエンドポイント
    """
    try:
        # クエリパラメータで渡されたリスト（カンマ区切りの整数）
        my_list = request.args.get("my_list")
        
        if not my_list:
            return jsonify({"error": "my_list is required"}), 400
        
        # リストを整数に変換
        my_list = [int(item) for item in my_list.split(",")]
        
        # リストを表示
        print(my_list)
        
        return jsonify({"list": my_list})
    
    except ValueError:
        return jsonify({"error": "Invalid list format. Ensure all items are integers."}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
