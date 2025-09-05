from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline

# FastAPIアプリのインスタンスを作成
app = FastAPI()

# HTMLテンプレートのディレクトリを指定
templates = Jinja2Templates(directory=".")

# Hugging Faceのテキスト生成パイプラインをロード
# gpt2モデルは初回実行時にダウンロードされます
generator = pipeline("text-generation", model="gpt2")

class TextRequest(BaseModel):
    text: str

# ユーザーがサイトにアクセスしたときに、index.htmlを返すエンドポイント
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# AIがテキストを生成するためのAPIエンドポイント
@app.post("/generate/")
async def generate_text_api(text_request: TextRequest):
    # ユーザーからのテキストを受け取り、AIでテキストを生成
    generated_text = generator(text_request.text, max_length=100, num_return_sequences=1)
    
    # 生成されたテキストをJSON形式で返す
    return {"generated_text": generated_text[0]['generated_text']}
