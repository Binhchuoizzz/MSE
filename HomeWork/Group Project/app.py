from flask import Flask, render_template, request, jsonify
import aiohttp
import asyncio
import json

app = Flask(__name__)

# API URL của Hugging Face (thay bằng API của bạn)
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {
    "Authorization": "Bearer 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"
}

# Hàm gọi API Hugging Face để tóm tắt văn bản bất đồng bộ
async def fetch_summary(session, chunk):
    async with session.post(HF_API_URL, headers=HEADERS, json={"inputs": chunk}) as response:
        if response.status == 200:
            result = await response.json()
            return result[0]['summary_text']
        else:
            return "Error in summarization."

# Hàm tóm tắt văn bản bất đồng bộ
async def summarize_text_async(text):
    max_token_length = 1024  # Giới hạn token cho BART model, có thể thay đổi tùy theo mô hình
    text_chunks = [text[i:i + max_token_length] for i in range(0, len(text), max_token_length)]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_summary(session, chunk) for chunk in text_chunks]
        summaries = await asyncio.gather(*tasks)
    
    return " ".join(summaries)

# Hàm tạo mind map dưới dạng JSON từ văn bản
def generate_mindmap(text):
    sentences = text.split('. ')
    mindmap = {"nodes": []}
    
    for idx, sentence in enumerate(sentences):
        node = {
            "id": idx,
            "text": sentence[:100]  # Chỉ lấy 100 ký tự đầu của mỗi câu
        }
        mindmap["nodes"].append(node)
        
    return mindmap

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["text_input"]
        # Sử dụng async để tóm tắt văn bản
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        summary = loop.run_until_complete(summarize_text_async(input_text))
        mindmap = generate_mindmap(summary)
        return jsonify({
            "summary": summary,
            "mindmap": mindmap
        })
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
