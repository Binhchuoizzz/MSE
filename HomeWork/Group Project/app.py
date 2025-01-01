from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

# API URL của Hugging Face (thay bằng API của bạn)
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {
    "Authorization": "Bearer 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"
}

# Hàm gọi API Hugging Face để tóm tắt văn bản
def summarize_text(text):
    response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": text})
    if response.status_code == 200:
        summary = response.json()[0]['summary_text']
        return summary
    else:
        return "Error in summarization."

# Hàm tạo mind map dưới dạng JSON từ văn bản
def generate_mindmap(text):
    sentences = text.split('. ')
    mindmap = {"nodes": []}
    
    for idx, sentence in enumerate(sentences):
        node = {
            "id": idx,
            "text": sentence[:50]  # Chỉ lấy 50 ký tự đầu của mỗi câu
        }
        mindmap["nodes"].append(node)
        
        if idx > 0:
            if "edges" not in mindmap:
                mindmap["edges"] = []
            mindmap["edges"].append({"from": idx-1, "to": idx})
    
    return mindmap

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["text_input"]
        summary = summarize_text(input_text)
        mindmap = generate_mindmap(summary)
        return jsonify({
            "summary": summary,
            "mindmap": mindmap
        })
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
