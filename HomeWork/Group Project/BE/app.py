from flask import Flask, request, jsonify, render_template
import requests
from graphviz import Digraph
import os
from youtube_transcript_api import YouTubeTranscriptApi
import re

app = Flask(__name__)

# API Hugging Face
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Hàm lấy phụ đề từ URL YouTube
def get_youtube_transcript(url):
    video_id = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if not video_id:
        return "URL không hợp lệ"
    video_id = video_id.group(1)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        return "Không thể tìm thấy phụ đề cho video này."

# Hàm chia văn bản thành các đoạn nhỏ hơn để không vượt quá giới hạn token
def split_text_into_chunks(text, max_tokens=1024):
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Hàm tạo mindmap với các từ khóa chính và nội dung bên trong
def create_mindmap(text, output_file="static/mindmap"):
    dot = Digraph(
        comment='Mindmap',
        node_attr={'style': 'filled', 'fillcolor': 'lightyellow', 'fontname': 'Helvetica', 'fontsize': '12'}
    )
    dot.node('A', 'Tóm tắt Văn Bản', shape='ellipse', style='filled', fillcolor='lightblue', fontcolor='black')

    # Giả sử mỗi câu là một chủ đề
    sentences = text.split('.')
    main_topic_count = 0
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            main_topic_count += 1
            # Tạo node cho mỗi từ khóa chính (chủ đề)
            main_topic = f"MainTopic{main_topic_count}"
            dot.node(main_topic, sentence.strip(), shape='ellipse', style='filled', fillcolor='lightgreen', fontcolor='black')

            # Tạo các nhánh con cho mỗi từ khóa chính
            sub_topic_count = 0
            sub_sentences = sentence.split(',')  # Chia câu thành các ý nhỏ bằng dấu phẩy
            for sub_sentence in sub_sentences:
                if sub_sentence.strip():
                    sub_topic_count += 1
                    sub_topic = f"SubTopic{main_topic_count}_{sub_topic_count}"
                    dot.node(sub_topic, sub_sentence.strip(), shape='box', style='filled', fillcolor='lightyellow', fontcolor='black')
                    dot.edge(main_topic, sub_topic, color='blue', fontcolor='black', len='2.0')

    # Kết nối các chủ đề chính với nhau
    output_path = f"{output_file}.png"
    dot.render(output_file, format='png', cleanup=True)
    return output_path


# Route hiển thị giao diện chính
@app.route('/')
def home():
    return render_template('index.html')

# API tóm tắt văn bản từ URL YouTube và tạo mindmap
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Lấy văn bản phụ đề từ video YouTube
    text = get_youtube_transcript(url)
    if text == "Không thể tìm thấy phụ đề cho video này.":
        return jsonify({"error": text}), 400

    # Chia văn bản thành các đoạn nhỏ hơn nếu văn bản quá dài
    chunks = split_text_into_chunks(text)

    summary_text = ""
    for chunk in chunks:
        # Tóm tắt từng đoạn văn bản
        output = query({"inputs": chunk})
        
        # Kiểm tra xem phản hồi có hợp lệ và có chứa phần tử tóm tắt không
        if isinstance(output, list) and len(output) > 0:
            summary_text += output[0].get('summary_text', '') + " "
        else:
            summary_text += "Không thể tóm tắt đoạn này. "

    # Tạo mindmap
    mindmap_path = create_mindmap(summary_text.strip())

    return jsonify({
        "summary": summary_text.strip(),
        "mindmap_path": mindmap_path
    })

if __name__ == '__main__':
    app.run(debug=True)
