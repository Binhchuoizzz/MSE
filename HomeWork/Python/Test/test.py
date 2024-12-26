import openai

# Thêm API Key của bạn vào đây
openai.api_key = ""

def chat_with_ai():
    print("Chat với AI (Nhập 'exit' để thoát):")
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        # Nhập tin nhắn từ người dùng
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Kết thúc phiên trò chuyện. Tạm biệt!")
            break

        # Thêm tin nhắn của người dùng vào danh sách
        messages.append({"role": "user", "content": user_input})

        # Gọi API OpenAI
        response = openai.ChatCompletion.create(  # Cú pháp đã được sửa
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Trích xuất câu trả lời từ AI
        ai_response = response['choices'][0]['message']['content'].strip()
        print(f"AI: {ai_response}")

        # Lưu câu trả lời của AI vào lịch sử
        messages.append({"role": "assistant", "content": ai_response})

# Chạy hàm
chat_with_ai()
