from rag_retrieval.model.model_factory import get_embedder, get_chat_model

# --- CẤU HÌNH TRUNG TÂM ---
# Bạn chỉ cần thay đổi các giá trị ở đây để chuyển đổi model/nhà cung cấp
CONFIG = {
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text",
    
    "chat_provider": "ollama",
    "chat_model": "llama4",
}

def run_embedding_demo():
    print("\n--- DEMO EMBEDDING ---")
    try:
        # Lấy embedder từ factory
        embedder = get_embedder(
            provider=CONFIG["embedding_provider"],
            model_name=CONFIG["embedding_model"]
        )

        documents = [
            "Việt Nam là một đất nước xinh đẹp.",
            "Phở là một món ăn truyền thống nổi tiếng."
        ]

        # Logic chính không cần biết đây là Ollama hay OpenAI
        embeddings = embedder.get_embeddings(documents)
        
        for doc, emb in zip(documents, embeddings):
            if emb:
                print(f"\n'{doc}'")
                print(f" -> Vector dimension: {len(emb)}")
                print(f" -> First 5 values: {emb[:5]}")
            
    except (ValueError, ConnectionError) as e:
        print(f"Lỗi trong demo embedding: {e}")


def run_chat_demo():
    print("\n--- DEMO CHAT ---")
    try:
        # Lấy chat model từ factory
        chat_model = get_chat_model(
            provider=CONFIG["chat_provider"],
            model_name=CONFIG["chat_model"]
        )

        system_prompt = "Bạn là một trợ lý AI hữu ích, trả lời ngắn gọn và đi thẳng vào vấn đề."
        user_prompt = "Thủ đô của Pháp là gì?"

        # Logic chính không thay đổi khi bạn đổi nhà cung cấp
        response = chat_model.generate(user_prompt, system_prompt)

        print(f"\nUser: {user_prompt}")
        print(f"AI ({chat_model.model_name}): {response}")

    except (ValueError, ConnectionError) as e:
        print(f"Lỗi trong demo chat: {e}")


if __name__ == "__main__":
    run_embedding_demo()
    run_chat_demo()