import requests
import os

def summarize_text_ollama(text: str) -> str:
    """
    Goi Ollama local API de tom tat noi dung.
    """
    try:
        # Lay thong tin tu bien moi truong, su dung gia tri mac dinh tu thiet lap cua ban
        ollama_url = os.getenv("OLLAMA_URL", "http://10.1.1.237:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": "You are a summarizing assistant. Summarize the following content in about 20 words, don't use any introductory or concluding phrases."},
                {"role": "user", "content": text[:2000]}
            ],
            "stream": False # Tat streaming de nhan mot ket qua duy nhat
        }
        resp = requests.post(f"{ollama_url}/api/chat", json=payload)
        resp.raise_for_status() # Kiem tra loi HTTP (vi du: 404, 500)
        
        # Bay gio resp.json() se hoat dong vi chi co mot object JSON duy nhat
        return resp.json()["message"]["content"].strip()

    except Exception as e:
        print(f"Summarize failed: {e}")
        return ""

