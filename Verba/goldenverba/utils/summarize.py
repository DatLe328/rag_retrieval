import requests
import os
import json
import re

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
def extract_keywords_ollama(text: str) -> list[str]:
    """
    Goi Ollama de trich xuat keywords tu van ban.
    """
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://10.1.1.237:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

        system_prompt = "You are a keyword extraction assistant. Your task is to extract the 10 most important keywords or key phrases from the user's text. You MUST return the result as a single, valid JSON array of strings and nothing else. For example: [\"keyword 1\", \"keyword 2\"]"
        user_prompt = text[:2000]

        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        resp = requests.post(f"{ollama_url}/api/chat", json=payload)
        resp.raise_for_status()
        
        response_content = resp.json()["message"]["content"].strip()

        # Model co the tra ve markdown ```json ... ```, can phai xu ly
        json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
        if not json_match:
            print("Keyword extraction failed: No JSON array found in the response.")
            return []
            
        json_string = json_match.group(0)
        keywords = json.loads(json_string)

        if isinstance(keywords, list) and all(isinstance(item, str) for item in keywords):
            print(f"Keywords extracted: {keywords}")
            return keywords
        else:
            print("Keyword extraction failed: Response was not a list of strings.")
            return []

    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        return []
