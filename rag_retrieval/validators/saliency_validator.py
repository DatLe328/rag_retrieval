from guardrails import Guard
from guardrails.hub import SaliencyCheck
import builtins
import io
import os


def run_saliency_check(
    text: str,
    assets_dir: str = "../data/",
    model_name: str = "ollama/llama3.2:3b",
    api_base: str = "http://10.1.1.237:11434",
    threshold: float = 0.1,
    strict: bool = True
) -> tuple[bool, str]:
    """
    Kiểm tra độ trung thực (saliency) của câu trả lời so với dữ liệu nguồn.

    Args:
        text: Chuỗi cần kiểm tra (thường là generated_answer)
        assets_dir: Thư mục chứa dữ liệu nguồn (context)
        model_name: Tên model để kiểm tra (vd: 'ollama/llama3.2:3b')
        api_base: URL API Ollama (không kèm '/api/chat')
        threshold: Ngưỡng chấp nhận khác biệt
        strict: Nếu True -> raise lỗi khi không đạt

    Returns:
        (is_valid, message)
        - is_valid: bool, True nếu hợp lệ
        - message: thông báo chi tiết
    """
    # Ép mọi thao tác đọc file trong Guardrails dùng UTF-8
    def safe_open(file, mode='r', *args, **kwargs):
        return io.open(file, mode, encoding='utf-8', errors='ignore')

    old_open = builtins.open
    builtins.open = safe_open

    # Cấu hình endpoint Ollama cho LiteLLM/Guardrails
    os.environ["OLLAMA_API_BASE"] = api_base

    try:
        guard = Guard().use(
            SaliencyCheck,
            assets_dir,
            llm_callable=model_name,
            threshold=threshold,
            on_fail="exception" if strict else "noop",
        )
        guard.validate(text)
        return True, "Passed saliency check"
    except Exception as e:
        return False, f"Failed saliency check: {e}"
    finally:
        builtins.open = old_open  # khôi phục open gốc
