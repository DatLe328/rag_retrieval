from agentscope.agent import ReActAgent
from agentscope.model import OllamaChatModel
from agentscope.formatter import OllamaChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, ToolResponse
import json, os, asyncio


# ---- HÀM HỖ TRỢ: đọc keyword từ từng file ----
def load_keywords_from_files(folder_path: str) -> dict[str, list[str]]:
    """
    Trả về dict {filename: [list keyword]}
    """
    data = {}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data[file] = [line.strip().lower() for line in f if line.strip()]
    return data


# ---- TOOL: Phân loại domain ----
def classify_domain(text: str) -> ToolResponse:
    """
    Kiểm tra input thuộc domain nào dựa trên file txt.
    """
    folder = "../data/domain_kw"
    all_keywords = load_keywords_from_files(folder)

    text_lower = text.lower()
    matched_file = None
    for file, kws in all_keywords.items():
        if any(k in text_lower for k in kws):
            matched_file = file
            break

    if matched_file:
        result = {"type": "indomain", "source_file": matched_file}
    else:
        result = {"type": "outdomain", "source_file": None}

    return ToolResponse(content=json.dumps(result))


# ---- KHỞI TẠO AGENT ----
async def create_classifier_agent():
    toolkit = Toolkit()
    toolkit.register_tool_function(classify_domain)

    agent = ReActAgent(
        name="ClassifierAgent",
        sys_prompt=(
    "You are a strict JSON-only classifier. "
    "You must not chat or generate explanations. "
    "Call classify_domain(text) to decide and RETURN ONLY a single JSON object "
    "in the format: {\"type\": \"indomain\"|\"outdomain\", \"source_file\": <file_name_or_null>}. "
    "Do not include any additional text, explanation, or reasoning."
),
        model=OllamaChatModel(
            model_name="qwen3:4b",
            host="http://10.1.1.237:11434",
            stream=True,
            enable_thinking=True,
        ),
        memory=InMemoryMemory(),
        formatter=OllamaChatFormatter(),
        toolkit=toolkit,
    )
    return agent
