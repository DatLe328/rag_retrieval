from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import OllamaChatModel
from agentscope.formatter import OllamaChatFormatter
from agentscope.memory import InMemoryMemory
from classifier_agent import create_classifier_agent
import asyncio


async def main():
    # --- Khởi tạo classifier ---
    classifier = await create_classifier_agent()

    # --- User mô phỏng ---
    user = UserAgent(name="user")

    print("Type your query (exit to quit):")
    msg = None
    while True:
        # 1️⃣ Nhận input
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break

        # 2️⃣ Gọi classifier để xác định loại input
        classify_res = await classifier(msg)
        ctype = classify_res.get_text_content().lower()

        # 3️⃣ In ra loại domain
        print(f"[Classifier] → {ctype}")


if __name__ == "__main__":
    asyncio.run(main())
