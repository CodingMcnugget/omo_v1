import os
import asyncio
from dotenv import load_dotenv   # ← 这一行是 python-dotenv 提供的函数
from google import genai

# -----------------------
# 1) 在程序最开始处加载 .env 文件
# -----------------------
# load_dotenv() 会查找当前工作目录下的 .env 文件，并将其中的 KEY=VALUE 加入到 os.environ
load_dotenv()

# -----------------------
# 2) 从环境变量中读取 GEMINI_API_KEY
# -----------------------
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    raise RuntimeError("❌ 未检测到环境变量 GEMINI_API_KEY，请检查 .env 文件是否正确，或环境变量是否已导出。")

# -----------------------
# 3) 用读取到的 api_key 来初始化 client
# -----------------------
client = genai.Client(api_key=api_key)

model = "gemini-2.0-flash-live-001"
config = {"response_modalities": ["TEXT"]}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        print("Session started")

if __name__ == "__main__":
    asyncio.run(main())
