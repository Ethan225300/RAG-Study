import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("API Key 10:", api_key[:10])  # 检查是否成功读取

client = OpenAI(api_key=api_key)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "this is a test"}]
)

print(resp.choices[0].message.content)
