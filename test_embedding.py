import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

url = "https://open.bigmodel.cn/api/paas/v4/embeddings"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "embedding-2",   # 先用 embedding-2，如果报错再试 embedding-3
    "input": "今天天气怎么样"
}

response = requests.post(url, headers=headers, json=data, timeout=30)
print("状态码：", response.status_code)
result = response.json()
# 打印向量长度
vector = result["data"][0]["embedding"]
print("向量长度：", len(vector))
print("前5个值：", vector[:5])