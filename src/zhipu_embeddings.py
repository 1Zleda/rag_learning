from langchain_core.embeddings import Embeddings
import requests

class ZhipuEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _embed(self, texts):
        # 调用 API，texts 是字符串列表
        data = {
            "model": "embedding-2",
            "input": texts
        }
        response = requests.post(self.url, headers=self.headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        # 返回向量列表，顺序与 texts 一致
        return [item["embedding"] for item in result["data"]]

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    emb = ZhipuEmbeddings(os.getenv("ZHIPU_API_KEY"))
    vec = emb.embed_query("测试一下")
    print("查询向量长度：", len(vec))