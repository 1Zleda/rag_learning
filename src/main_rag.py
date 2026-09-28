from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from src.zhipu_embeddings import ZhipuEmbeddings
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

app = FastAPI()
embeddings = ZhipuEmbeddings(api_key)

# 1. 加载本地向量库
vectorstore = Chroma(
    persist_directory=r"D:\my_code\rag_learning\chroma_db", 
    embedding_function=embeddings
)

# 2. 定义大模型调用函数
def chat_with_llm(prompt: str) -> str:
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# 3. 定义请求体
class AskRequest(BaseModel):
    question: str

# 4. 实现 /ask 接口
@app.post("/ask")
def ask(req: AskRequest):
    # 4.1 检索相关文本块
    docs = vectorstore.similarity_search(req.question, k=3)
    
    # 4.2 拼接上下文
    context = "".join([doc.page_content for doc in docs])
    
    # 4.3 构造提示词
    prompt = f"""你是一个智能助手，请根据以下提供的上下文信息来回答问题。
如果你在上下文中找不到答案，请直接说“根据提供的文档，我无法回答这个问题”，不要编造。
    
上下文信息：
{context}
    
用户问题：{req.question}
"""
    # 4.4 调用大模型
    answer = chat_with_llm(prompt)
    
    # 4.5 返回结果
    return {"answer": answer}