from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from src.zhipu_embeddings import ZhipuEmbeddings
import os
import json
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


# 2. 定义大模型流式调用函数（生成器）
def chat_with_llm(prompt: str):
    """
    调用智谱 API，开启 stream=True，逐行读取 SSE 数据，
    每收到一个 token 就 yield 出来。
    """
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    data = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": True,  # 关键：开启流式
    }

    # 使用 with 确保生成器被提前关闭（客户端断开）时连接也能释放
    with requests.post(url, headers=headers, json=data, stream=True, timeout=60) as response:
        response.raise_for_status()
        # 智谱返回的是 UTF-8 编码的 SSE，显式设置避免乱码
        response.encoding = "utf-8"

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            line = line.strip()

            # SSE 格式一般为 "data: {...}"，最后一条为 "data: [DONE]"
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
            else:
                data_str = line

            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                # 非 JSON 行（如心跳/空行）直接跳过
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield content


# 3. 定义请求体
class AskRequest(BaseModel):
    question: str


# 4. 实现 /ask 接口（返回流式响应）
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

    # 4.4 包装成生成器，交给 StreamingResponse
    def stream_generator():
        try:
            for token in chat_with_llm(prompt):
                yield token
        except Exception as e:
            # 流已经开始返回，无法再改状态码，这里把错误作为流内容返回
            yield f"\n[ERROR] {e}"

    # 4.5 返回流式响应
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 如果前面有 Nginx，避免它缓冲流
        },
    )
