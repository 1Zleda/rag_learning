import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from zhipu_embeddings import ZhipuEmbeddings  

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

# ================= 第1步：加载并切分 PDF =================
print("1. 正在加载并切分 PDF...")

loader = PyPDFLoader(r"D:\my_code\rag_learning\test.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"   切分完成，共 {len(chunks)} 块")

# ================= 第2步：创建 Embeddings 对象 =================
print("2. 正在初始化 Embeddings...")
embeddings = ZhipuEmbeddings(api_key)

# ================= 第3步：存入 Chroma =================
print("3. 正在存入向量数据库（这可能稍微有点慢，需要耐心等）...")
# 提示：用 Chroma.from_documents()
# 参数包括：documents=chunks, embedding=embeddings, persist_directory="chroma_db"
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=r"D:\my_code\rag_learning\chroma_db"
)
print("   存入成功！")

# ================= 第4步：检索 =================
print("4. 开始检索...")
query = "产品应用行业是什么？"  
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"--- 结果 {i+1} ---")
    print(doc.page_content[:200]) 