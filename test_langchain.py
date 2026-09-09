from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载 PDF
loader = PyPDFLoader(r"D:\my_code\test.pdf")  
documents = loader.load()

# 2. 打印第一页的部分内容，确认加载成功
print("总共加载页数：", len(documents))
print("第一页前200个字：", documents[0].page_content[:200])

# 3. 切分文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 4. 打印切分结果
print("切分后的总块数：", len(chunks))
print("第一块内容：", chunks[0].page_content)
print("第二块内容：", chunks[1].page_content)