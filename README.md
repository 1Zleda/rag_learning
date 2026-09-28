\# 基于 RAG 的知识库问答系统



\## 💡 项目简介

基于大模型和 RAG（检索增强生成）技术，实现一个能够根据上传 PDF 文档进行智能问答的后端服务。用户上传文档后，系统自动切分、向量化并存入向量数据库，提问时检索相关段落并调用大模型生成回答，有效解决大模型知识盲区和幻觉问题。



\## 🛠️ 技术栈

\- 语言：Python 3.10+

\- 框架：FastAPI、LangChain

\- 大模型：智谱 AI（GLM-4-Flash、Embedding-2）

\- 向量数据库：Chroma

\- 测试：pytest、requests



\## 🚀 核心功能

1\. PDF 文档加载与文本切分（chunk\_size=500, overlap=50）

2\. 调用 Embedding 接口生成向量并存入 Chroma

3\. 提供 `/ask` 接口，输入问题返回基于文档的答案

4\. 加入提示词约束，有效降低大模型幻觉



\## 🏗️ 系统架构

graph TD

&#x20;   A\[用户提问 /ask] --> B(FastAPI 后端接口)

&#x20;   B --> C{LangChain 检索链}

&#x20;   C -->|1. 问题向量化| D\[智谱 Embedding API]

&#x20;   D -->|2. 向量检索| E\[(Chroma 向量数据库)]

&#x20;   E -->|3. 返回相关文本块| C

&#x20;   C -->|4. 拼接上下文与Prompt| F\[智谱 GLM-4 大模型]

&#x20;   F -->|5. 生成回答| B

&#x20;   B --> G\[返回 JSON 答案]

&#x20;

&#x20;   style A fill:#f9f,stroke:#333,stroke-width:2px

&#x20;   style G fill:#f9f,stroke:#333,stroke-width:2px

&#x20;   style E fill:#bbf,stroke:#333,stroke-width:2px



\## 🎬 演示效果

!\[接口调用演示]（docs/demo.png）



\## 🚦 快速开始

1\. 克隆仓库

2\. 安装依赖：`pip install -r requirements.txt`

3\. 配置 `.env`，填入 `ZHIPU\\\_API\\\_KEY=你的Key`

4\. 运行：`uvicorn src.main\\\_rag:app --reload`

5\. 访问 `http://127.0.0.1:8000/docs` 测试 `/ask`



\## 📌 作者

蒋柏芝| 2027届计算机科学与技术

