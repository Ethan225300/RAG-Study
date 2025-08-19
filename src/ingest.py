# src/ingest.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ★ 新增：pinecone & langchain-pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, EMBED_DIM
)
from utils import timer

# 如果你已经做了 loaders.py，也可以换成加载整个 data/ 目录
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "docs.txt")

def main():
    assert os.path.exists(DATA_PATH), f"未找到数据文件：{DATA_PATH}"
    text = open(DATA_PATH, "r", encoding="utf-8").read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.create_documents([text])
    print(f"切分后文档块数：{len(docs)}")

    embeddings = OpenAIEmbeddings()

    # ★ 连接 / 创建 Pinecone 索引（不存在则新建）
    pc = Pinecone(api_key=PINECONE_API_KEY)
    exist = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in exist:
        print(f"创建 Pinecone 索引：{PINECONE_INDEX}")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    with timer("Embedding + 云端建库(Pinecone)"):
        # 直接把分块文档 upsert 到云索引
        PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=PINECONE_INDEX,
        )

    print(f"已写入 Pinecone 索引：{PINECONE_INDEX}（云端）")

if __name__ == "__main__":
    main()
