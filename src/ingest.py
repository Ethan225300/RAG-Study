import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_PATH, VECTOR_DIR
from utils import timer

def main():
    assert os.path.exists(DATA_PATH), f"未找到数据文件：{DATA_PATH}"
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.create_documents([text])
    print(f"切分后文档块数：{len(docs)}")

    with timer("Embedding + 建库"):
        embeddings = OpenAIEmbeddings()  # 使用 OpenAI embedding
        vs = FAISS.from_documents(docs, embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    vs.save_local(VECTOR_DIR)
    print(f"向量库已保存到：{os.path.abspath(VECTOR_DIR)}")

if __name__ == "__main__":
    main()
