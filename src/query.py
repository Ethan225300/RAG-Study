import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

from config import VECTOR_DIR, TOP_K
from utils import timer

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    # 允许反序列化（本地开发 OK，生产环境请谨慎）
    return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)

def build_qa():
    llm = ChatOpenAI(model="gpt-4o-mini")
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # 简单拼接，先跑通流程
        return_source_documents=True
    )
    return qa

def ask(question: str):
    qa = build_qa()
    with timer("RAG 推理"):
        out = qa.invoke({"query": question})
    answer = out["result"]
    sources = out.get("source_documents", [])
    print("\n=== 回答 ===\n", answer)
    if sources:
        print("\n=== 命中文档块(Top-{} 条) ===".format(len(sources)))
        for i, d in enumerate(sources, 1):
            snippet = d.page_content[:160].replace("\n", " ")
            print(f"[{i}] {snippet}...")
    return answer

if __name__ == "__main__":
    import sys
    q = "RAG 是什么？" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    ask(q)
