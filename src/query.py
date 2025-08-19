# src/query.py
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore   # ★ 这里

from config import TOP_K, PINECONE_INDEX
from utils import timer

def build_qa():
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings()

    # ★ 连接到现有 Pinecone 索引
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa

def ask(question: str):
    qa = build_qa()
    with timer("RAG 推理 (Pinecone)"):
        out = qa.invoke({"query": question})
    answer = out["result"]
    sources = out.get("source_documents", [])
    print("\n=== 回答 ===\n", answer)
    if sources:
        print(f"\n=== 命中文档块(Top-{len(sources)} 条) ===")
        for i, d in enumerate(sources, 1):
            src = d.metadata.get("source", "unknown")
            print(f"[{i}] {src} :: {d.page_content[:160]}...")
    return answer

if __name__ == "__main__":
    import sys
    q = "RAG 的核心流程是什么？" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    ask(q)
