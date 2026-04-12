def retrieve_strategies(query: str, vectorstore, k: int = 3) -> str:
    """Retrieve top-k relevant retention strategies from the FAISS index."""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])
