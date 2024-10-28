from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class EmbeddingAgent:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None

    def create_vector_store(self, text_chunks):
        self.vector_store = FAISS.from_texts(text_chunks, embedding=self.embedding_model)
        return self.vector_store
