from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

def setup_rag_pipeline(text_chunks):
    model = HuggingFaceEndpoint(
        repo_id="EleutherAI/gpt-neo-125M",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
        temperature=0.7
    )

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)

    qa_chain = RetrievalQA.from_llm(llm=model, retriever=vector_store.as_retriever())
    return qa_chain
