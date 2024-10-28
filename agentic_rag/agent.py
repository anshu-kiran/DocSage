import os
from utils.pdf_extractor import extract_text_from_pdf
from utils.text_processor import chunk_text
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

class RAGAgent:
    def __init__(self):
        self.model = HuggingFaceEndpoint(
            repo_id="EleutherAI/gpt-neo-125M",
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            temperature=0.7
        )
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.qa_chain = None

    def process_pdf(self, pdf_path):
        text = extract_text_from_pdf(pdf_path)
        
        text_chunks = chunk_text(text, 200)
        
        self.vector_store = FAISS.from_texts(text_chunks, embedding=self.embedding_model)
        
        self.qa_chain = RetrievalQA.from_llm(
            llm=self.model, 
            retriever=self.vector_store.as_retriever()
        )
    
    def summarize_document(self):
        response = self.qa_chain({"query": "Summarize the document", "max_new_tokens": 200})
        return response.get("result") if isinstance(response, dict) else response

    def answer_question(self, question):
        response = self.qa_chain({"query": question})
        return response.get("result") if isinstance(response, dict) else response
