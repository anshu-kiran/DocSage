import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

class QnAAgent:
    def __init__(self, vector_store):
        self.model = HuggingFaceEndpoint(
            repo_id="EleutherAI/gpt-neo-125M",
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            temperature=0.7
        )
        self.qa_chain = RetrievalQA.from_llm(
            llm=self.model,
            retriever=vector_store.as_retriever()
        )

    def summarize_document(self):
        response = self.qa_chain({"query": "Summarize the document", "max_new_tokens": 200})
        return response.get("result") if isinstance(response, dict) else response

    def answer_question(self, question):
        response = self.qa_chain({"query": question})
        return response.get("result") if isinstance(response, dict) else response
