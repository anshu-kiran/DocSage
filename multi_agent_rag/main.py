import argparse
from pdf_processing_agent import PDFProcessingAgent
from embedding_agent import EmbeddingAgent
from qa_agent import QnAAgent

def main():
    parser = argparse.ArgumentParser(description="RAG-based PDF Summarization and Q&A with Multi-Agent Setup")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--question", "--q", type=str, help="Question for Q&A based on the summary", default=None)
    args = parser.parse_args()

    pdf_agent = PDFProcessingAgent()
    embedding_agent = EmbeddingAgent()

    text_chunks = pdf_agent.process_pdf(args.pdf_path)

    vector_store = embedding_agent.create_vector_store(text_chunks)

    qa_agent = QnAAgent(vector_store)

    summary = qa_agent.summarize_document()
    print("Document Summary:\n", summary)

    if args.question:
        answer = qa_agent.answer_question(args.question)
        print(f"\nQ: {args.question}\nA: {answer}")

if __name__ == "__main__":
    main()
