import argparse
from agent import RAGAgent

def main():
    parser = argparse.ArgumentParser(description="RAG-based PDF Summarization and Q&A")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--question", "--q", type=str, help="Question for Q&A based on the summary", default=None)
    args = parser.parse_args()

    agent = RAGAgent()

    agent.process_pdf(args.pdf_path)

    summary = agent.summarize_document()
    print("Document Summary:\n", summary)

    if args.question:
        answer = agent.answer_question(args.question)
        print(f"\nQ: {args.question}\nA: {answer}")

if __name__ == "__main__":
    main()
