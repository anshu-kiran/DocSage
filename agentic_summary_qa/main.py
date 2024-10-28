import argparse
from agent import PDFSummarizationQAAgent

def main():
    parser = argparse.ArgumentParser(description="PDF Summarization and Q&A Agent")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--question", "--q", type=str, help="Question for Q&A based on the summary", default=None)
    args = parser.parse_args()

    agent = PDFSummarizationQAAgent()

    summary = agent.process(args.pdf_path)
    print("Document Summary:\n", summary)

    if args.question:
        answer = agent.ask_question(args.question)
        print(f"\nQ: {args.question}\nA: {answer}")

if __name__ == "__main__":
    main()