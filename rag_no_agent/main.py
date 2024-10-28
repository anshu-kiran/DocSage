import argparse
from utils.pdf_extractor import extract_text_from_pdf
from utils.text_processor import chunk_text
from rag_pipeline import setup_rag_pipeline

def main():
    parser = argparse.ArgumentParser(description="Basic PDF Summarization Script")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--question", "--q", type=str, help="Question for Q&A based on the summary", default=None)
    args = parser.parse_args()

    text = extract_text_from_pdf(args.pdf_path)

    text_chunks = chunk_text(text, 200)

    qa_chain = setup_rag_pipeline(text_chunks)

    response = qa_chain({"query": "Summarize the document", "max_new_tokens": 200})
    summary = response.get("result") if isinstance(response, dict) else response
    print("Document Summary:\n", summary)

    if args.question:
        answer_response = qa_chain({"query": args.question})
        answer = answer_response.get("result") if isinstance(answer_response, dict) else answer_response
        print(f"\nQ: {args.question}\nA: {answer}")

if __name__ == "__main__":
    main()
