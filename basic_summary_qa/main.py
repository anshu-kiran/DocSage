import argparse
from utils.pdf_extractor import extract_text_from_pdf
from utils.model_interaction import summarize_text, answer_question
from utils.text_processor import chunk_text, extract_summary
from utils.misc import start_animation, stop_animation

def main():
    parser = argparse.ArgumentParser(description="Basic PDF Summarization Script")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--question", "--q", type=str, help="Question for Q&A based on the summary", default=None)
    args = parser.parse_args()

    event, thread = start_animation()
    text = extract_text_from_pdf(args.pdf_path)
    text_chunks = chunk_text(text)
    summaries = [summarize_text(chunk) for chunk in text_chunks]
    full_summary = extract_summary(" ".join(summaries))     
    stop_animation(event, thread)

    print("\n", full_summary)

    if args.question:
        answer = answer_question(full_summary, args.question)
        print(f"\nQ: {args.question}\nA: {answer}")

if __name__ == "__main__":
    main()
