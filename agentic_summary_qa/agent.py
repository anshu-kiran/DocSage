from utils.pdf_extractor import extract_text_from_pdf
from utils.text_processor import chunk_text, extract_summary
from utils.model_interaction import summarize_text, answer_question

class PDFSummarizationQAAgent:
    def __init__(self):
        self.summary = None
    
    def process(self, pdf_path):
        text = extract_text_from_pdf(pdf_path)
        
        text_chunks = chunk_text(text)
        summaries = [summarize_text(chunk) for chunk in text_chunks]
        
        self.summary = extract_summary(" ".join(summaries)) 
        return self.summary
    
    def ask_question(self, question):
        if not self.summary:
            raise ValueError("Summary not generated yet. Run the process() method first.")
        return answer_question(self.summary, question)
