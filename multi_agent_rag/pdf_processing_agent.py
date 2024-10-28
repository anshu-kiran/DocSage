from utils.pdf_extractor import extract_text_from_pdf
from utils.text_processor import chunk_text

class PDFProcessingAgent:
    def __init__(self, max_chunk_length=200):
        self.max_chunk_length = max_chunk_length

    def process_pdf(self, pdf_path):
        text = extract_text_from_pdf(pdf_path)
        text_chunks = chunk_text(text, max_length=self.max_chunk_length)
        return text_chunks
