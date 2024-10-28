import pymupdf

def extract_text_from_pdf(pdf_path):
    document = pymupdf.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document[page_num]
        text += page.get_text("text")
    document.close()
    return text