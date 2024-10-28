def chunk_text(text, max_length=1024):
    words = text.split()
    return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

def extract_summary(text):
    summary_start = text.find("Summary:")
    if summary_start != -1:
        return text.split("Summary:")[-1].rstrip()
    else:
        return "Error during generation"