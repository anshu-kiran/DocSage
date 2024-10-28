import torch
from utils.model_loader import load_model
from transformers import StoppingCriteria, StoppingCriteriaList

model_name = "meta-llama/Meta-Llama-3-8B" 
tokenizer, model = load_model(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class StopOnNewLine(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # Stop if the generated text contains another "Q:"
        return "\nQ:" in decoded_text

def summarize_text(chunk):
    prompt = f"Please summarize the content of the following passage:\n\n{chunk}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    summary_ids = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=512, 
        num_beams=4, 
        early_stopping=True, 
        pad_token_id=tokenizer.eos_token_id
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary  

def answer_question(summary, question):
    prompt = f"{summary}\n\nQ: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output_ids = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=256, 
        min_length=50,
        num_beams=2, 
        early_stopping=False, 
        pad_token_id=tokenizer.eos_token_id, 
        # stopping_criteria=StoppingCriteriaList([StopOnNewLine(tokenizer)])
        )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    answer = answer[len(prompt):].strip()
    return answer