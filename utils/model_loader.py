from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

load_dotenv()

def load_model(model_name):
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_TOKEN)

    return tokenizer, model