import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os

logging.basicConfig(level=logging.DEBUG)

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    print("Model loaded.")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")

except Exception as e:
    logging.error(f"Error loading model: {e}")
    tokenizer = None
    model = None

def get_tokenizer():
    return tokenizer

def get_model():
    return model