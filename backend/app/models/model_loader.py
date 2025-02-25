import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os

logging.basicConfig(level=logging.DEBUG)

# Change to a smaller model
model_name = "microsoft/phi-2"  # ~2.7GB model

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")
    
    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("Model loaded successfully!")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")

except Exception as e:
    logging.error(f"Error loading model: {e}")
    tokenizer = None
    model = None

def get_tokenizer():
    return tokenizer

def get_model():
    return model