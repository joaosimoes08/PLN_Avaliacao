from huggingface_hub import login
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

login(token=TOKEN)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
import os
import torch

# Model setup
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically assigns to MPS/CPU
    torch_dtype=torch.float32  # Use full precision for MPS compatibility
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    num_beams=4,
    early_stopping=True,
    repetition_penalty=1.4,
)

# Define the prompt
prompt = """User: What is your favourite country?
Assistant: Well, I am quite fascinated with Peru.
User: What can you tell me about Peru?
Assistant:"""

# Generate response
outputs = pipe(prompt)
print(outputs[0]["generated_text"])