from huggingface_hub import login
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 
import os

TOKEN = os.getenv("TOKEN")

login(token=TOKEN)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

print("Codigo Simoes")

# Model setup
model_name = "EleutherAI/gpt-j-6B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",  # Use Metal Performance Shaders backend
    torch_dtype=torch.float16  # Use half-precision for reduced memory usage
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # Required for MPS backend
    max_new_tokens=64,  # Generate fewer tokens
    num_beams=1,        # Use greedy decoding for faster results
    early_stopping=True,
    repetition_penalty=1.2,  # Slight penalty
)


# Define the prompt
prompt = """User: What is your favourite country?
Assistant: Well, I am quite fascinated with Peru.
User: What can you tell me about Peru?
Assistant:"""

# Generate response
outputs = pipe(prompt)
print(outputs[0]["generated_text"])