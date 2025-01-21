from huggingface_hub import login
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch

# Load the token from the .env file
load_dotenv()
token = os.getenv("TOKEN")
if not token:
    raise ValueError("The token was not found in the .env file!")
login(token=token)

# Model configuration
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically chooses GPU/CPU
    torch_dtype=torch.float16  # Use half-precision if available
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,  # Limit token generation
    temperature=0.7,    # Add randomness
    top_p=0.9,          # Focus on likely tokens
    repetition_penalty=2.0  # Penalize repetition
)

# Conversation loop
print("Welcome! Type your prompt or 'exit' to quit.")
while True:
    user_prompt = input("\nEnter your prompt: ")
    if user_prompt.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break
    
    full_prompt = f"User: {user_prompt}\nAssistant:"
    print("\nGenerating response...")
    outputs = pipe(full_prompt)
    print("Response generated:")
    print(outputs[0]["generated_text"])
