from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TOKEN")

# Login to Hugging Face
login(token=TOKEN)

# Model setup
model_name = "gpt2"

# Load the model with MPS backend and reduced precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",  # Use Metal Performance Shaders backend
    torch_dtype=torch.float16  # Use half-precision for reduced memory usage
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the pipeline without specifying device
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,  # Generate fewer tokens
    num_beams=1,        # Use greedy decoding for faster results
    early_stopping=True,
    repetition_penalty=1.2,  # Slight penalty
)

print("Interactive Chat with GPT-2. Type 'exit' to quit.")
context = ""  # Holds the conversation context

while True:
    # User input
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Ending the chat. Goodbye!")
        break
    
    # Append user input to the context
    context += f"User: {user_input}\nAssistant:"

    # Generate response
    response = pipe(context)[0]["generated_text"]

    # Extract the assistant's response
    assistant_response = response[len(context):].strip()

    # Print the assistant's reply
    print(f"Assistant: {assistant_response}")

    # Update the context with the assistant's response
    context += f" {assistant_response}\n"