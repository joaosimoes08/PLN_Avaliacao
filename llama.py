from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, GenerationConfig, pipeline

load_dotenv()
TOKEN = os.getenv("TOKEN")

login(token=TOKEN)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type= "nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline("text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    pad_token_id = tokenizer.eos_token_id,
    eos_token_id=model.config.eos_token_id,
    num_beams=4,
    early_stopping=True,
    repetition_penalty=1.4
)

print("Interactive Chat with LLama. Type 'exit' to quit.")
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
