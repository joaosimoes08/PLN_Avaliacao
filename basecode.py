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

prompt = [
{"role": "user", "content": "What is your favourite country?"},
{"role": "assistant", "content": "Well, I am quite fascinated with Peru."},
{"role": "user", "content": "What can you tell me about Peru?"}
]

outputs = pipe(
prompt,
max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1]['content'])
