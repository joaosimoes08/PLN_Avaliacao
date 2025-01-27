from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

load_dotenv()
TOKEN = os.getenv("TOKEN")

login(token=TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", device_map="auto",load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",padding_side="left")

generation_config = GenerationConfig(
    num_beams=4,
    early_stopping=True,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.eos_token_id,
    max_new_tokens=900,
)

seed_sentence = "Step by step way on how to make an apple pie:"
model_inputs = tokenizer([seed_sentence], return_tensors="pt").to(device)
generated_ids = model.generate(**model_inputs,generation_config=generation_config)

generated_tokens = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]
print(generated_tokens)
