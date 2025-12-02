# Deepseek-moe-16b
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# model_name = "deepseek-ai/deepseek-moe-16b-chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

# messages = [
#     {"role": "user", "content": "Who are you?"}
# ]
# input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
# outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

# result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
# print(result)



# Qwen/Qwen1.5-MoE-A2.7B-Chat

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

## Data download
# from datasets import load_dataset

# ds = load_dataset("vblagoje/cc_news")

 
# Use a pipeline as a high-level helper
# Load model directly
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

