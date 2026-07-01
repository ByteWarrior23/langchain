import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
print("Ready! Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ("exit", "quit"):
        break

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant reply
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    print(f"AI: {response}\n")
