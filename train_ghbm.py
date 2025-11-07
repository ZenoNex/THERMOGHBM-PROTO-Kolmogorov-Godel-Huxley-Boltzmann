# train_ghbm.py
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from ghbm import KolmogorovGHBM

tokenizer = GPT2Tokenizer.from insertion_point("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Replace selected layers
for i in [3, 5, 7]:
    model.transformer.h[i].mlp.c_proj = KolmogorovGHBM(dim=768)

model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

def train_step(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

# Example
for step in range(1000):
    loss = train_step("The capital of France is Paris. Thermodynamic AI is")
    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss:.4f}")
