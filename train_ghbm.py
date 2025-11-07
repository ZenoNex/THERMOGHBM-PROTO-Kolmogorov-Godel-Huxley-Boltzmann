"""
train_ghbm.py: Fine-tuning GPT-2 with Thermodynamic GHBM Replacements

Design Choices & Rationale:
- Fine-tuning pre-trained GPT2: Leverages existing linguistic priors, improving convergence and sample efficiency.
- Replacing selected MLP projection layers (c_proj): Integrates GHBM stochastic/thermodynamic behavior within critical generative pathways. Chosen layers ([3, 5, 7]) inject physical noise at multiple depths for robustness.
- KolmogorovGHBM Model: Implements ensemble, virtual time, and Boltzmann attenuation to diversify activations during training and inference (see ghbm.py for detailed rationale).
- AdamW optimizer: Standard for transformer-based models. Handles sparse updates and adapts well to noisy gradients.
- Loss function and training loop: Follows standard language modeling, simulating realistic generative tasks.
- On-the-fly text example: Encourages generative exploration and assesses model stability.
- Use of tokenizer.pad_token: GPT2's default behavior uses eos as pad, maintaining compatibility with Huggingface standards.

This script demonstrates how to inject thermodynamic principles into modern neural language models for research on physically-aware deep learning.

"""

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
