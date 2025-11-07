# ghbm.py
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

thermo_cuda = load(
    name="thermo_cuda",
    sources=["thermo_cuda.cu"],
    extra_cuda_cflags=["-O3"]
)

class KolmogorovGHBM(nn.Module):
    def __init__(self, dim=768, ensemble_size=8, vte_depth=4):
        super().__init__()
        self.dim = dim
        self.ensemble_size = ensemble_size
        self.vte_rnn = nn.GRU(dim, dim, vte_depth, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.ensemble_size = ensemble_size

    def forward(self, x, stochastic=True):
        if not stochastic:
            return self.proj(x)

        B, L, D = x.shape
        h = x.clone()

        # Simulate ensemble via repeated perturbed passes
        ensemble = []
        for _ in range(self.ensemble_size):
            noise = torch.randn_like(h) * 0.05
            h_pert = h + noise

            # Pseudo-time evolution
            vte_input = h_pert.unsqueeze(1).repeat(1, 4, 1)  # 4 virtual steps
            vte_out, _ = self.vte_rnn(vte_input)
            h_vte = vte_out[:, -1, :]

            # Huxley-Boltzmann projection
            h_out = self.proj(h_vte)
            ensemble.append(h_out)

        ensemble = torch.stack(ensemble)  # [E, B, D]
        mean = ensemble.mean(0)
        std = ensemble.std(0) + 1e-6
        weights = torch.exp(-torch.abs(ensemble - mean) / std)
        attenuated = (weights * ensemble).sum(0) / weights.sum(0)
        return attenuated
