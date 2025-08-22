import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import deepwell as dw

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.ln = nn.LayerNorm(16)
        self.head = nn.Linear(16, 100)

    def forward(self, x):
        x = self.embed(x)
        x = self.ln(x)
        return self.head(x)


def test_simple_transformer_training():
    model = TinyTransformer()
    optimized = dw.optimize_for_blackwell(model, precision="mxfp8", seq_len=8, batch_size=1)
    optimizer = torch.optim.SGD(optimized.parameters(), lr=0.1)
    inputs = torch.randint(0, 100, (1, 8))
    before = optimized.embed.weight.clone()
    output = optimized(inputs)
    loss = output.mean()
    loss.backward()
    optimizer.step()

    assert not torch.allclose(before, optimized.embed.weight)
