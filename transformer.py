from torch import nn
import torch
from attention import MultiHeadAttention


class LayerNorm(nn.Module):
  def __init__(self, d_in,eps=1e-5):
    super().__init__()
    self.eps = eps
    self.shift = nn.Parameter(torch.zeros(d_in))
    self.scale = nn.Parameter(torch.ones(d_in))


  def forward(self,inputs):
    # Make sure activations has 0 mean and 1 std
    mean = inputs.mean(dim=-1,keepdim=True)
    std = inputs.std(dim=-1,keepdim=True)
    return self.scale * ((inputs - mean) / (std + self.eps)) + self.shift

class GELU(nn.Module):
  def __init__(self):
    super().__init__()


  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
      torch.sqrt(torch.tensor(2.0 / torch.pi)) *
      (x + 0.044715 * torch.pow(x, 3))
    ))


class FeedForward(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"],4 * cfg["emb_dim"]),
        GELU(),
        nn.Linear(4 * cfg["emb_dim"],cfg["emb_dim"])
      )

  def forward(self,x):
    return self.layers(x)

class GPT2TransformerBlock(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.ln1 = LayerNorm(cfg["emb_dim"])
    self.mha = MultiHeadAttention(cfg["emb_dim"],cfg["n_heads"],cfg["context_length"],cfg["qkv_bias"],cfg["drop_rate"])
    self.dropout = nn.Dropout(cfg["drop_rate"])
    self.ln2 = LayerNorm(cfg["emb_dim"])
    self.ffn = FeedForward(cfg)

  def forward(self,x):
    x = x + self.dropout(self.mha(self.ln1(x)))
    x = x + self.dropout(self.ffn(self.ln2(x)))
    return x
  