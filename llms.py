from torch import nn
from transformer import GPT2TransformerBlock
from transformer import LayerNorm
import torch

""" GPT-2 Model """

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

class GPT2(nn.Module):

  def __init__(self,cfg):
    super().__init__() # Add this line
    self.tok_embed = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
    self.pos_embed = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    self.transformers_blocks = nn.ModuleList([GPT2TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
    self.layer_norm = LayerNorm(cfg["emb_dim"])
    self.lm_head = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)


  def forward(self,in_idx):
    _,seq_len = in_idx.shape

    tok_emb = self.tok_embed(in_idx) # Shape of (batch_size,seq_len,emb_dim)
    pos_emb = self.pos_embed(torch.arange(seq_len,device=in_idx.device)) # Shape of (seq_len,emb_dim)
    x = tok_emb + pos_emb

    x = self.drop_emb(x)
    for block in self.transformers_blocks: # Iterate through the ModuleList
      x = block(x)
    x = self.layer_norm(x)
    logits = self.lm_head(x) # Shape must be (batch_size,context_len,vocab_size)

    return logits