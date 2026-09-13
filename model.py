"""
If you want to see the full code without modules then go to the link:
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, num_dims, num_heads, p=0.1):
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.head_dims = num_dims // num_heads

        self.w_q = nn.Linear(num_dims, num_dims, bias=False)
        self.w_k = nn.Linear(num_dims, num_dims, bias=False)
        self.w_v = nn.Linear(num_dims, num_dims, bias=False)

        self.attn_drop = nn.Dropout(p)
        self.resid_drop = nn.Dropout(p)
        self.proj_out = nn.Linear(num_dims, num_dims)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = Q.view(B, T, self.num_heads, self.head_dims).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dims).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dims).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dims)
        masks = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(masks, float("-inf"))

        scores = F.softmax(scores, dim=-1)
        attn_scores = self.attn_drop(scores)
        
        out = attn_scores @ V
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        out = self.proj_out(out)
        out = self.resid_drop(out)

        return out


class MyMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.mlp_drop = nn.Dropout(p)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.mlp_drop(x)
        return x


class Transformer(nn.Module):
    def __init__(self,num_dims,num_heads,p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(num_dims,eps=1e-5)
        self.attn = CausalSelfAttention(num_dims,num_heads,p=0.1)
        self.ln2 = nn.LayerNorm(num_dims,eps=1e-5)
        self.mlp = MyMLP(in_features=num_dims,hidden_features=4*num_dims,out_features=num_dims)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x+self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, num_dims, num_heads, num_layers, p=0.1):
        super().__init__()
        self.seq_len = seq_len

        self.tok = nn.Embedding(vocab_size, num_dims)
        self.pos = nn.Embedding(seq_len, num_dims)
        self.blocks = nn.ModuleList(
            [Transformer(num_dims, num_heads, p=p) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(num_dims, eps=1e-5)
        self.lm_head = nn.Linear(num_dims, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok = self.tok(idx)
        pos = self.pos(positions)
        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, logits.size(-1))
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
        