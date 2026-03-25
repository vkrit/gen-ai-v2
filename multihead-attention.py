import torch, torch.nn as nn, torch.nn.functional as F, math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.d_model = d_model
        # Fused QKV: one projection for all heads
        self.W_qkv   = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o     = nn.Linear(d_model, d_model,     bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        qkv = self.W_qkv(x)              # (B, T, 3*d_model)
        Q, K, V = qkv.chunk(3, dim=-1)   # each: (B, T, d_model)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        # Q, K, V → (B, n_heads, T, d_k)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask   = torch.triu(torch.ones(T, T, device=x.device),
                             diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))  # (B, H, T, T)
        out     = weights @ V                               # (B, H, T, d_k)

        # Merge heads → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)
        return (out, weights) if return_weights else out

mha = MultiHeadAttention(d_model=256, n_heads=8)
x   = torch.randn(2, 10, 256)
out = mha(x)
print(f"MHA output: {out.shape}")   # (2, 10, 256)
