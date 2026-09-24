"""Step 2: from the attention formula to a full (tiny) GPT-style Transformer.

Pipeline, matching the slides:

    token ids -> Embedding (+ position) -> [Attention -> Feed-forward] x L
              -> final projection -> softmax -> next-token probabilities
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import causal_mask, scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """Several attention "heads" running in parallel on smaller projections.

    Q, K and V are *learned linear projections* of each token's vector.
    Each head looks at the sequence through its own projection, then the head
    outputs are concatenated and mixed by a final linear layer.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, d_model) -> (B, heads, N, d_head)
        b, n, _ = x.shape
        return x.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, d_model = x.shape
        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        out, weights = scaled_dot_product_attention(q, k, v, mask)
        # (B, heads, N, d_head) -> (B, N, d_model)
        out = out.transpose(1, 2).contiguous().view(b, n, d_model)
        return self.w_o(out), weights  # weights: (B, heads, N, N)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Attention alone is order-blind (it treats the tokens as a set), so we add a
    position-dependent vector to each token embedding.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class FeedForward(nn.Module):
    """Position-wise MLP: refines each token's vector on its own."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One block = attention + feed-forward, each wrapped with LayerNorm and a
    residual connection (pre-norm variant)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        attn_out, weights = self.attn(self.ln1(x), mask)
        x = x + attn_out                 # residual connection
        x = x + self.ff(self.ln2(x))     # residual connection
        return x, weights


class TinyGPT(nn.Module):
    """A tiny decoder-only Transformer language model (next-token predictor)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        max_len: int = 64,
    ):
        super().__init__()
        self.max_len = max_len  # <- the context window N
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff or 4 * d_model) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor, return_attention: bool = False):
        """idx: (B, N) token ids  ->  logits: (B, N, vocab_size)."""
        n = idx.size(1)
        if n > self.max_len:
            raise ValueError(f"sequence length {n} exceeds the context window {self.max_len}")
        x = self.pos_enc(self.tok_emb(idx))
        mask = causal_mask(n, idx.device)
        all_weights = []
        for block in self.blocks:
            x, w = block(x, mask)
            all_weights.append(w)
        logits = self.head(self.ln_f(x))
        return (logits, all_weights) if return_attention else logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation: predict a token, append it, repeat."""
        for _ in range(max_new_tokens):
            # Context window: anything older than max_len tokens is simply dropped.
            window = idx[:, -self.max_len :]
            logits = self(window)[:, -1, :]                  # scores for the *next* token
            if temperature == 0:                             # greedy
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)            # probability distribution
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)           # feed it back in
        return idx
