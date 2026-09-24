"""Step 1: the attention formula, and nothing else.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Q, K and V have shape (..., N, d_k) where N is the number of tokens in the
context window. The leading "..." can be a batch dimension, a head dimension,
or both.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def causal_mask(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Lower-triangular boolean mask of shape (n, n).

    mask[i, j] is True when token i is allowed to look at token j (j <= i).
    This is what makes a decoder-only LLM *autoregressive*: a token can never
    see the future, otherwise predicting the next token would be trivial.
    """
    return torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.

    Args:
        q, k, v: tensors of shape (..., N, d_k).
        mask: optional boolean tensor broadcastable to (..., N, N).
              Positions where mask is False are hidden (weight 0).

    Returns:
        output:  (..., N, d_k)  each token's new, context-enriched vector.
        weights: (..., N, N)    row i = how much token i attends to each token.
    """
    d_k = q.shape[-1]

    # 1) Relevance score of every token against every other token: an N x N grid.
    #    This grid is why the context window is an *architectural* limit
    #    and why the cost grows as O(N^2).
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    # 2) Hide the tokens that must not be seen (e.g. the future).
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 3) Turn scores into weights that sum to 1 along each row.
    weights = F.softmax(scores, dim=-1)

    # 4) Blend the values according to those weights.
    return weights @ v, weights
