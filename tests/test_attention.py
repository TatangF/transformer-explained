import torch
import torch.nn.functional as F

from tinytransformer import TinyGPT, causal_mask, scaled_dot_product_attention


def test_shapes_and_rows_sum_to_one():
    q = k = v = torch.randn(2, 4, 6, 8)  # (batch, heads, N, d_k)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 6, 8)
    assert w.shape == (2, 4, 6, 6)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 6), atol=1e-5)


def test_matches_pytorch_reference():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 5, 16) for _ in range(3))
    ours, _ = scaled_dot_product_attention(q, k, v, causal_mask(5))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_causal_mask_hides_the_future():
    q = k = v = torch.randn(1, 5, 8)
    _, w = scaled_dot_product_attention(q, k, v, causal_mask(5))
    assert torch.all(torch.triu(w, diagonal=1) == 0)


def test_changing_a_future_token_does_not_change_the_past():
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=20, d_model=16, n_heads=2, n_layers=2, max_len=8).eval()
    a = torch.randint(0, 20, (1, 6))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 20  # change only the last token
    la, lb = model(a), model(b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)  # earlier positions unchanged


def test_generate_respects_context_window():
    model = TinyGPT(vocab_size=10, d_model=16, n_heads=2, n_layers=1, max_len=4).eval()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=12)
    assert out.shape == (1, 13)  # can generate far beyond max_len, one window at a time
