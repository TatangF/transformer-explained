"""Example 2: train a tiny character-level Transformer, then generate text.

This shows the full loop from the slides:
    predict the next token -> append it -> predict again (autoregressive).

The default corpus is tiny, so the model will mostly *memorize* it. That is
expected: the goal is to see every mechanism working end to end on a CPU in
under a minute, not to build a useful language model.

Run:
    python examples/02_train_and_generate.py
    python examples/02_train_and_generate.py --file my_text.txt --steps 2000
    python examples/02_train_and_generate.py --plot   # saves attention heatmaps
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from tinytransformer import TinyGPT

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=os.path.join(os.path.dirname(__file__), "..", "data", "tiny_corpus.txt"))
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--max-len", type=int, default=64, help="context window N (in characters)")
parser.add_argument("--d-model", type=int, default=64)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--prompt", default="The transformer ")
parser.add_argument("--new-tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--plot", action="store_true", help="save attention heatmaps to outputs/")
args = parser.parse_args()

torch.manual_seed(args.seed)

# ---- Data: character-level tokenizer (each character is one token) ----------
text = open(args.file, encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text))
print(f"corpus: {len(text)} characters, vocabulary: {len(chars)} tokens")


def get_batch():
    """Random windows of N+1 tokens: the target is the input shifted by one."""
    ix = torch.randint(0, len(data) - args.max_len - 1, (args.batch_size,))
    x = torch.stack([data[i : i + args.max_len] for i in ix])
    y = torch.stack([data[i + 1 : i + args.max_len + 1] for i in ix])
    return x, y  # y[t] is the token that follows x[t]


# ---- Model ------------------------------------------------------------------
model = TinyGPT(len(chars), args.d_model, args.heads, args.layers, max_len=args.max_len)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}   context window: {args.max_len}")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

# ---- Training: minimize the cross-entropy of the next-token prediction -------
t0 = time.time()
for step in range(1, args.steps + 1):
    x, y = get_batch()
    logits = model(x)                                            # (B, N, vocab)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step == 1 or step % 100 == 0:
        print(f"step {step:5d}   loss {loss.item():.3f}   ({time.time() - t0:.0f}s)")
print(f"(a random guess would give a loss of about {torch.log(torch.tensor(float(len(chars)))).item():.2f})")

# ---- Generation: predict, append, repeat ------------------------------------
model.eval()
prompt_ids = torch.tensor([encode(args.prompt)])
out = model.generate(prompt_ids, args.new_tokens, temperature=args.temperature, top_k=10)
print("\n--- generated text " + "-" * 40)
print(decode(out[0].tolist()))
print("-" * 58)

# ---- Optional: look at what the heads attend to ------------------------------
if args.plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("outputs", exist_ok=True)
    sample = decode(data[: args.max_len // 2].tolist())
    ids = torch.tensor([encode(sample)])
    with torch.no_grad():
        _, all_w = model(ids, return_attention=True)
    w = all_w[-1][0]  # last layer, first batch item: (heads, N, N)
    fig, axes = plt.subplots(1, w.size(0), figsize=(3.2 * w.size(0), 3.4))
    for h, ax in enumerate(axes if w.size(0) > 1 else [axes]):
        ax.imshow(w[h].numpy(), cmap="Blues")
        ax.set_title(f"layer {len(all_w)} - head {h + 1}")
        ax.set_xlabel("attended position")
        ax.set_ylabel("current position")
    fig.tight_layout()
    fig.savefig("outputs/attention_heads.png", dpi=150)
    print("Saved outputs/attention_heads.png (lower-triangular: causal mask)")
