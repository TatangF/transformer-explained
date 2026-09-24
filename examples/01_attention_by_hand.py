"""Example 1: attention computed by hand on a toy sentence.

We hand-design tiny 3-feature vectors so that the result is easy to read:

    feature 0 = "is a noun"   feature 1 = "is animate"   feature 2 = "is a pronoun"

The pronoun "it" gets a query that says "I'm looking for an animate noun".
Nothing here is learned: it only illustrates *what* attention does.
A trained model discovers such patterns by itself (see example 02).

Run:  python examples/01_attention_by_hand.py [--plot]
"""

import argparse
import math

import torch

from tinytransformer import scaled_dot_product_attention

tokens = ["the", "animal", "crossed", "the", "street", "because", "it"]
#                       noun animate pronoun
features = {
    "the":     [0.0, 0.0, 0.0],
    "animal":  [1.0, 1.0, 0.0],
    "crossed": [0.0, 0.0, 0.0],
    "street":  [1.0, 0.0, 0.0],
    "because": [0.0, 0.0, 0.0],
    "it":      [0.0, 0.0, 1.0],
}
x = torch.tensor([features[t] for t in tokens])  # (N, 3): one vector per token

# Hand-made projection matrices (in a real model these are *learned*).
# A pronoun (feature 2) emits a query asking for "noun" and "animate".
W_q = torch.tensor([[0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 3.0, 0.0]])
W_k = torch.eye(3)  # keys simply advertise the token's own features
W_v = torch.eye(3)  # values carry the token's own features

Q, K, V = x @ W_q, x @ W_k, x @ W_v

out, weights = scaled_dot_product_attention(Q, K, V)  # no mask: the whole sentence is visible

print("Step 1 - Query / Key / Value for 'it':")
print(f"  Q[it] = {Q[-1].tolist()}   (what 'it' is looking for)\n")

scores = Q @ K.T / math.sqrt(K.shape[-1])
print("Step 2 - relevance scores of 'it' against every token (before softmax):")
for t, s in zip(tokens, scores[-1]):
    print(f"  {t:>8}  {s.item():5.2f}")

print("\nStep 3 - attention weights of 'it' (softmax, sums to 1):")
for t, w in zip(tokens, weights[-1]):
    print(f"  {t:>8}  {w.item():5.1%}  {'#' * int(w.item() * 40)}")

print("\nStep 4 - new vector for 'it' = weighted mix of the Values:")
print(f"  before: {x[-1].tolist()}   (only 'pronoun')")
print(f"  after : {[round(v, 2) for v in out[-1].tolist()]}   (now carries 'noun' and 'animate' from 'animal')")

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="save a heatmap to outputs/attention_by_hand.png")
if parser.parse_args().plot:
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(weights.numpy(), cmap="Blues")
    ax.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens)), tokens)
    ax.set_xlabel("attended token (key)")
    ax.set_ylabel("current token (query)")
    ax.set_title("Hand-made attention weights")
    fig.tight_layout()
    fig.savefig("outputs/attention_by_hand.png", dpi=150)
    print("\nSaved outputs/attention_by_hand.png")
