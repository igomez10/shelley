"""Train an MLP language model (Bengio et al., 2003) on the prepared dataset.

The model predicts the next token from a fixed window of previous tokens:

    context tokens ──embedding lookup──► [E[t1] | E[t2] | E[t3]]   (concatenated)
                   ──hidden layer──────► relu(x · W1 + b1)
                   ──output layer──────► logits = h · W2 + b2      (one score per token)
                   ──softmax───────────► probability of every possible next token

Training minimizes cross-entropy, -ln p[actual next token], with backpropagation
(loss.backward()) to compute gradients and AdamW to update every parameter,
the embeddings included.

Run from the repo root after `make tokenizer` and `make prepare`:

    make pretrain
    uv run pretraining/train_mlp.py --steps 5000
"""

import argparse
import csv
import itertools
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA = Path("data")
TRAIN_TXT = DATA / "preparation" / "train.txt"
VAL_TXT = DATA / "preparation" / "val.txt"
VOCAB_CSV = DATA / "tokenizer" / "vocab.csv"


def load_ids(path: Path) -> torch.Tensor:
    """Read a dataset file (one token ID per line) into a 1-D tensor."""
    with path.open() as f:
        ids = [int(line) for line in f if line.strip()]
    return torch.tensor(ids, dtype=torch.long)


def load_vocab(path: Path) -> list[str]:
    """Read vocab.csv into a list where position = token ID (the file is in ID order)."""
    with path.open(encoding="utf-8", newline="") as f:
        return [row["token"] for row in csv.DictReader(f)]


class Windows(Dataset):
    """Every window in the corpus: `context` tokens and the token right after them."""

    def __init__(self, ids: torch.Tensor, context: int):
        self.ids = ids
        self.context = context

    def __len__(self) -> int:
        return len(self.ids) - self.context  # number of valid start positions

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ids[index : index + self.context], self.ids[index + self.context]


def make_loader(ids: torch.Tensor, args, seed: int) -> DataLoader:
    """Shuffled mini-batches: each window is used exactly once per epoch, in random order.

    The DataLoader calls Windows[i] for batch_size random i and stacks the results
    into x: (batch, context) and y: (batch,).
    """
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(Windows(ids, args.context), batch_size=args.batch_size, shuffle=True, generator=gen)


class MLP(torch.nn.Module):
    def __init__(self, vocab_size: int, context: int, embed_dim: int, hidden: int):
        super().__init__()
        self.context = context
        # One learned vector per token. Starts random; meaning comes from training.
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.hidden = torch.nn.Linear(context * embed_dim, hidden)
        self.output = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """x: (batch, context) token IDs → logits: (batch, vocab_size)."""
        emb = self.embedding(x)  # (batch, context, embed_dim)
        emb = emb.view(x.shape[0], -1)  # (batch, context * embed_dim): concatenate the window
        h = torch.relu(self.hidden(emb))  # (batch, hidden)
        return self.output(h)  # (batch, vocab_size), unnormalized scores


@torch.no_grad()
def estimate_loss(model, ids, args) -> float:
    """Average loss over the same random batches every call, without updating weights."""
    loader = make_loader(ids, args, seed=0)  # same seed, same batches, so changes are real
    losses = []
    for (x, y), _ in zip(loader, range(args.eval_batches)):
        losses.append(F.cross_entropy(model(x.to(args.device)), y.to(args.device)).item())
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(model, prompt: list[int], n: int, device: str) -> list[int]:
    """Repeatedly sample the next token from the model's distribution and append it."""
    out = list(prompt)
    for _ in range(n):
        x = torch.tensor([out[-model.context :]], device=device)
        probs = F.softmax(model(x)[0], dim=-1).cpu()
        out.append(int(torch.multinomial(probs, 1).item()))
    return out


def main():
    # Keep the docstring's layout and show each flag's default in --help.
    class Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    p = argparse.ArgumentParser(description=__doc__, formatter_class=Formatter)
    p.add_argument(
        "--context",
        type=int,
        default=3,
        help="number of previous tokens the model sees",
    )
    p.add_argument(
        "--embed-dim", type=int, default=64, help="size of each token embedding"
    )
    p.add_argument("--hidden", type=int, default=256, help="size of the hidden layer")
    p.add_argument("--batch-size", type=int, default=256, help="windows per training step")
    p.add_argument("--steps", type=int, default=20000, help="number of training steps")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--eval-interval", type=int, default=1000, help="steps between loss reports")
    p.add_argument("--eval-batches", type=int, default=50, help="batches averaged for each loss report")
    p.add_argument(
        "--generate", type=int, default=50, help="tokens to sample after training"
    )
    p.add_argument("--seed", type=int, default=1337, help="random seed, for repeatable runs")
    p.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="where to run: cpu, mps (Apple GPU) or cuda",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)  # repeatable weight init and sampling

    # The DataLoader slices and stacks windows on the CPU; each batch moves to the device.
    train_ids = load_ids(TRAIN_TXT)
    val_ids = load_ids(VAL_TXT)
    vocab = load_vocab(VOCAB_CSV)
    vocab_size = len(vocab)
    print(
        f"train: {len(train_ids):,} tokens | val: {len(val_ids):,} tokens | vocab: {vocab_size:,}"
    )

    model = MLP(vocab_size, args.context, args.embed_dim, args.hidden).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters on {args.device}")
    print(
        f"random guessing would give loss ln({vocab_size}) = {math.log(vocab_size):.2f}"
    )

    loader = make_loader(train_ids, args, seed=args.seed)
    print(f"one epoch = {len(loader):,} batches of {args.batch_size}")

    # Batches forever: each pass over `loader` is a new, freshly shuffled epoch.
    train_batches = (batch for _ in itertools.count() for batch in loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            train_loss = estimate_loss(model, train_ids, args)
            val_loss = estimate_loss(model, val_ids, args)
            print(
                f"step {step:>6} | epoch {step / len(loader):.2f} | train {train_loss:.3f} | val {val_loss:.3f} | {time.time() - start:.0f}s"
            )
        if step == args.steps:
            break

        x, y = next(train_batches)
        logits = model(x.to(args.device))
        loss = F.cross_entropy(logits, y.to(args.device))  # mean of -ln p[target]

        optimizer.zero_grad()
        loss.backward()  # backpropagation: gradient of the loss for every parameter
        optimizer.step()  # gradient descent: move each parameter against its gradient

    print(f"\nresult after {args.steps:,} steps ({time.time() - start:.0f}s):")
    print(f"  train loss {train_loss:.3f} | val loss {val_loss:.3f}")
    print(f"  random guessing {math.log(vocab_size):.3f}, so val is {math.log(vocab_size) - val_loss:.2f} lower")

    if args.generate > 0:
        prompt = val_ids[: args.context].tolist()
        sample = generate(model, prompt, args.generate, args.device)
        print("\nprompt:", " ".join(vocab[i] for i in prompt))
        print("sample:", " ".join(vocab[i] for i in sample))


if __name__ == "__main__":
    main()
