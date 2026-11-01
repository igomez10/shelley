"""Train an MLP language model (Bengio et al., 2003) on the prepared dataset.

The model predicts the next token from a fixed window of previous tokens:

    context tokens ──embedding lookup──► [E[t1] | E[t2] | E[t3]]   (concatenated)
                   ──hidden layer──────► relu(x · W1 + b1)
                   ──output layer──────► logits = h · W2 + b2      (one score per token)
                   ──softmax───────────► probability of every possible next token

Training minimizes cross-entropy, -ln p[actual next token], with backpropagation
(loss.backward()) to compute gradients and AdamW to update every parameter,
the embeddings included.

Whenever the validation loss reaches a new low, the model is saved to
data/training/mlp.pt, so the file always holds the best model of the run.
generate.py loads it to sample text.

Run from the repo root after `make tokenizer` and `make prepare`:

    make pretrain
    uv run pretraining/train_mlp.py --steps 5000
"""

import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

DATA = Path("data")
TRAIN_TXT = DATA / "preparation" / "train.txt"
VAL_TXT = DATA / "preparation" / "val.txt"
VOCAB_CSV = DATA / "tokenizer" / "vocab.csv"
CHECKPOINT = DATA / "training" / "mlp.pt"


def load_ids(path: Path) -> torch.Tensor:
    """Read a dataset file (one token ID per line) into a 1-D tensor.

    Parses the digits with whole-array numpy operations, about 15 s for the 626M
    lines of the default mix. A Python loop, or np.fromfile(sep="\\n"), which keeps
    growing its buffer, takes many minutes at that size.
    """
    text = np.fromfile(path, dtype=np.uint8)
    if text.size == 0:
        return torch.zeros(0, dtype=torch.long)
    if text[-1] != ord("\n"):
        text = np.append(text, np.uint8(ord("\n")))
    is_digit = (text >= ord("0")) & (text <= ord("9"))
    if not (is_digit | (text == ord("\n"))).all():
        raise ValueError(
            f"{path}: expected only digits and newlines, one token ID per line"
        )

    ends = np.flatnonzero(text == ord("\n"))  # where each line ends
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts  # digits per line
    if (lengths == 0).any():
        raise ValueError(f"{path}: blank line; expected one token ID per line")

    # Build every number at once, one digit position per pass: "467" -> 4, 46, 467.
    ids = np.zeros(len(ends), dtype=np.int64)
    for k in range(int(lengths.max())):
        longer = lengths > k
        ids[longer] = ids[longer] * 10 + (text[starts[longer] + k] - ord("0"))
    return torch.from_numpy(ids)


def load_vocab(path: Path) -> list[str]:
    """Read vocab.csv into a list where position = token ID (the file is in ID order)."""
    with path.open(encoding="utf-8", newline="") as f:
        return [row["token"] for row in csv.DictReader(f)]


class Windows(Dataset):
    """Every window in the corpus: `window_size` input tokens and the token right after them."""

    def __init__(self, ids: torch.Tensor, window_size: int):
        self.token_ids = ids
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.window_size  # number of valid start positions

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.token_ids[index : index + self.window_size], self.token_ids[
            index + self.window_size
        ]


def make_loader(dataset: Dataset, args, seed: int) -> DataLoader:
    """Shuffled mini-batches: each window is used exactly once per epoch, in random order.

    The DataLoader calls Windows[i] for batch_size random i and stacks the results
    into x: (batch, window_size) and y: (batch,).
    """
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=gen)


class MLP(torch.nn.Module):
    def __init__(self, vocab_size: int, window_size: int, embed_dim: int, hidden: int):
        super().__init__()
        self.window_size = window_size
        # One learned vector per token. Starts random; meaning comes from training.
        self.word_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.hidden_layer_1 = torch.nn.Linear(window_size * embed_dim, hidden)
        self.hidden_layer_2 = torch.nn.Linear(hidden, hidden)
        self.hidden_layer_3 = torch.nn.Linear(hidden, hidden)
        self.output_layer = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """x: (batch, window_size) token IDs → logits: (batch, vocab_size)."""
        embedded_tokens = self.word_embedding(x)  # (batch, window_size, embed_dim)
        embedded_tokens = embedded_tokens.view(
            x.shape[0], -1
        )  # (batch, window_size * embed_dim): concatenate the window
        hidden_output = torch.relu(
            self.hidden_layer_1(embedded_tokens)
        )  # (batch, hidden)
        hidden_output = torch.relu(
            self.hidden_layer_2(hidden_output)
        )  # (batch, hidden)
        hidden_output = torch.relu(
            self.hidden_layer_3(hidden_output)
        )  # (batch, hidden)
        return self.output_layer(
            hidden_output
        )  # (batch, vocab_size), unnormalized scores


def estimate_loss(model, dataset: Windows, args) -> float:
    """Average loss over the same random batches every call, without updating weights."""
    # Draw only the windows needed, instead of shuffling the whole dataset each check.
    # Same seed every call, so the batches are the same and changes are real.
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=args.eval_batches * args.batch_size,
        generator=torch.Generator().manual_seed(0),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    losses = []
    for (x, y), _ in zip(loader, range(args.eval_batches)):
        losses.append(
            F.cross_entropy(model(x.to(args.device)), y.to(args.device)).item()
        )
    return sum(losses) / len(losses)


def load_checkpoint(path: Path, device: str) -> tuple[MLP, list[str], dict]:
    """Rebuild a trained model from a checkpoint saved by train(), with its vocabulary."""
    if not path.exists():
        sys.exit(f"no trained model at {path}; run `make pretrain` first")
    checkpoint = torch.load(path, map_location=device)
    if "vocab" not in checkpoint:
        sys.exit(
            f"{path} was saved before checkpoints included the vocabulary; run `make pretrain` again"
        )
    model = MLP(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint["vocab"], checkpoint


def train(train_ds: Windows, val_ds: Windows, vocab: list[str], args) -> None:
    """Train a fresh model on train_ds, reporting losses on train_ds and val_ds.

    Saves a checkpoint each time the validation loss reaches a new low, and stops
    early once it hasn't improved for `args.patience` loss checks in a row.
    """
    torch.manual_seed(args.seed)  # repeatable weight init
    vocab_size = len(vocab)
    config = {
        "vocab_size": vocab_size,
        "window_size": args.window_size,
        "embed_dim": args.embed_dim,
        "hidden": args.hidden,
    }
    model = MLP(**config).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters on {args.device}")
    print(
        f"random guessing would give loss ln({vocab_size}) = {math.log(vocab_size):.2f}"
    )

    loader = make_loader(train_ds, args, seed=args.seed)
    print(f"one epoch = {len(loader):,} batches of {args.batch_size}")

    # Batches forever: each pass over `loader` is a new, freshly shuffled epoch.
    train_batches = (batch for _ in itertools.count() for batch in loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.time()
    best_val, best_step, best_train = math.inf, 0, math.inf
    checks_without_improvement = 0
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            train_loss = estimate_loss(model, train_ds, args)
            val_loss = estimate_loss(model, val_ds, args)
            if val_loss < best_val:
                best_val, best_step, best_train = val_loss, step, train_loss
                checks_without_improvement = 0
                # The config and vocabulary are saved too, so generate.py can rebuild the
                # same model and turn its IDs back into the right words.
                checkpoint = {
                    "model": model.state_dict(),
                    "config": config,
                    "vocab": vocab,
                    "step": step,
                    "val_loss": val_loss,
                }
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, args.checkpoint)
                note = "saved"
            else:
                checks_without_improvement += 1
                # One worse check can be noise; several in a row means overfitting.
                note = f"no improvement {checks_without_improvement}/{args.patience or '-'}"
            # gap = val - train; it widens as the model fits the training text more than new text.
            print(
                f"step {step:>6} | epoch {step / len(loader):.2f} | train {train_loss:.3f} | val {val_loss:.3f} | gap {val_loss - train_loss:.3f} | {time.time() - start:.0f}s | {note}"
            )
            if args.patience and checks_without_improvement >= args.patience:
                print(
                    f"\nsince the best val loss at step {best_step:,} (epoch {best_step / len(loader):.2f}):"
                )
                print(f"  train {best_train:.3f} -> {train_loss:.3f}")
                print(f"  val   {best_val:.3f} -> {val_loss:.3f}")
                gap_growth = (val_loss - train_loss) - (best_val - best_train)
                # Overfitting: train keeps improving while val falls behind, so the gap widens.
                # A plateau: both have flattened and the gap barely moves (eval noise is ~0.01).
                if gap_growth > 0.05:
                    print(
                        f"  the gap widened by {gap_growth:.3f}: overfitting (memorizing the training text)"
                    )
                else:
                    print(
                        f"  the gap moved by {gap_growth:+.3f}: a plateau, not overfitting; train and val"
                    )
                    print(
                        "  flattened together (more --patience, a bigger model or learning-rate decay may help)"
                    )
                print(
                    f"early stopping: no new best val loss in {args.patience} checks ({args.patience * args.eval_interval:,} steps)"
                )
                break
        if step == args.steps:
            break

        x, y = next(train_batches)
        logits = model(x.to(args.device))
        loss = F.cross_entropy(logits, y.to(args.device))  # mean of -ln p[target]

        optimizer.zero_grad()
        loss.backward()  # backpropagation: gradient of the loss for every parameter
        optimizer.step()  # gradient descent: move each parameter against its gradient

    print(f"\nresult after {step:,} steps ({time.time() - start:.0f}s):")
    print(f"  train loss {train_loss:.3f} | val loss {val_loss:.3f}")
    print(
        f"  random guessing {math.log(vocab_size):.3f}, so val is {math.log(vocab_size) - val_loss:.2f} lower"
    )
    print(
        f"  best val loss {best_val:.3f} at step {best_step:,}, saved to {args.checkpoint}"
    )


def main():
    # Keep the docstring's layout and show each flag's default in --help.
    class Formatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    p = argparse.ArgumentParser(description=__doc__, formatter_class=Formatter)
    p.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="number of previous tokens the model sees (the context window)",
    )
    p.add_argument(
        "--embed-dim", type=int, default=64, help="size of each token embedding"
    )
    p.add_argument("--hidden", type=int, default=256, help="size of the hidden layer")
    p.add_argument(
        "--batch-size", type=int, default=256, help="windows per training step"
    )
    p.add_argument("--steps", type=int, default=20000, help="number of training steps")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument(
        "--eval-interval", type=int, default=1000, help="steps between loss reports"
    )
    p.add_argument(
        "--eval-batches",
        type=int,
        default=50,
        help="batches averaged for each loss report",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="stop after this many loss checks in a row without a new best val loss (0 = never stop early)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT,
        help="where to save the model with the lowest validation loss",
    )
    p.add_argument(
        "--seed", type=int, default=1337, help="random seed, for repeatable runs"
    )
    p.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="where to run: cpu, mps (Apple GPU) or cuda",
    )
    args = p.parse_args()

    # The DataLoader slices and stacks windows on the CPU; each batch moves to the device.
    train_ids = load_ids(TRAIN_TXT)
    val_ids = load_ids(VAL_TXT)
    vocab = load_vocab(VOCAB_CSV)
    vocab_size = len(vocab)
    print(
        f"train: {len(train_ids):,} tokens | val: {len(val_ids):,} tokens | vocab: {vocab_size:,}"
    )

    train(
        Windows(train_ids, args.window_size),
        Windows(val_ids, args.window_size),
        vocab,
        args,
    )


if __name__ == "__main__":
    main()
