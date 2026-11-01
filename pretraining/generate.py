"""Generate text with the MLP model trained by train_mlp.py.

Loads the checkpoint from data/training/mlp.pt, encodes the prompt with the
vocabulary saved in it, then repeatedly samples the next token from the model's
probabilities and appends it:

    [the history of] → model → p(next) → sample "the" → [history of the] → ...

The model only sees the last `window_size` tokens, so anything earlier in the
text no longer affects what comes next.

Run from the repo root after `make pretrain`:

    make generate PROMPT="the history of"
    uv run pretraining/generate.py --prompt "the history of" --tokens 100
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from clean_text import split_punctuation
from train_mlp import CHECKPOINT, MLP, load_checkpoint

UNK_ID = 0  # <unk>; IDs 0-3 are the special tokens (see pkg/tokenizer)
NUM_SPECIAL = 4


def encode(text: str, vocab: list[str]) -> list[int]:
    """Separate punctuation like the training text, then look up each word like the Go tokenizer's Encode."""
    # Special tokens are left out, so typing "<eos>" gives <unk>, as in Go.
    ids = {token: i for i, token in enumerate(vocab) if i >= NUM_SPECIAL}
    return [ids.get(word, UNK_ID) for word in split_punctuation(text).split()]


def generate(model: MLP, prompt: list[int], n: int, device: str) -> list[int]:
    """Repeatedly sample the next token from the model's distribution and append it."""
    out = list(prompt)
    for _ in range(n):
        x = torch.tensor([out[-model.window_size :]], device=device)
        probs = F.softmax(model(x)[0], dim=-1).cpu()
        out.append(int(torch.multinomial(probs, 1).item()))
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prompt", default="The", help="text to continue (default: %(default)s)")
    p.add_argument("--tokens", type=int, default=50, help="tokens to generate (default: %(default)s)")
    p.add_argument("--seed", type=int, help="random seed, for repeatable output (default: different every run)")
    p.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT, help="trained model to load (default: %(default)s)"
    )
    p.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="where to run: cpu, mps (Apple GPU) or cuda (default: %(default)s)",
    )
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # The checkpoint carries its own vocabulary, so IDs always map back to the words
    # the model was trained on, even if the tokenizer was rebuilt since.
    model, vocab, checkpoint = load_checkpoint(args.checkpoint, args.device)
    print(f"model from step {checkpoint['step']:,} (val loss {checkpoint['val_loss']:.3f})")

    prompt = encode(args.prompt, vocab)
    if len(prompt) < model.window_size:
        # shortcut: pads short prompts with <unk>, which training only saw where words
        # were unknown; use a prompt of at least window_size words for better output.
        prompt = [UNK_ID] * (model.window_size - len(prompt)) + prompt

    out = generate(model, prompt, args.tokens, args.device)
    print("prompt:", " ".join(vocab[i] for i in prompt))
    print("output:", " ".join(vocab[i] for i in out))


if __name__ == "__main__":
    main()
