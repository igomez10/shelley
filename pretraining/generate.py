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
END_OF_STORY = "<|endoftext|>"  # TinyStories' story separator, an ordinary token
NUM_SPECIAL = 4
# A finished story to fill the window in front of a short prompt, followed by one
# <|endoftext|>, so the model sees the same pattern as between training stories.
FILLER = (
    "Once upon a time , there was a little dog named Max . Max liked to play in the park "
    "with his friend Sam . One day , they found a big red ball under a tree . They played "
    "with the ball all day . When the sun went down , Max and Sam went home . They were "
    "tired but happy . The end ."
)


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
        logits = model(x)[0]
        # Special tokens never appear in the training data, so their embeddings are
        # untrained; sampling one would feed noise back into the window.
        logits[:NUM_SPECIAL] = float("-inf")
        probs = F.softmax(logits, dim=-1).cpu()
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
    # The model always reads exactly window_size tokens, so a shorter prompt is padded
    # on the left. Repeating one token (34 x <|endoftext|>) is a window training never
    # saw, and the model's output on it is garbage. Instead, pad with the end of a
    # finished story and one <|endoftext|>, which in training is followed by a new story.
    padding = max(0, model.window_size - len(prompt))
    filler = encode(FILLER, vocab) + ([vocab.index(END_OF_STORY)] if END_OF_STORY in vocab else [])
    filler = filler * (padding // len(filler) + 1)  # repeat until long enough
    filler = filler[len(filler) - padding :]  # keep the last `padding` tokens
    out = generate(model, filler + prompt, args.tokens, args.device)[padding:]
    if padding:
        print(f"prompt padded with {padding} tokens of a sample story to fill the {model.window_size}-token window")
    print("prompt:", " ".join(vocab[i] for i in prompt))
    print("output:", " ".join(vocab[i] for i in out))


if __name__ == "__main__":
    main()
