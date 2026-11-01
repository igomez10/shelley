"""Find the words whose learned embeddings are closest to a given word.

Every token has a row in the model's embedding table (vocab_size × embed_dim).
Training pushes tokens that appear in similar contexts towards similar vectors,
so nearby vectors tend to mean related words. Closeness is measured with cosine
similarity, the cosine of the angle between two vectors:

    similarity(a, b) = a · b / (|a| |b|)      1 = same direction, 0 = unrelated

Run from the repo root after `make pretrain`:

    make similar WORDS="king happy dog"
    uv run pretraining/similar.py king happy dog --top 5
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from train_mlp import CHECKPOINT, load_checkpoint

NUM_SPECIAL = 4  # <unk> <bos> <eos> <pad>, never shown as neighbours


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("words", nargs="+", help="words to find neighbours for")
    p.add_argument("--top", type=int, default=10, help="neighbours to show per word (default: %(default)s)")
    p.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT, help="trained model to load (default: %(default)s)"
    )
    args = p.parse_args()

    model, vocab, checkpoint = load_checkpoint(args.checkpoint, "cpu")
    print(f"model from step {checkpoint['step']:,} (val loss {checkpoint['val_loss']:.3f})")
    ids = {token: i for i, token in enumerate(vocab)}

    # Rows scaled to length 1, so a dot product is the cosine similarity.
    embeddings = F.normalize(model.word_embedding.weight.detach(), dim=1)  # (vocab, embed_dim)
    for word in args.words:
        if word not in ids or ids[word] < NUM_SPECIAL:
            print(f"\n{word}: not in the vocabulary")
            continue
        similarity = embeddings @ embeddings[ids[word]]  # (vocab,): cosine with every token
        similarity[ids[word]] = -1  # skip the word itself
        similarity[:NUM_SPECIAL] = -1
        top = torch.topk(similarity, args.top)
        print(f"\n{word}:")
        for score, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {score:.3f}  {vocab[i]}")


if __name__ == "__main__":
    main()
