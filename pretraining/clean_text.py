"""Clean raw TinyStories text so the word-level tokenizer gets sensible words.

Reads raw text on stdin and writes cleaned text to stdout:

  - drops anything after the last story separator (e.g. a cut-off download)
  - turns curly quotes and dashes into plain ASCII ones
  - puts spaces around punctuation, so "said," becomes "said ," and the
    vocabulary doesn't fill up with "said", "said,", "said." and "said!"

Apostrophes inside words are kept: "don't" and "Lily's" stay one token.

    curl ... | uv run pretraining/clean_text.py > data/raw/tinystories/train.txt

`make data` runs this on the full download.
"""

import re
import sys

END_OF_STORY = "<|endoftext|>"  # TinyStories puts this line between stories

PLAIN = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})
PUNCTUATION = re.compile(r'([.,!?;:"()])')


def split_punctuation(text: str) -> str:
    """Put spaces around punctuation marks, so they become separate tokens."""
    text = PUNCTUATION.sub(r" \1 ", text.translate(PLAIN))
    return re.sub(r"[ \t]+", " ", text)


def main():
    text = sys.stdin.read()
    # Drop anything after the last separator, e.g. a story cut off by a partial download.
    last = text.rfind(END_OF_STORY)
    if last == -1:
        sys.exit(f"no {END_OF_STORY} found: not TinyStories text, or the slice is too small")
    text = text[: last + len(END_OF_STORY)]
    lines = (split_punctuation(line).strip() for line in text.splitlines())
    sys.stdout.write("\n".join(line for line in lines if line) + "\n")


if __name__ == "__main__":
    main()
