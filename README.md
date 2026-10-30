# shelley

A large language model built from scratch in Go, for learning.

The goal is to understand how an LLM works by writing every piece myself, from the tokenizer to the training loop, without ML frameworks. Expect rough edges. Readable code matters more here than speed.

## Status

Early. There's a tokenizer and the training dataset it produces.

| Stage                         | Status         |
| ----------------------------- | -------------- |
| Tokenizer (word-level, trie)  | in progress    |
| Dataset preparation           | done           |
| Subword tokenizer (BPE)       | not started    |
| Embeddings                    | not started    |
| Self-attention                | not started    |
| Transformer block             | not started    |
| Training loop / backprop      | not started    |
| Text generation (sampling)    | not started    |

## Layout

```text
cmd/cli/                    CLI entry point (urfave/cli v3)
  main.go                   registers subcommands
  flags/                    shared flags (--verbose, --tokenizer)
  input/                    reads command input from args or stdin
  commands/init/            `shelley init`: build a tokenizer from a corpus
  commands/tokenize/        `shelley tokenize`: encode text into token IDs
  commands/vocab/           `shelley vocab`: export the vocabulary as CSV
  commands/prepare/         `shelley prepare`: encode a corpus into a training dataset
pkg/tokenizer/              tokenizer + trie
pkg/dataset/                dataset file format + training examples and batches
```

### Tokenizer

`pkg/tokenizer` is a word-level tokenizer:

- **`Init(corpus, out)`** reserves IDs 0–3 for special tokens, then splits the corpus on whitespace, sorts and deduplicates the words, and numbers them from 4 in that order. It also counts how many times each word appears and stores that in `Counts`, indexed by ID. A `vocabSize` greater than 0 keeps only the most frequent words (ties broken alphabetically). The words left out encode as `<unk>`, and their occurrences are added to `<unk>`'s count. It builds both an ID → word map and a word → ID trie, then gob-encodes the tokenizer to `out` (if not nil).
- **`Encode(text)`** splits on whitespace and looks up each word in the trie. Unknown words get `UnkID` (0), including words that are only a prefix of a known word.
- **`Decode(ids)`** maps IDs back to words and joins them with spaces, writing `<unk>` for unknown IDs.
- **`NewFromFile(r)`** loads a tokenizer saved by `Init`.

Special tokens:

| ID | Token   | Meaning                                             |
| -- | ------- | --------------------------------------------------- |
| 0  | `<unk>` | word not in the vocabulary                          |
| 1  | `<bos>` | beginning of a sequence                             |
| 2  | `<eos>` | end of a sequence                                   |
| 3  | `<pad>` | filler to make sequences in a batch the same length |

Every ID is a valid index, so it can be used directly as a row in the embedding table. `Encode` never produces `<bos>`, `<eos>` or `<pad>`: the caller adds them, and the text `"<eos>"` in the input encodes as `<unk>`.

The trie (`Trie`, `TrieNode`) is keyed by rune, so it handles any UTF-8 text, and supports `Insert`, `Search` and `StartsWith`.

## Usage

Requires Go 1.24+.

```sh
go build -o cli ./cmd/cli

# build a tokenizer from a corpus (args or stdin)
./cli init < data/raw/wikitext-2-raw/wiki.train.raw

# tokenize arguments or stdin
./cli tokenize the cat sat
echo "the cat sat" | ./cli tokenize
```

All commands read and write the tokenizer at `data/tokenizer/tokenizer.gob` by default, relative to where you run the CLI (normally the repo root). Output folders are created as needed. Pass `--tokenizer <path>` (`-t`) to use a different file.

### Tokenizing a sequence

1. Download a corpus and build the vocabulary from it (only needed once):

   ```sh
   make data        # WikiText-2 → data/raw/
   make tokenizer   # builds ./cli and saves a 10,000-token data/tokenizer/tokenizer.gob
   ```

2. Tokenize text, passed either as arguments or on stdin:

   ```sh
   ./cli tokenize "the history of the Roman Empire ."
   echo "the history of the Roman Empire ." | ./cli tokenize
   ./cli tokenize < data/raw/wikitext-2-raw/wiki.test.raw   # a whole file
   ```

   `to` is a short alias for `tokenize`. Log lines go to stderr, so add `2>/dev/null` to see only the result.

   Or use make, which builds the CLI first and hides the logs:

   ```sh
   make tokenize TEXT="the history of the Roman Empire ."
   make tokenize FILE=data/raw/wikitext-2-raw/wiki.test.raw
   ```

3. Read the output. There is one ID per word, in the same order as the input:

   ```text
   Tokenized output: [9339 6333 7421 9339 2903 1381 31]
   ```

   Repeated words get the same ID (`the` is `9339` both times), and words not in the vocabulary get `0` (`<unk>`). The exact numbers depend on the corpus the tokenizer was built from.

### Inspecting the vocabulary

`vocab` exports every token with its ID and how many times it appeared in the corpus, as CSV ordered by ID:

```sh
make vocab                              # writes data/tokenizer/vocab.csv
./cli vocab > vocab.csv                 # or to stdout
./cli vocab -t /tmp/small.gob -o small.csv
```

```csv
token,id,count
<unk>,0,0
<bos>,1,0
<eos>,2,0
<pad>,3,0
!,4,486
"""",5,28309
...
```

Tokens that contain commas or quotes are quoted in standard CSV style, so `","` is the comma token and `""""` is the `"` token. The file opens directly in a spreadsheet. The `<unk>` count is the number of corpus words left out of the vocabulary. The other special tokens never appear in the corpus, so their count is 0.

To list the most frequent tokens:

```sh
tail -n +2 data/tokenizer/vocab.csv | sort -t, -k3 -rn | head
```

To use a tokenizer saved somewhere else, pass it with `-t`:

```sh
./cli init -t /tmp/small.gob the cat sat on the mat
./cli tokenize -t /tmp/small.gob the dog sat   # [8 0 7]
```

From Go code:

```go
f, _ := os.Open("data/tokenizer/tokenizer.gob")
tkn, err := tokenizer.NewFromFile(f)
ids := tkn.Encode("the cat sat")
text := tkn.Decode(ids) // "the cat sat"
```

### Preparing the training data

Training needs the corpus as token IDs, not text. `prepare` encodes a corpus once with the tokenizer and saves the IDs to a text file, so training reads numbers from disk instead of re-tokenizing text on every pass:

```sh
make prepare     # data/preparation/train.txt and val.txt from the WikiText-2 splits
./cli prepare -o data/preparation/train.txt < data/raw/wikitext-2-raw/wiki.train.raw
```

```text
Encoded 2051910 tokens (202121 unknown, 9.9%) to data/preparation/train.txt
Example with context length 8:
  inputs:  [467 3397 1039 1850 467 0 7337 3397]  = Valkyria Chronicles III = <unk> no Valkyria
  targets: [3397 1039 1850 467 0 7337 3397 349]  Valkyria Chronicles III = <unk> no Valkyria 3
```

The file is plain text with one token ID per line, in corpus order, so you can open it in any editor or use `head`:

```sh
head -4 data/preparation/train.txt
# 467
# 3397
# 1039
# 1850
```

Line `n` holds token `n-1`, since IDs are counted from 0. Real LLM pipelines store tokens as fixed-size binary integers instead (nanoGPT uses `uint16`), which makes huge datasets smaller and lets training jump straight to any position. At 2 million tokens that makes no difference, so this project keeps the readable format. Only train and validation are encoded. The test split stays untouched for a final evaluation.

`pkg/dataset` reads these files and cuts them into training examples for **next-token prediction**:

- **`Example(start, contextLen)`** returns a **context window** of `contextLen` tokens and its **targets**, which are the same window shifted one token ahead. Each position is one prediction: after `=` comes `Valkyria`, after `= Valkyria` comes `Chronicles`, and so on, so one window holds `contextLen` training examples.
- **`Batch(rng, batchSize, contextLen)`** returns `batchSize` windows starting at random positions, the input to one training step. Pass a seeded `rand.Rand` to get the same batches every run.

### Make targets

Run `make` to see them all. The main ones:

```sh
make build       # build ./cli
make test        # run tests
make check       # gofmt check, go vet, tests with the race detector
make tokenizer   # build a 10,000-token tokenizer from the WikiText-2 train split
make tokenizer VOCAB_SIZE=0        # no limit: keep every word
make tokenizer CORPUS=other.txt TOKENIZER=/tmp/other.gob
make tokenize TEXT="the cat sat"   # or FILE=path; TOKENIZER= also works
make vocab       # export the vocabulary to data/tokenizer/vocab.csv
make prepare     # encode train/val splits into data/preparation/
make clean
```

## Data

Everything generated lives in `data/`, which git ignores. It has one folder per pipeline step, and each step is built from the one before it:

```text
data/
  raw/            1. make data       downloaded corpus
    wikitext-2-raw/wiki.{train,valid,test}.raw
  tokenizer/      2. make tokenizer  tokenizer built from the train split
    tokenizer.gob
    vocab.csv        make vocab      the vocabulary as token,id,count
  preparation/    3. make prepare    train/validation splits as token IDs
    train.txt
    val.txt
```

The corpus is [WikiText-2](https://wikitext.smerity.com/) (raw), about 11 MB of English Wikipedia articles already split into train, validation and test. Only the train split is used to build the tokenizer, and the test split is never encoded, so it stays untouched for a final evaluation.

If you rebuild the tokenizer, for example with a different `VOCAB_SIZE`, run `make vocab` and `make prepare` again: the token IDs in later steps depend on it. `make clean-data` deletes all three folders.
