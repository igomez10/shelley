# shelley

A large language model built from scratch in Go, for learning.

The goal is to understand how an LLM works by writing every piece myself, from the tokenizer to the training loop, without ML frameworks. Expect rough edges. Readable code matters more here than speed.

## Status

Early. Right now there's only the tokenizer.

| Stage                         | Status         |
| ----------------------------- | -------------- |
| Tokenizer (word-level, trie)  | in progress    |
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
pkg/tokenizer/              tokenizer + trie
```

### Tokenizer

`pkg/tokenizer` is a word-level tokenizer:

- **`Init(corpus, out)`** reserves IDs 0–3 for special tokens, then splits the corpus on whitespace, sorts and deduplicates the words, and numbers them from 4 in that order. It also counts how many times each word appears and stores that in `Counts`, indexed by ID. It builds both an ID → word map and a word → ID trie, then gob-encodes the tokenizer to `out` (if not nil).
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
cat data/corpus.txt | ./cli init

# tokenize arguments or stdin
./cli tokenize the cat sat
echo "the cat sat" | ./cli tokenize
```

Both commands read and write the tokenizer at `/tmp/tokenizer.gob` by default. Pass `--tokenizer <path>` (`-t`) to use a different file.

### Tokenizing a sequence

1. Download a corpus and build the vocabulary from it (only needed once):

   ```sh
   make data        # WikiText-2 → data/corpus.txt
   make tokenizer   # builds ./cli and saves /tmp/tokenizer.gob
   ```

2. Tokenize text, passed either as arguments or on stdin:

   ```sh
   ./cli tokenize "the history of the Roman Empire ."
   echo "the history of the Roman Empire ." | ./cli tokenize
   ./cli tokenize < data/wikitext-2-raw/wiki.test.raw   # a whole file
   ```

   `to` is a short alias for `tokenize`. Log lines go to stderr, so add `2>/dev/null` to see only the result.

   Or use make, which builds the CLI first and hides the logs:

   ```sh
   make tokenize TEXT="the history of the Roman Empire ."
   make tokenize FILE=data/wikitext-2-raw/wiki.test.raw
   ```

3. Read the output. There is one ID per word, in the same order as the input:

   ```text
   Tokenized output: [72167 54867 61476 72167 31193 13129 271]
   ```

   Repeated words get the same ID (`the` is `72167` both times), and words not in the vocabulary get `0` (`<unk>`). The exact numbers depend on the corpus the tokenizer was built from.

### Inspecting the vocabulary

`vocab` exports every token with its ID and how many times it appeared in the corpus, as CSV ordered by ID:

```sh
make vocab                              # writes data/vocab.csv
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

Tokens that contain commas or quotes are quoted in standard CSV style, so `","` is the comma token and `""""` is the `"` token. The file opens directly in a spreadsheet. Special tokens never appear in the corpus, so their count is 0.

To list the most frequent tokens:

```sh
tail -n +2 data/vocab.csv | sort -t, -k3 -rn | head
```

To use a tokenizer saved somewhere else, pass it with `-t`:

```sh
./cli init -t /tmp/small.gob the cat sat on the mat
./cli tokenize -t /tmp/small.gob the dog sat   # [8 0 7]
```

From Go code:

```go
f, _ := os.Open("/tmp/tokenizer.gob")
tkn, err := tokenizer.NewFromFile(f)
ids := tkn.Encode("the cat sat")
text := tkn.Decode(ids) // "the cat sat"
```

### Make targets

Run `make` to see them all. The main ones:

```sh
make build       # build ./cli
make test        # run tests
make check       # gofmt check, go vet, tests with the race detector
make tokenizer   # build a tokenizer from data/corpus.txt
make tokenizer CORPUS=other.txt TOKENIZER=/tmp/other.gob
make tokenize TEXT="the cat sat"   # or FILE=path; TOKENIZER= also works
make vocab       # export the vocabulary to data/vocab.csv
make clean
```

## Data

Training corpora go in `data/`, which git ignores.

`make data` downloads [WikiText-2](https://wikitext.smerity.com/) (raw), about 11 MB of English Wikipedia articles split into train, validation and test files under `data/wikitext-2-raw/`. It copies the train split to `data/corpus.txt`, which `make tokenizer` reads by default. `make clean-data` deletes it all.
