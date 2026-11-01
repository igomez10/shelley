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
| MLP model (PyTorch reference) | done           |
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
pretraining/                PyTorch reference models
  train_mlp.py              MLP language model trained on data/preparation
pyproject.toml, uv.lock     Python dependencies (uv project at the repo root)
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
   make data        # TinyStories + WikiText-103 → data/raw/ (see DATASETS)
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

### Training the MLP model (PyTorch)

`pretraining/train_mlp.py` trains the MLP language model from Bengio et al. (2003) on the prepared data. It's a reference implementation in PyTorch, to compare against when the same model is written by hand in Go. It needs [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and PyTorch into `.venv` at the repo root on the first run.

```sh
make pretrain                          # 20,000 steps, about 1 minute on an Apple Silicon GPU
make pretrain ARGS="--steps 5000 --window-size 5 --hidden 512"
uv run pretraining/train_mlp.py --help
```

The model looks up the embeddings of the previous `--window-size` tokens (default 3), joins them into one vector, passes it through a ReLU hidden layer, and outputs a score for every token in the vocabulary. Training minimizes cross-entropy with AdamW. Mini-batches come from a PyTorch `DataLoader` over every window in the corpus, shuffled so that each window is used exactly once per **epoch** (one full pass over the data, about 8,000 batches of 256). It prints the loss on train and validation every 1,000 steps (measured on the same batches each time, so the numbers are comparable), then a summary of the final losses. Whenever the validation loss reaches a new low, it saves the model to `data/training/mlp.pt` (`| saved` in the log), so the file keeps the best model of the run rather than the last one, which has usually started to overfit:

```text
model: 3,259,408 parameters on mps
random guessing would give loss ln(10000) = 9.21
one epoch = 8,016 batches of 256
step      0 | epoch 0.00 | train 9.221 | val 9.210 | gap -0.011 | 0s | saved
step  16000 | epoch 2.00 | train 4.635 | val 5.009 | gap 0.374 | 38s | saved
step  17000 | epoch 2.12 | train 4.593 | val 5.057 | gap 0.464 | 40s | no improvement 1/3
step  18000 | epoch 2.25 | train 4.567 | val 5.075 | gap 0.508 | 42s | no improvement 2/3
step  19000 | epoch 2.37 | train 4.540 | val 5.076 | gap 0.536 | 45s | no improvement 3/3

overfitting since step 16,000 (epoch 2.00):
  train 4.635 -> 4.540 (still learning the training text)
  val   5.009 -> 5.076 (no longer improving on new text)
early stopping: no new best val loss in 3 checks (3,000 steps)

result after 19,000 steps (45s):
  train loss 4.540 | val loss 5.076
  random guessing 9.210, so val is 4.13 lower
  best val loss 5.009 at step 16,000, saved to data/training/mlp.pt
```

**Early stopping:** training stops once the validation loss hasn't set a new best for `--patience` checks in a row (default 3, so 3,000 steps). `gap` is val minus train. It grows as the model fits the training text better than new text, and when train keeps falling while val stops improving, the model is **overfitting**: memorizing training windows instead of learning patterns that carry over. One worse check alone can be noise, which is why it waits for several. `--patience 0` turns early stopping off.

Each run replaces `data/training/mlp.pt` with its own best model, so a short test run overwrites a better model from an earlier run. Use `--checkpoint /tmp/test.pt` for experiments.

For reference, on the validation split:

| Model                                  | Validation loss |
| -------------------------------------- | --------------- |
| uniform guess                          | 9.21            |
| unigram (overall word frequency)       | 6.19            |
| bigram by counting (previous word)     | 5.29            |
| MLP, window 3 (this script, ReLU)      | ~5.01 (best)    |

### Generating text

`pretraining/generate.py` loads the saved model and continues a prompt by sampling one token at a time: it computes the probability of every next token, picks one at random according to those probabilities, appends it, and repeats with the last `--window-size` tokens.

```sh
make generate                                        # prompt "The", 50 tokens
make generate PROMPT="the history of the" ARGS="--tokens 100 --seed 1"
uv run pretraining/generate.py --help
```

```text
model from step 16,000 (val loss 5.005)
prompt: the history of the
output: the history of the series . about <unk> of the family members . While this eyes response came and <unk> ...
```

The prompt is split on whitespace and looked up in the vocabulary saved inside the checkpoint, like `./cli tokenize`; unknown words become `<unk>`. Saving the vocabulary with the model means a rebuilt tokenizer can't make an old model decode its IDs into the wrong words. A checkpoint saved before this change is rejected with a message to retrain. A prompt shorter than the window is padded at the start with the end of a short sample story and one `<|endoftext|>`, the same pattern the model saw between training stories (a run of one repeated token is a window it never saw, and its output on it is garbage). Special tokens are never sampled: they never appear in the training data, so their embeddings are untrained. Output is different every run unless you pass `--seed`. If there's no trained model yet, it tells you to run `make pretrain`.

### Word similarity

`pretraining/similar.py` looks up a word's learned embedding and lists the words whose embeddings point in the most similar direction (**cosine similarity**, `a·b / (|a||b|)`, from 1 for the same direction down to 0 for unrelated):

```sh
make similar                                     # default words: king happy dog Monday
make similar WORDS="dog big ran" ARGS="--top 5"
```

```text
dog:  0.475 cat   0.458 kitten   0.407 Therefore   0.407 owl   0.393 families
big:  0.469 small   0.461 huge   0.400 commercial   0.400 Bonnie   0.388 tall
ran:  0.484 danced   0.450 flying   0.447 comes   0.446 crawled   0.426 flew
```

Nobody tells the model that `dog` and `cat` are related. They end up close because they appear in similar places (`the ___ ran`, `a little ___`), and similar contexts give similar gradients. `small` sits next to `big` for the same reason: opposites fill the same slots in a sentence. Words seen less often, and models trained for fewer steps, give noisier neighbours (the list above is from the default 20,000 steps, under 1% of the data).

### Make targets

Run `make` to see them all. The main ones:

```sh
make build       # build ./cli
make test        # run tests
make check       # gofmt check, go vet, tests with the race detector
make tokenizer   # build a 10,000-token tokenizer from the train split
make tokenizer VOCAB_SIZE=0        # no limit: keep every word
make tokenizer CORPUS=other.txt TOKENIZER=/tmp/other.gob
make tokenize TEXT="the cat sat"   # or FILE=path; TOKENIZER= also works
make vocab       # export the vocabulary to data/tokenizer/vocab.csv
make prepare     # encode train/val splits into data/preparation/
make pretrain    # train the PyTorch MLP model, save the best one to data/training/mlp.pt
make generate PROMPT="the history of"   # generate text with the trained model
make similar WORDS="dog big"            # nearest words by learned embedding
make clean
```

## Data

Everything generated lives in `data/`, which git ignores. It has one folder per pipeline step, and each step is built from the one before it:

```text
data/
  raw/            1. make data       downloaded corpora, one folder per source
    tinystories/{train,valid}.txt                cleaned on download
    wikitext-103-raw/wiki.{train,valid,test}.raw
    wikitext-2-raw/wiki.{train,valid,test}.raw   only with DATASETS=... wikitext
  tokenizer/      2. make tokenizer  tokenizer built from the train split
    tokenizer.gob
    vocab.csv        make vocab      the vocabulary as token,id,count
  preparation/    3. make prepare    train/validation splits as token IDs
    train.txt
    val.txt
  training/       4. make pretrain   the model with the lowest validation loss
    mlp.pt
```

The pipeline trains on several sources at once. `DATASETS` in the Makefile lists them (default `tinystories wikitext103`). Their train files are concatenated into one corpus for `make tokenizer` and `make prepare`, and their validation files into one validation set:

| Source | Train words | Share | Unknown words in its validation text (10k vocabulary) |
| ------ | ----------- | ----- | ---------------------------------------------------- |
| `tinystories` | 525M | 84% | 0.4% |
| `wikitext103` | 101M | 16% | 12.4% |
| `wikitext` (WikiText-2, not in the default) | 2M | | |

The mix matters. With `tinystories wikitext` (WikiText-2), Wikipedia was only 0.4% of the text: the shared 10,000-token vocabulary was almost all TinyStories words, and 21% of Wikipedia validation words became `<unk>`. WikiText-103 has the same kind of articles, 50 times more of them. Use either `wikitext` or `wikitext103`, not both: WikiText-2's train articles are a subset of WikiText-103's, and their validation and test files are identical. The combined validation set is still 96% TinyStories, so its loss mostly measures TinyStories.

To add a source, write a Makefile rule that downloads it into `data/raw/<name>/`, define `<name>_TRAIN` and `<name>_VALID`, and add the name to `DATASETS`. Changing `DATASETS` means rerunning `make tokenizer vocab prepare pretrain`.

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (V2, the GPT-4-written stories): about 2.2 GB of short children's stories in simple, modern English, with a separate validation file. Stories are separated by a `<|endoftext|>` line, which stays in the text as an ordinary token so the model learns where a story ends. `make data` cleans the text while downloading (`pretraining/clean_text.py`): curly quotes become plain ones, and punctuation gets spaces around it, so `said,` becomes `said ,` and the word-level tokenizer doesn't learn `said`, `said,` and `said.` as separate words. `generate.py` cleans prompts the same way.

[WikiText-103](https://wikitext.smerity.com/) (raw) is about 540 MB of English Wikipedia articles, already split into train, validation and test, with punctuation already separated by spaces.

The combined corpus is big for this pipeline: about 626M tokens, and every step loads the whole corpus into memory (about 36 GB at peak). Building the tokenizer, vocabulary and prepared data takes about 1.5 minutes, plus a few minutes to download. One epoch is about 2.4 million training steps, so the default `make pretrain` (20,000 steps) sees under 1% of the data. Pass a larger `--steps` to use more. Only the train files build the tokenizer, and the test files are never encoded, so they stay untouched for a final evaluation.

If you rebuild the tokenizer, for example with a different `VOCAB_SIZE`, run `make vocab` and `make prepare` again: the token IDs in later steps depend on it. `make clean-data` deletes all four folders.
