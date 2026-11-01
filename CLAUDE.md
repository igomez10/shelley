# CLAUDE.md

This file guides Claude Code when working in this repository.

## Project

shelley is an LLM built from scratch in Go, for learning. The author wants to understand each component by writing it by hand. Prefer clear, explicit code over clever or heavily optimized code, and don't pull in ML/numeric libraries unless asked. When explaining changes, explain the underlying concept too.

Module: `github.com/igomez10/shelley`. Go 1.24. Dependencies: `urfave/cli/v3`, `golang.org/x/sync/errgroup`.

The repo root is also a uv project (`pyproject.toml`, `uv.lock`; Python 3.12, PyTorch, numpy). `pretraining/` holds the PyTorch reference models, written at the author's request. The Go code stays free of ML libraries. Run scripts with `uv run pretraining/<script>.py` from the repo root, since they use paths relative to it.

## Commands

```sh
make build       # build the CLI (binary `cli` is gitignored)
make test        # all tests
make check       # gofmt check + go vet + tests with -race; run before committing
make data        # step 1: download every source in DATASETS (default: tinystories wikitext103, ~2.7 GB) into data/raw/<source>/
make tokenizer   # step 2: build data/tokenizer/tokenizer.gob from the train split (override CORPUS=, TOKENIZER=, VOCAB_SIZE=; default 10000)
make tokenize TEXT="the cat sat"   # or FILE=path; hides logs
make vocab       # step 2: export the vocabulary as token,id,count CSV to data/tokenizer/vocab.csv
make prepare     # step 3: encode the train/val splits to data/preparation/{train,val}.txt
make pretrain    # step 4: train pretraining/train_mlp.py, saves the lowest-val-loss model to data/training/mlp.pt and stops early after --patience (default 3) checks without improvement; every run overwrites the checkpoint, so pass --checkpoint for test runs (ARGS="--steps 5000")
make generate PROMPT="the history of"   # step 5: pretraining/generate.py samples text from data/training/mlp.pt
go test ./pkg/tokenizer -run TestTrie   # a single test
```

Run `make` to list all targets.

## Architecture

- `cmd/cli/main.go`: urfave/cli v3 root command. Each subcommand lives in `cmd/cli/commands/<name>/` and exposes `GetCmd() *cli.Command`.
- `cmd/cli/flags/`: shared flags (`VerboseFlag`, `TokenizerFlag`).
- `cmd/cli/input/`: `Read(cmd)` returns args joined by spaces, or stdin when there are no args. Commands read stdin and write output through `cmd.Root().Reader`/`Writer` so tests can swap them.
- `cmd/cli/main_test.go`: end-to-end CLI tests that run `newRootCmd()` against a temp tokenizer file.
- `pkg/tokenizer/`: word-level tokenizer backed by a rune-keyed trie. `Init` builds the vocabulary and gob-serializes the whole `Tokenizer` struct. `NewFromFile` deserializes it. Exported struct fields are part of the gob format, so renaming them breaks saved tokenizers.
- `pkg/dataset/`: dataset files are plain text, one token ID per line (`Read` tolerates blank lines, surrounding spaces and CRLF). `Example(start, contextLen)` returns a window and its targets shifted by one; `Batch(rng, batchSize, contextLen)` samples random windows. Batches share memory with the dataset.
- `prepare` encodes a corpus (stdin) with the tokenizer and writes a dataset file (`--output`, required). The test split is deliberately not encoded.
- `init` builds a tokenizer and saves it, and `tokenize` loads it. All commands use `--tokenizer`, which defaults to `data/tokenizer/tokenizer.gob` (relative to the working directory). Commands that write files go through `cmd/cli/output.Create`, which creates parent directories.
- IDs 0–3 are special tokens (`UnkID`, `BosID`, `EosID`, `PadID`), then the sorted, deduplicated corpus words. Special tokens are not inserted into the trie. `Counts[id]` is how many times the token appeared in the corpus. `Init(corpus, vocabSize, out)` keeps the `vocabSize` most frequent tokens (special tokens included, 0 = no limit); the dropped words are added to `Counts[UnkID]`, the other special tokens stay 0. `NewFromFile` rejects files saved before special tokens or counts existed.

## Data layout

`DATASETS` (Makefile) lists the sources; their train files are `cat`-ed into one corpus for `init`/`prepare` and their valid files into one validation set. A source is a download rule plus `<name>_TRAIN`/`<name>_VALID` variables. `wikitext` and `wikitext103` overlap (same valid/test, train-2 ⊂ train-103), so never list both. `pretraining/clean_text.py` cleans TinyStories on download (plain quotes, spaces around punctuation) and `generate.py` reuses its `split_punctuation` for prompts. The default mix is ~626M tokens: every step loads it all into memory, so keep loading code vectorized (e.g. `np.fromfile` in `load_ids`) and avoid per-check work over the whole dataset.

`data/` (gitignored) has one folder per pipeline step: `raw/` (downloaded corpus), `tokenizer/` (tokenizer.gob, vocab.csv), `preparation/` (train.txt, val.txt), `training/` (mlp.pt: weights + the config needed to rebuild the model). `generate.py` imports `MLP` and paths from `train_mlp.py`, so model changes in one apply to both. New pipeline steps get their own `data/<step>/` folder, with paths defined at the top of the Makefile.

## Conventions

- Logging uses `log/slog`. Function-level logs are prefixed with `[FuncName]`, and the detailed ones use `slog.Debug`.
- Tests are table-driven.

## Gotchas

- gob encodes empty maps as nil, so trie nodes loaded from a file can have a nil `Children` map. `Trie.Insert` handles this case.
