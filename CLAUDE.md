# CLAUDE.md

This file guides Claude Code when working in this repository.

## Project

shelley is an LLM built from scratch in Go, for learning. The author wants to understand each component by writing it by hand. Prefer clear, explicit code over clever or heavily optimized code, and don't pull in ML/numeric libraries unless asked. When explaining changes, explain the underlying concept too.

Module: `github.com/igomez10/shelley`. Go 1.24. Dependencies: `urfave/cli/v3`, `golang.org/x/sync/errgroup`.

## Commands

```sh
make build       # build the CLI (binary `cli` is gitignored)
make test        # all tests
make check       # gofmt check + go vet + tests with -race; run before committing
make data        # download WikiText-2 into data/ and copy its train split to data/corpus.txt
make tokenizer   # build a tokenizer from data/corpus.txt (override CORPUS=, TOKENIZER=, VOCAB_SIZE=; default 10000)
make tokenize TEXT="the cat sat"   # or FILE=path; hides logs
make vocab       # export the vocabulary as token,id,count CSV to data/vocab.csv
go test ./pkg/tokenizer -run TestTrie   # a single test
```

Run `make` to list all targets.

## Architecture

- `cmd/cli/main.go`: urfave/cli v3 root command. Each subcommand lives in `cmd/cli/commands/<name>/` and exposes `GetCmd() *cli.Command`.
- `cmd/cli/flags/`: shared flags (`VerboseFlag`, `TokenizerFlag`).
- `cmd/cli/input/`: `Read(cmd)` returns args joined by spaces, or stdin when there are no args. Commands read stdin and write output through `cmd.Root().Reader`/`Writer` so tests can swap them.
- `cmd/cli/main_test.go`: end-to-end CLI tests that run `newRootCmd()` against a temp tokenizer file.
- `pkg/tokenizer/`: word-level tokenizer backed by a rune-keyed trie. `Init` builds the vocabulary and gob-serializes the whole `Tokenizer` struct. `NewFromFile` deserializes it. Exported struct fields are part of the gob format, so renaming them breaks saved tokenizers.
- `init` builds a tokenizer and saves it, and `tokenize` loads it. Both use `--tokenizer`, which defaults to `/tmp/tokenizer.gob`.
- IDs 0–3 are special tokens (`UnkID`, `BosID`, `EosID`, `PadID`), then the sorted, deduplicated corpus words. Special tokens are not inserted into the trie. `Counts[id]` is how many times the token appeared in the corpus. `Init(corpus, vocabSize, out)` keeps the `vocabSize` most frequent tokens (special tokens included, 0 = no limit); the dropped words are added to `Counts[UnkID]`, the other special tokens stay 0. `NewFromFile` rejects files saved before special tokens or counts existed.

## Conventions

- Logging uses `log/slog`. Function-level logs are prefixed with `[FuncName]`, and the detailed ones use `slog.Debug`.
- Tests are table-driven.

## Gotchas

- gob encodes empty maps as nil, so trie nodes loaded from a file can have a nil `Children` map. `Trie.Insert` handles this case.
