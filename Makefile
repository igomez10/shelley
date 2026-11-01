BINARY := cli

# data/ holds one folder per pipeline step, each built from the one before:
#   data/raw/          1. downloaded corpora, one folder per source in DATASETS
#   data/tokenizer/    2. tokenizer built from the train split, and its vocabulary
#   data/preparation/  3. train/validation splits encoded as token IDs
#   data/training/     4. the trained model with the lowest validation loss
RAW_DIR         := data/raw
TOKENIZER_DIR   := data/tokenizer
PREPARATION_DIR := data/preparation

# The sources the pipeline trains on. Their train files are concatenated into
# one corpus for the tokenizer and `prepare`, and their validation files into
# one validation set. Changing the list means rerunning
# `make tokenizer vocab prepare pretrain`.
#
# To add a source: write a rule that downloads it into $(RAW_DIR)/<name>/ and
# define <name>_TRAIN and <name>_VALID, like the two below.
DATASETS ?= tinystories wikitext103

# TinyStories (V2, GPT-4 written): ~2.2 GB of short stories in simple, modern
# English. Cleaned on download by pretraining/clean_text.py.
TINYSTORIES_URL := https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
TINYSTORIES_DIR := $(RAW_DIR)/tinystories
TINYSTORIES_TRAIN := $(TINYSTORIES_DIR)/train.txt
TINYSTORIES_VALID := $(TINYSTORIES_DIR)/valid.txt
tinystories_TRAIN := $(TINYSTORIES_TRAIN)
tinystories_VALID := $(TINYSTORIES_VALID)

# WikiText-2: ~11 MB of English Wikipedia articles (raw text)
WIKITEXT_URL := https://wikitext.smerity.com/wikitext-2-raw-v1.zip
WIKITEXT_DIR := $(RAW_DIR)/wikitext-2-raw
WIKITEXT_TRAIN := $(WIKITEXT_DIR)/wiki.train.raw
WIKITEXT_VALID := $(WIKITEXT_DIR)/wiki.valid.raw
wikitext_TRAIN := $(WIKITEXT_TRAIN)
wikitext_VALID := $(WIKITEXT_VALID)

# WikiText-103: ~540 MB, the same kind of articles as WikiText-2 (whose train
# articles are a subset of these, and whose valid/test files are identical), so
# use one or the other in DATASETS, not both.
WIKITEXT103_URL := https://wikitext.smerity.com/wikitext-103-raw-v1.zip
WIKITEXT103_DIR := $(RAW_DIR)/wikitext-103-raw
wikitext103_TRAIN := $(WIKITEXT103_DIR)/wiki.train.raw
wikitext103_VALID := $(WIKITEXT103_DIR)/wiki.valid.raw

$(foreach d,$(DATASETS),$(if $($(d)_TRAIN),,$(error unknown dataset "$(d)" in DATASETS; known: tinystories wikitext wikitext103)))
TRAIN_RAW := $(foreach d,$(DATASETS),$($(d)_TRAIN))
VALID_RAW := $(foreach d,$(DATASETS),$($(d)_VALID))

TOKENIZER ?= $(TOKENIZER_DIR)/tokenizer.gob
VOCAB_CSV ?= $(TOKENIZER_DIR)/vocab.csv
CORPUS    ?= $(TRAIN_RAW)
# Most frequent words to keep, special tokens included (0 = no limit)
VOCAB_SIZE ?= 10000

TRAIN_TXT := $(PREPARATION_DIR)/train.txt
VAL_TXT   := $(PREPARATION_DIR)/val.txt

# Text for `make tokenize`, e.g. make tokenize TEXT="the cat sat".
# Exported so the recipe reads it from the environment, which keeps quotes and
# other shell characters in the text intact.
export TEXT
# Prompt for `make generate`, e.g. make generate PROMPT="the history of".
# Exported for the same reason as TEXT.
export PROMPT
FILE ?=

.DEFAULT_GOAL := help

.PHONY: help build test test-race cover fmt fmt-check vet check data tokenizer tokenize vocab prepare pretrain generate clean clean-data

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

build: ## build the CLI binary
	go build -o $(BINARY) ./cmd/cli

test: ## run all tests
	go test ./...

test-race: ## run all tests with the race detector
	go test -race ./...

cover: ## run tests and open a coverage report
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out

fmt: ## format all Go files
	gofmt -w .

fmt-check: ## fail if any Go file is not formatted
	@test -z "$$(gofmt -l .)" || (echo "unformatted files:"; gofmt -l .; exit 1)

vet: ## run go vet
	go vet ./...

check: fmt-check vet test-race ## run formatting check, vet and race tests

data: $(TRAIN_RAW) $(VALID_RAW) ## step 1: download every source in DATASETS into data/raw

# Cleaned while downloading; written to a .tmp file first so a failed download
# never leaves a half-written file that make would treat as done.
$(TINYSTORIES_TRAIN): REMOTE := TinyStoriesV2-GPT4-train.txt
$(TINYSTORIES_VALID): REMOTE := TinyStoriesV2-GPT4-valid.txt
$(TINYSTORIES_TRAIN) $(TINYSTORIES_VALID):
	mkdir -p $(TINYSTORIES_DIR)
	curl -fL $(TINYSTORIES_URL)/$(REMOTE) | uv run pretraining/clean_text.py > $@.tmp
	mv $@.tmp $@

$(WIKITEXT_TRAIN) $(WIKITEXT_VALID):
	mkdir -p $(RAW_DIR)
	curl -fL -o $(RAW_DIR)/wikitext-2-raw-v1.zip $(WIKITEXT_URL)
	unzip -o -q $(RAW_DIR)/wikitext-2-raw-v1.zip -d $(RAW_DIR)
	rm $(RAW_DIR)/wikitext-2-raw-v1.zip
	touch $(WIKITEXT_DIR)/*.raw

$(wikitext103_TRAIN) $(wikitext103_VALID):
	mkdir -p $(RAW_DIR)
	curl -fL -o $(RAW_DIR)/wikitext-103-raw-v1.zip $(WIKITEXT103_URL)
	unzip -o -q $(RAW_DIR)/wikitext-103-raw-v1.zip -d $(RAW_DIR)
	rm $(RAW_DIR)/wikitext-103-raw-v1.zip
	touch $(WIKITEXT103_DIR)/*.raw

tokenizer: build $(CORPUS) ## step 2: build a tokenizer of VOCAB_SIZE tokens from CORPUS (all train files) into TOKENIZER
	cat $(CORPUS) | ./$(BINARY) init --tokenizer $(TOKENIZER) --vocab-size $(VOCAB_SIZE)

tokenize: build ## tokenize TEXT="..." or FILE=path with the tokenizer at TOKENIZER
ifneq ($(FILE),)
	@./$(BINARY) tokenize --tokenizer $(TOKENIZER) < $(FILE) 2>/dev/null
else ifneq ($(TEXT),)
	@./$(BINARY) tokenize --tokenizer $(TOKENIZER) "$$TEXT" 2>/dev/null
else
	@echo 'usage: make tokenize TEXT="some text"  or  make tokenize FILE=path/to/file.txt'; exit 1
endif

vocab: build ## step 2: export the vocabulary of TOKENIZER to VOCAB_CSV
	@./$(BINARY) vocab --tokenizer $(TOKENIZER) --output $(VOCAB_CSV) 2>/dev/null

prepare: build $(TRAIN_RAW) $(VALID_RAW) ## step 3: encode all train and validation files with TOKENIZER into data/preparation
	@cat $(TRAIN_RAW) | ./$(BINARY) prepare --tokenizer $(TOKENIZER) --output $(TRAIN_TXT) 2>/dev/null
	@cat $(VALID_RAW) | ./$(BINARY) prepare --tokenizer $(TOKENIZER) --output $(VAL_TXT) 2>/dev/null

pretrain: ## step 4: train the PyTorch MLP model, saving the best one to data/training/mlp.pt (ARGS="--steps 5000")
	uv run pretraining/train_mlp.py $(ARGS)

clean: ## remove build and coverage output
	rm -f $(BINARY) coverage.out

generate: ## step 5: generate text with the trained model (PROMPT="the history of", ARGS="--tokens 100")
	uv run pretraining/generate.py $(if $(PROMPT),--prompt "$$PROMPT") $(ARGS)

clean-data: ## remove everything under data/raw, data/tokenizer, data/preparation, data/training
	rm -rf $(RAW_DIR) $(TOKENIZER_DIR) $(PREPARATION_DIR) data/training
