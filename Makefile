BINARY := cli

# data/ holds one folder per pipeline step, each built from the one before:
#   data/raw/          1. downloaded corpus (WikiText-2)
#   data/tokenizer/    2. tokenizer built from the train split, and its vocabulary
#   data/preparation/  3. train/validation splits encoded as token IDs
RAW_DIR         := data/raw
TOKENIZER_DIR   := data/tokenizer
PREPARATION_DIR := data/preparation

# WikiText-2: ~11 MB of English Wikipedia articles (raw text)
WIKITEXT_URL := https://wikitext.smerity.com/wikitext-2-raw-v1.zip
WIKITEXT_DIR := $(RAW_DIR)/wikitext-2-raw
WIKITEXT_TRAIN := $(WIKITEXT_DIR)/wiki.train.raw
WIKITEXT_VALID := $(WIKITEXT_DIR)/wiki.valid.raw

TOKENIZER ?= $(TOKENIZER_DIR)/tokenizer.gob
VOCAB_CSV ?= $(TOKENIZER_DIR)/vocab.csv
CORPUS    ?= $(WIKITEXT_TRAIN)
# Most frequent words to keep, special tokens included (0 = no limit)
VOCAB_SIZE ?= 10000

TRAIN_TXT := $(PREPARATION_DIR)/train.txt
VAL_TXT   := $(PREPARATION_DIR)/val.txt

# Text for `make tokenize`, e.g. make tokenize TEXT="the cat sat".
# Exported so the recipe reads it from the environment, which keeps quotes and
# other shell characters in the text intact.
export TEXT
FILE ?=

.DEFAULT_GOAL := help

.PHONY: help build test test-race cover fmt fmt-check vet check data tokenizer tokenize vocab prepare pretrain clean clean-data

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

data: $(WIKITEXT_TRAIN) ## step 1: download WikiText-2 into data/raw

$(WIKITEXT_TRAIN) $(WIKITEXT_VALID):
	mkdir -p $(RAW_DIR)
	curl -fL -o $(RAW_DIR)/wikitext-2-raw-v1.zip $(WIKITEXT_URL)
	unzip -o -q $(RAW_DIR)/wikitext-2-raw-v1.zip -d $(RAW_DIR)
	rm $(RAW_DIR)/wikitext-2-raw-v1.zip
	touch $(WIKITEXT_DIR)/*.raw

tokenizer: build $(CORPUS) ## step 2: build a tokenizer of VOCAB_SIZE tokens from CORPUS into TOKENIZER
	./$(BINARY) init --tokenizer $(TOKENIZER) --vocab-size $(VOCAB_SIZE) < $(CORPUS)

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

prepare: build $(WIKITEXT_TRAIN) $(WIKITEXT_VALID) ## step 3: encode the train and validation splits with TOKENIZER into data/preparation
	@./$(BINARY) prepare --tokenizer $(TOKENIZER) --output $(TRAIN_TXT) < $(WIKITEXT_TRAIN) 2>/dev/null
	@./$(BINARY) prepare --tokenizer $(TOKENIZER) --output $(VAL_TXT) < $(WIKITEXT_VALID) 2>/dev/null

pretrain: ## step 4: train the PyTorch MLP model, print losses, a summary and sample text (ARGS="--steps 5000")
	uv run pretraining/train_mlp.py $(ARGS)

clean: ## remove build and coverage output
	rm -f $(BINARY) coverage.out

clean-data: ## remove everything under data/raw, data/tokenizer, data/preparation
	rm -rf $(RAW_DIR) $(TOKENIZER_DIR) $(PREPARATION_DIR)
