BINARY    := cli
TOKENIZER ?= /tmp/tokenizer.gob
CORPUS    ?= data/corpus.txt

# Text for `make tokenize`, e.g. make tokenize TEXT="the cat sat".
# Exported so the recipe reads it from the environment, which keeps quotes and
# other shell characters in the text intact.
export TEXT
FILE ?=
VOCAB_CSV ?= data/vocab.csv

# WikiText-2: ~11 MB of English Wikipedia articles (raw text, train split)
WIKITEXT_URL := https://wikitext.smerity.com/wikitext-2-raw-v1.zip
WIKITEXT_DIR := data/wikitext-2-raw

.DEFAULT_GOAL := help

.PHONY: help build test test-race cover fmt fmt-check vet check data tokenizer tokenize vocab clean clean-data

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

data: $(WIKITEXT_DIR)/wiki.train.raw ## download WikiText-2 and use its train split as data/corpus.txt
	cp $(WIKITEXT_DIR)/wiki.train.raw data/corpus.txt

$(WIKITEXT_DIR)/wiki.train.raw:
	mkdir -p data
	curl -fL -o data/wikitext-2-raw-v1.zip $(WIKITEXT_URL)
	unzip -o -q data/wikitext-2-raw-v1.zip -d data
	rm data/wikitext-2-raw-v1.zip
	touch $@

tokenizer: build ## build a tokenizer from CORPUS into TOKENIZER
	./$(BINARY) init --tokenizer $(TOKENIZER) < $(CORPUS)

tokenize: build ## tokenize TEXT="..." or FILE=path with the tokenizer at TOKENIZER
ifneq ($(FILE),)
	@./$(BINARY) tokenize --tokenizer $(TOKENIZER) < $(FILE) 2>/dev/null
else ifneq ($(TEXT),)
	@./$(BINARY) tokenize --tokenizer $(TOKENIZER) "$$TEXT" 2>/dev/null
else
	@echo 'usage: make tokenize TEXT="some text"  or  make tokenize FILE=path/to/file.txt'; exit 1
endif

vocab: build ## export the vocabulary of TOKENIZER to VOCAB_CSV
	@./$(BINARY) vocab --tokenizer $(TOKENIZER) --output $(VOCAB_CSV) 2>/dev/null

clean: ## remove build and coverage output
	rm -f $(BINARY) coverage.out

clean-data: ## remove downloaded datasets
	rm -rf $(WIKITEXT_DIR) data/corpus.txt
