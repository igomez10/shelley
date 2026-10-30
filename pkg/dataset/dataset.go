// Package dataset stores a tokenized corpus and cuts it into training examples
// for next-token prediction.
//
// A dataset file is the corpus encoded once with the tokenizer, as plain text:
// one token ID per line, in corpus order. Encoding ahead of time means
// training reads numbers from disk instead of re-tokenizing text on every
// pass, and keeping it as text means the file can be opened in any editor.
package dataset

import (
	"bufio"
	"fmt"
	"io"
	"math/rand/v2"
	"strconv"
	"strings"
)

// Dataset is a tokenized corpus: one long sequence of token IDs.
type Dataset struct {
	IDs []int
}

// Write saves ids to w in the dataset file format.
func Write(w io.Writer, ids []int) error {
	bw := bufio.NewWriter(w)
	for i, id := range ids {
		if id < 0 {
			return fmt.Errorf("token %d has negative ID %d", i, id)
		}
		if _, err := bw.WriteString(strconv.Itoa(id) + "\n"); err != nil {
			return fmt.Errorf("writing dataset: %w", err)
		}
	}
	if err := bw.Flush(); err != nil {
		return fmt.Errorf("writing dataset: %w", err)
	}
	return nil
}

// Read loads a dataset written by Write.
func Read(r io.Reader) (*Dataset, error) {
	ids := []int{}
	scanner := bufio.NewScanner(r)
	line := 0
	for scanner.Scan() {
		line++
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue // tolerate blank lines, e.g. a trailing newline added by an editor
		}
		id, err := strconv.Atoi(text)
		if err != nil || id < 0 {
			return nil, fmt.Errorf("line %d: %q is not a token ID", line, text)
		}
		ids = append(ids, id)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading dataset: %w", err)
	}
	return &Dataset{IDs: ids}, nil
}

// Example returns the training example that starts at position start: a
// context window of contextLen tokens, and the targets, which are the same
// window shifted one token ahead. targets[i] is the token that follows
// inputs[:i+1], so one window holds contextLen next-token predictions:
//
//	inputs:  the cat sat on
//	targets: cat sat on  the
func (d *Dataset) Example(start, contextLen int) (inputs, targets []int, err error) {
	if contextLen < 1 {
		return nil, nil, fmt.Errorf("context length must be at least 1, got %d", contextLen)
	}
	// The targets reach one token past the window, so the last valid start
	// leaves contextLen+1 tokens.
	if start < 0 || start+contextLen+1 > len(d.IDs) {
		return nil, nil, fmt.Errorf("example at %d with context length %d does not fit in a dataset of %d tokens", start, contextLen, len(d.IDs))
	}
	return d.IDs[start : start+contextLen], d.IDs[start+1 : start+contextLen+1], nil
}

// Batch returns batchSize examples whose windows start at random positions.
// Training on random windows, rather than walking the corpus in order, keeps
// each batch from being dominated by a single article.
//
// The returned slices share memory with the dataset, so callers must not
// modify them.
func (d *Dataset) Batch(rng *rand.Rand, batchSize, contextLen int) (inputs, targets [][]int, err error) {
	if batchSize < 1 {
		return nil, nil, fmt.Errorf("batch size must be at least 1, got %d", batchSize)
	}
	if contextLen < 1 {
		return nil, nil, fmt.Errorf("context length must be at least 1, got %d", contextLen)
	}
	starts := len(d.IDs) - contextLen // number of valid start positions
	if starts < 1 {
		return nil, nil, fmt.Errorf("dataset of %d tokens is too short for context length %d", len(d.IDs), contextLen)
	}

	inputs = make([][]int, batchSize)
	targets = make([][]int, batchSize)
	for b := range batchSize {
		inputs[b], targets[b], err = d.Example(rng.IntN(starts), contextLen)
		if err != nil {
			return nil, nil, err
		}
	}
	return inputs, targets, nil
}
