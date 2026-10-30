package tokenizer

import (
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"

	"golang.org/x/sync/errgroup"
)

// Special tokens are always the first IDs in the vocabulary, before any word
// from the corpus. They give the model real, non-negative IDs for things that
// are not words, so every ID can be used as a row in the embedding table.
const (
	UnkID = iota // a word that is not in the vocabulary
	BosID        // beginning of a sequence
	EosID        // end of a sequence
	PadID        // filler used to make sequences in a batch the same length
)

// SpecialTokens holds the text of each special token, indexed by its ID.
var SpecialTokens = []string{
	UnkID: "<unk>",
	BosID: "<bos>",
	EosID: "<eos>",
	PadID: "<pad>",
}

func New() *Tokenizer {
	t := &Tokenizer{
		StringToIntSlice: []string{},
		IntToString:      map[int][]byte{},
		Trie:             NewTrie(),
		Counts:           []int{},
	}
	return t
}

func NewFromFile(reader io.Reader) (*Tokenizer, error) {
	slog.Info("[NewFromFile] Start loading tokenizer from file...")
	defer func() {
		slog.Info("[NewFromFile] End loading tokenizer from file.")
	}()
	var t Tokenizer
	gobDecoder := gob.NewDecoder(reader)
	if err := gobDecoder.Decode(&t); err != nil {
		return nil, err
	}
	if t.Trie == nil {
		return nil, errors.New("tokenizer file has no trie")
	}
	if len(t.StringToIntSlice) < len(SpecialTokens) || !slices.Equal(t.StringToIntSlice[:len(SpecialTokens)], SpecialTokens) {
		return nil, errors.New("tokenizer file has no special tokens; rebuild it with init")
	}
	if len(t.Counts) != len(t.StringToIntSlice) {
		return nil, errors.New("tokenizer file has no token counts; rebuild it with init")
	}
	return &t, nil
}

type Tokenizer struct {
	StringToIntSlice []string
	IntToString      map[int][]byte
	Trie             *Trie
	// Counts holds how many times each token appeared in the corpus, indexed by
	// ID. Special tokens never appear in the corpus, so their count is 0.
	Counts []int
}

func (t *Tokenizer) Encode(input string) []int {
	slog.Debug("[Encode] Start encoding input...")
	defer func() {
		slog.Debug("[Encode] End encoding input.")
	}()

	if t.Trie == nil {
		panic("Tokenizer trie is nil")
	}

	splitted := strings.Fields(input)
	res := make([]int, len(splitted))
	for i := range splitted {
		node := t.Trie.findNode(splitted[i])
		if node == nil || !node.IsEnd {
			// Not found, or only a prefix of a known word
			res[i] = UnkID
			continue
		}

		res[i] = node.ID
	}

	return res
}

func (t *Tokenizer) Decode(input []int) string {
	res := strings.Builder{}
	for i := range input {
		if word, ok := t.IntToString[input[i]]; !ok {
			res.Write([]byte(SpecialTokens[UnkID]))
		} else {
			res.Write([]byte(word))
		}
		if i != len(input)-1 {
			res.Write([]byte(" "))
		}
	}
	return res.String()
}

// Init builds the vocabulary from the words in input. The special tokens take
// the first IDs, then each distinct word gets the next ID in sorted order. It
// also records how many times each word appears in Counts. If out is not nil, the
// tokenizer is gob-encoded to it so it can be loaded later with NewFromFile.
func (t *Tokenizer) Init(input string, out io.Writer) error {
	slog.Info("[Init] Initializing tokenizer...")
	defer func() {
		slog.Info("[Init] Tokenizer initialized.")
	}()

	slog.Debug("Start counting tokens...")
	counts := map[string]int{}
	for _, word := range strings.Fields(input) {
		counts[word]++
	}
	// A corpus word that looks like a special token must not get a second ID.
	for _, special := range SpecialTokens {
		delete(counts, special)
	}
	slog.Debug("Tokens counted.", slog.Int("distinct_words", len(counts)))

	slog.Debug("Start sorting tokens...")
	words := make([]string, 0, len(counts))
	for word := range counts {
		words = append(words, word)
	}
	slices.Sort(words)
	t.StringToIntSlice = append(slices.Clone(SpecialTokens), words...)
	slog.Debug("Tokens sorted.", slog.Int("vocab_size", len(t.StringToIntSlice)))

	t.Counts = make([]int, len(t.StringToIntSlice)) // special tokens stay at 0
	for i := len(SpecialTokens); i < len(t.StringToIntSlice); i++ {
		t.Counts[i] = counts[t.StringToIntSlice[i]]
	}

	t.IntToString = make(map[int][]byte, len(t.StringToIntSlice))
	t.Trie = NewTrie()

	// Both goroutines only read the vocabulary and each one writes to its own structure.
	g, _ := errgroup.WithContext(context.Background())
	g.SetLimit(10) // Limit concurrency to 10 goroutines
	g.Go(func() error {
		slog.Info("Start creating int to string mappings...")
		for i, word := range t.StringToIntSlice {
			t.IntToString[i] = []byte(word)
		}
		slog.Debug("int to string mappings created.")
		return nil
	})
	g.Go(func() error {
		slog.Info("Start creating trie structure...")
		// Special tokens stay out of the trie, so input text such as "<eos>"
		// encodes as unknown instead of injecting a control token.
		for i := len(SpecialTokens); i < len(t.StringToIntSlice); i++ {
			t.Trie.Insert(t.StringToIntSlice[i], i)
		}
		slog.Debug("Trie structure created.")
		return nil
	})

	if err := g.Wait(); err != nil {
		return fmt.Errorf("creating mappings: %w", err)
	}

	if out != nil {
		gobEncoder := gob.NewEncoder(out)
		if err := gobEncoder.Encode(t); err != nil {
			return fmt.Errorf("saving tokenizer: %w", err)
		}
	}
	return nil
}

// TrieNode represents each node in the Trie
type TrieNode struct {
	Children map[rune]*TrieNode
	IsEnd    bool
	ID       int
}

// Trie represents the whole trie structure
type Trie struct {
	Root *TrieNode
}

// NewTrie initializes a new Trie
func NewTrie() *Trie {
	return &Trie{
		Root: &TrieNode{
			Children: make(map[rune]*TrieNode),
			ID:       -1, // Root node does not represent a valid word
		},
	}
}

// Insert adds a word into the trie
func (t *Trie) Insert(word string, id int) {
	node := t.Root
	for _, ch := range word {
		if node.Children == nil {
			// gob decodes empty maps as nil, so nodes loaded from a file may not have one
			node.Children = make(map[rune]*TrieNode)
		}
		if _, exists := node.Children[ch]; !exists {
			node.Children[ch] = &TrieNode{
				Children: make(map[rune]*TrieNode),
			}
		}
		node = node.Children[ch]
	}
	node.ID = id
	node.IsEnd = true
}

// Search checks if a word is in the trie
func (t *Trie) Search(word string) bool {
	node := t.findNode(word)
	return node != nil && node.IsEnd
}

// StartsWith checks if any word in the trie starts with the given prefix
func (t *Trie) StartsWith(prefix string) bool {
	return t.findNode(prefix) != nil
}

// findNode returns the last node of the path that matches the input string
func (t *Trie) findNode(s string) *TrieNode {
	if t == nil {
		panic("Trie is nil")
	}
	node := t.Root
	for _, ch := range s {
		next, exists := node.Children[ch]
		if !exists {
			return nil
		}
		node = next
	}
	return node
}
