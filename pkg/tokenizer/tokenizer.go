package tokenizer

import (
	"context"
	"encoding/gob"
	"io"
	"log/slog"
	"sort"
	"strings"

	"golang.org/x/sync/errgroup"
)

func New() *Tokenizer {
	t := &Tokenizer{
		StringToIntSlice: []string{},
		IntToString:      map[int][]byte{},
		Trie:             NewTrie(),
	}
	return t
}

func NewWithData(stringToInt map[string]int, stringToIntSlice []string, intToString map[int][]byte) *Tokenizer {
	slog.Info("[NewWithData] Start creating tokenizer with provided data...")
	defer func() {
		slog.Info("[NewWithData] Tokenizer created with provided data.")
	}()

	t := &Tokenizer{
		StringToIntSlice: stringToIntSlice,
		IntToString:      intToString,
		Trie:             NewTrie(),
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
		panic("Trie is nil in tokenizer")
	}
	return &t, nil
}

type Tokenizer struct {
	StringToIntSlice []string
	IntToString      map[int][]byte
	Trie             *Trie
}

func (t *Tokenizer) Encode(input string) []int {
	slog.Info("[Encode] Start encoding input...")
	defer func() {
		slog.Info("[Encode] End encoding input.")
	}()

	splitted := strings.Split(input, " ")
	res := make([]int, len(splitted))
	for i := range splitted {
		if t.Trie == nil {
			panic("Tokenizer trie is nil")
		}
		node := t.Trie.findNode(splitted[i])
		if node == nil {
			node = &TrieNode{IsEnd: false, ID: -1} // Not found, use -1 as unknown token
		}

		res[i] = node.ID
	}

	return res
}

func (t *Tokenizer) Decode(input []int) string {
	res := strings.Builder{}
	for i := range input {
		if word, ok := t.IntToString[input[i]]; !ok {
			res.Write([]byte("<unk>"))
		} else {
			res.Write([]byte(word))
		}
		if i != len(input)-1 {
			res.Write([]byte(" "))
		}
	}
	return res.String()
}

func (t *Tokenizer) Init(input string, out io.Writer) {
	slog.Info("[Init] Initializing tokenizer...")
	defer func() {
		slog.Info("[Init] Tokenizer initialized.")
	}()
	// t.StringToInt = map[string]int{}
	t.StringToIntSlice = []string{}
	t.IntToString = map[int][]byte{}
	splitted := strings.Split(input, " ")
	t.StringToIntSlice = splitted
	slog.Debug("Start sorting tokens...")
	t.StringToIntSlice = sort.StringSlice(t.StringToIntSlice)
	slog.Debug("Tokens sorted.")
	errgroup, _ := errgroup.WithContext(context.Background())
	errgroup.SetLimit(10) // Limit concurrency to 10 goroutines

	// errgroup.Go(func() error {
	// 	slog.Info("Start creating string to int mappings...")
	// 	for i := range t.StringToIntSlice {
	// 		currentWord := t.StringToIntSlice[i]
	// 		t.StringToInt[currentWord] = i
	// 	}
	// 	slog.Debug("string to int mappings created.")
	// 	return nil
	// })
	// errgroup.Go(func() error {
	// 	slog.Info("Start creating int to string mappings...")
	// 	for i := range t.StringToIntSlice {
	// 		currentWord := t.StringToIntSlice[i]
	// 		t.IntToString[i] = []byte(currentWord)
	// 	}
	// 	slog.Debug("int to string mappings created.")
	// 	return nil
	// })
	errgroup.Go(func() error {
		slog.Info("Start creating trie structure...")
		t.Trie = NewTrie()
		for i, word := range t.StringToIntSlice {
			t.Trie.Insert(word, i)
		}
		slog.Debug("Trie structure created.")
		return nil
	})

	if err := errgroup.Wait(); err != nil {
		slog.Error("Error creating mappings", "error", err)
		panic(err)
	}

	if out != nil {
		gobEncoder := gob.NewEncoder(out)
		if err := gobEncoder.Encode(t); err != nil {
			slog.Error("Error saving tokenizer to file", "error", err)
			panic(err)
		}
	}
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
