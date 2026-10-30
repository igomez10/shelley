package tokenizer

import (
	"bytes"
	"encoding/gob"
	"slices"
	"testing"
)

func TestTokenizer_EncodeDecode(t *testing.T) {
	tkn := New()
	if err := tkn.Init("hello world hello again", nil); err != nil {
		t.Fatal(err)
	}

	input := "hello world"
	e := tkn.Encode(input)
	if len(e) != 2 {
		t.Fatalf("Encode(%q) = %v; expected 2 tokens", input, e)
	}
	for _, id := range e {
		if id == UnkID {
			t.Fatalf("Encode(%q) = %v; expected only known tokens", input, e)
		}
	}

	if d := tkn.Decode(e); d != input {
		t.Errorf("Decode(%v) = %q; expected %q", e, d, input)
	}
}

func TestTokenizer_InitSortsAndDeduplicates(t *testing.T) {
	tkn := New()
	if err := tkn.Init("cherry apple banana apple cherry", nil); err != nil {
		t.Fatal(err)
	}

	expected := []string{"<unk>", "<bos>", "<eos>", "<pad>", "apple", "banana", "cherry"}
	if !slices.Equal(tkn.StringToIntSlice, expected) {
		t.Fatalf("vocabulary = %v; expected %v", tkn.StringToIntSlice, expected)
	}
	for id, word := range expected {
		if got := string(tkn.IntToString[id]); got != word {
			t.Errorf("IntToString[%d] = %q; expected %q", id, got, word)
		}
	}
	if got := tkn.Encode("apple banana cherry"); !slices.Equal(got, []int{4, 5, 6}) {
		t.Errorf("Encode = %v; expected [4 5 6]", got)
	}
}

func TestTokenizer_InitCounts(t *testing.T) {
	tests := []struct {
		name     string
		corpus   string
		expected map[string]int
	}{
		{"repeated words", "cherry apple banana apple cherry apple", map[string]int{"apple": 3, "banana": 1, "cherry": 2}},
		{"case sensitive", "The the THE the", map[string]int{"THE": 1, "The": 1, "the": 2}},
		{"whitespace", "a\na\ta  b\n", map[string]int{"a": 3, "b": 1}},
		// Special token text in the corpus is dropped, so it is not counted.
		{"special token text", "hello <eos> hello", map[string]int{"hello": 2}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tkn := New()
			if err := tkn.Init(test.corpus, nil); err != nil {
				t.Fatal(err)
			}
			if len(tkn.Counts) != len(tkn.StringToIntSlice) {
				t.Fatalf("len(Counts) = %d; expected %d", len(tkn.Counts), len(tkn.StringToIntSlice))
			}
			for id := range SpecialTokens {
				if tkn.Counts[id] != 0 {
					t.Errorf("Counts[%d] (%s) = %d; expected 0", id, SpecialTokens[id], tkn.Counts[id])
				}
			}
			for word, count := range test.expected {
				id := tkn.Encode(word)[0]
				if id == UnkID {
					t.Fatalf("%q not in vocabulary", word)
				}
				if tkn.Counts[id] != count {
					t.Errorf("Counts[%q] = %d; expected %d", word, tkn.Counts[id], count)
				}
			}
			if words := len(tkn.StringToIntSlice) - len(SpecialTokens); words != len(test.expected) {
				t.Errorf("vocabulary has %d words; expected %d", words, len(test.expected))
			}
		})
	}
}

func TestTokenizer_InitSplitsOnAnyWhitespace(t *testing.T) {
	tkn := New()
	if err := tkn.Init("one\ntwo\tthree  four\n", nil); err != nil {
		t.Fatal(err)
	}

	expected := []string{"four", "one", "three", "two"}
	if got := tkn.StringToIntSlice[len(SpecialTokens):]; !slices.Equal(got, expected) {
		t.Errorf("vocabulary = %v; expected words %v", tkn.StringToIntSlice, expected)
	}
}

func TestTokenizer_EncodeUnknown(t *testing.T) {
	tkn := New()
	if err := tkn.Init("apple banana", nil); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		input    string
		expected []int
	}{
		{"cherry", []int{UnkID}},
		{"app", []int{UnkID}},    // prefix of a known word
		{"apples", []int{UnkID}}, // known word is a prefix of the input
		{"apple app", []int{4, UnkID}},
		{"", []int{}},
	}

	for _, test := range tests {
		if got := tkn.Encode(test.input); !slices.Equal(got, test.expected) {
			t.Errorf("Encode(%q) = %v; expected %v", test.input, got, test.expected)
		}
	}
}

func TestTokenizer_DecodeUnknown(t *testing.T) {
	tkn := New()
	if err := tkn.Init("apple", nil); err != nil {
		t.Fatal(err)
	}

	if got := tkn.Decode([]int{4, -1, 42}); got != "apple <unk> <unk>" {
		t.Errorf("Decode = %q; expected %q", got, "apple <unk> <unk>")
	}
}

func TestTokenizer_SaveAndLoad(t *testing.T) {
	tkn := New()
	var buf bytes.Buffer
	if err := tkn.Init("the cat sat on the mat", &buf); err != nil {
		t.Fatal(err)
	}

	loaded, err := NewFromFile(&buf)
	if err != nil {
		t.Fatal(err)
	}

	input := "the cat sat on the mat"
	want := tkn.Encode(input)
	if got := loaded.Encode(input); !slices.Equal(got, want) {
		t.Errorf("loaded Encode(%q) = %v; expected %v", input, got, want)
	}
	if got := loaded.Decode(want); got != input {
		t.Errorf("loaded Decode(%v) = %q; expected %q", want, got, input)
	}
	if got := loaded.Encode("dog"); !slices.Equal(got, []int{UnkID}) {
		t.Errorf("loaded Encode(%q) = %v; expected [%d]", "dog", got, UnkID)
	}

	if !slices.Equal(loaded.Counts, tkn.Counts) {
		t.Errorf("loaded Counts = %v; expected %v", loaded.Counts, tkn.Counts)
	}

	// The loaded trie must still accept new words.
	loaded.Trie.Insert("dog", 99)
	if got := loaded.Encode("dog"); !slices.Equal(got, []int{99}) {
		t.Errorf("Encode(%q) after Insert = %v; expected [99]", "dog", got)
	}
}

func TestTokenizer_SpecialTokens(t *testing.T) {
	tkn := New()
	if err := tkn.Init("hello world", nil); err != nil {
		t.Fatal(err)
	}

	// Special tokens keep fixed IDs and decode to their text.
	for id, text := range SpecialTokens {
		if got := tkn.Decode([]int{id}); got != text {
			t.Errorf("Decode([%d]) = %q; expected %q", id, got, text)
		}
	}

	// Callers wrap a sequence with BOS/EOS themselves.
	seq := append(append([]int{BosID}, tkn.Encode("hello world")...), EosID)
	if got := tkn.Decode(seq); got != "<bos> hello world <eos>" {
		t.Errorf("Decode(%v) = %q; expected %q", seq, got, "<bos> hello world <eos>")
	}

	// Every ID Encode returns is a valid, non-negative index into the vocabulary.
	for _, id := range tkn.Encode("hello there world") {
		if id < 0 || id >= len(tkn.StringToIntSlice) {
			t.Errorf("Encode returned ID %d outside the vocabulary of size %d", id, len(tkn.StringToIntSlice))
		}
	}
}

func TestTokenizer_SpecialTokensInText(t *testing.T) {
	tkn := New()
	// The corpus itself contains special token text.
	if err := tkn.Init("hello <eos> world <unk>", nil); err != nil {
		t.Fatal(err)
	}

	// They must not get a second ID as ordinary words.
	expected := []string{"<unk>", "<bos>", "<eos>", "<pad>", "hello", "world"}
	if !slices.Equal(tkn.StringToIntSlice, expected) {
		t.Fatalf("vocabulary = %v; expected %v", tkn.StringToIntSlice, expected)
	}

	// Input text can't produce control tokens: it encodes as unknown.
	tests := []struct {
		input    string
		expected []int
	}{
		{"<eos>", []int{UnkID}},
		{"<bos> hello", []int{UnkID, 4}},
		{"<pad>", []int{UnkID}},
	}
	for _, test := range tests {
		if got := tkn.Encode(test.input); !slices.Equal(got, test.expected) {
			t.Errorf("Encode(%q) = %v; expected %v", test.input, got, test.expected)
		}
	}
}

func TestTokenizer_EmptyCorpus(t *testing.T) {
	tkn := New()
	if err := tkn.Init("", nil); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(tkn.StringToIntSlice, SpecialTokens) {
		t.Errorf("vocabulary = %v; expected only special tokens %v", tkn.StringToIntSlice, SpecialTokens)
	}
	if got := tkn.Encode("anything"); !slices.Equal(got, []int{UnkID}) {
		t.Errorf("Encode = %v; expected [%d]", got, UnkID)
	}
}

func TestNewFromFile_WithoutSpecialTokens(t *testing.T) {
	// A tokenizer saved before special tokens existed: words start at ID 0.
	old := &Tokenizer{
		StringToIntSlice: []string{"apple", "banana"},
		IntToString:      map[int][]byte{0: []byte("apple"), 1: []byte("banana")},
		Trie:             NewTrie(),
	}
	old.Trie.Insert("apple", 0)
	old.Trie.Insert("banana", 1)

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(old); err != nil {
		t.Fatal(err)
	}
	if _, err := NewFromFile(&buf); err == nil {
		t.Error("expected an error for a tokenizer file without special tokens")
	}
}

func TestNewFromFile_WithoutCounts(t *testing.T) {
	// A tokenizer saved before counts existed.
	old := New()
	if err := old.Init("apple banana", nil); err != nil {
		t.Fatal(err)
	}
	old.Counts = nil

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(old); err != nil {
		t.Fatal(err)
	}
	if _, err := NewFromFile(&buf); err == nil {
		t.Error("expected an error for a tokenizer file without counts")
	}
}

func TestNewFromFile_InvalidData(t *testing.T) {
	if _, err := NewFromFile(bytes.NewBufferString("not a gob")); err == nil {
		t.Error("expected an error for invalid data")
	}
}

func TestTrie_InsertAndSearch(t *testing.T) {
	trie := NewTrie()

	trie.Insert("apple", 0)
	trie.Insert("app", 1)
	trie.Insert("application", 2)

	tests := []struct {
		word     string
		expected bool
	}{
		{"apple", true},
		{"app", true},
		{"application", true},
		{"appl", false},
		{"banana", false},
	}

	for _, test := range tests {
		if result := trie.Search(test.word); result != test.expected {
			t.Errorf("Search(%q) = %v; expected %v", test.word, result, test.expected)
		}
	}
}

func TestTrie_StartsWith(t *testing.T) {
	trie := NewTrie()

	trie.Insert("dog", 0)
	trie.Insert("dove", 1)
	trie.Insert("door", 2)

	prefixTests := []struct {
		prefix   string
		expected bool
	}{
		{"do", true},
		{"dog", true},
		{"dov", true},
		{"cat", false},
		{"dor", false},
	}

	for _, test := range prefixTests {
		if result := trie.StartsWith(test.prefix); result != test.expected {
			t.Errorf("StartsWith(%q) = %v; expected %v", test.prefix, result, test.expected)
		}
	}
}

func TestTrie_UnicodeWords(t *testing.T) {
	trie := NewTrie()
	trie.Insert("niño", 0)
	trie.Insert("日本", 1)

	for _, word := range []string{"niño", "日本"} {
		if !trie.Search(word) {
			t.Errorf("Search(%q) = false; expected true", word)
		}
	}
	if trie.Search("nino") {
		t.Error(`Search("nino") = true; expected false`)
	}
}
