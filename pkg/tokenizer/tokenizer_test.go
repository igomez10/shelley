package tokenizer

import (
	"testing"
)

func TestTokenizer(t *testing.T) {
	// input := "Hello World"
	// tkn := New()
	// e := tkn.Encode(input)
	// if len(e) == 0 {
	// 	t.Error("unexpected lenght returned", e)
	// }
	// t.Log(e)
	// fmt.Println(e)

	// d := tkn.Decode(e)
	// if d != input {
	// 	t.Error("unexpected decoded message")
	// 	t.Error(d)
	// }
}

func TestTrie_InsertAndSearch(t *testing.T) {
	trie := NewTrie()

	trie.Insert("apple")
	trie.Insert("app")
	trie.Insert("application")

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

	trie.Insert("dog")
	trie.Insert("dove")
	trie.Insert("door")

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
