package tokenizer

import (
	"log/slog"
	"sort"
	"strings"
)

func New() *tokenizer {
	t := &tokenizer{
		StringToInt:      map[string]int{},
		StringToIntSlice: []string{},
		IntToString:      map[int][]byte{},
	}
	return t
}

type tokenizer struct {
	StringToInt      map[string]int
	StringToIntSlice []string
	IntToString      map[int][]byte
}

func (t *tokenizer) Encode(input string) []int {
	splitted := strings.Split(input, " ")
	t.StringToIntSlice = splitted
	slog.Info("Start sorting tokens...")
	t.StringToIntSlice = sort.StringSlice(t.StringToIntSlice)
	slog.Info("Tokens sorted.")
	slog.Info("Start creating string to int mappings...")
	for i := range t.StringToIntSlice {
		currentWord := t.StringToIntSlice[i]
		t.StringToInt[currentWord] = i
	}
	slog.Info("string to int mappings created.")
	slog.Info("Start creating int to string mappings...")
	for i := range t.StringToIntSlice {
		currentWord := t.StringToIntSlice[i]
		t.IntToString[i] = []byte(currentWord)
	}
	slog.Info("int to string mappings created.")

	slog.Info("Start converting input to int slice...")
	res := make([]int, len(splitted))
	for i := range len(splitted) {
		res[i] = t.StringToInt[splitted[i]]
	}
	slog.Info("Conversion complete.")

	return res
}

func (t *tokenizer) Decode(input []int) string {
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
