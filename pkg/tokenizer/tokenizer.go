package tokenizer

import (
	"sort"
	"strings"
)

func New() *tokenizer {
	t := &tokenizer{
		StringToInt:      map[string]int{},
		StringToIntSlice: []string{},
		IntToString:      map[int]string{},
	}
	return t
}

type tokenizer struct {
	StringToInt      map[string]int
	StringToIntSlice []string
	IntToString      map[int]string
}

func (t *tokenizer) Encode(input string) []int {
	splitted := strings.Split(input, " ")
	t.StringToIntSlice = splitted
	t.StringToIntSlice = sort.StringSlice(t.StringToIntSlice)
	for i := range t.StringToIntSlice {
		currentWord := t.StringToIntSlice[i]
		t.StringToInt[currentWord] = i
		t.IntToString[i] = currentWord
	}

	res := make([]int, len(splitted))
	for i := range len(splitted) {
		res[i] = t.StringToInt[splitted[i]]
	}

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
