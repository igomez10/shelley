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
		StringToInt:      map[string]int{},
		StringToIntSlice: []string{},
		IntToString:      map[int][]byte{},
	}
	return t
}

func NewWithData(stringToInt map[string]int, stringToIntSlice []string, intToString map[int][]byte) *Tokenizer {
	slog.Info("[NewWithData] Start creating tokenizer with provided data...")
	defer func() {
		slog.Info("[NewWithData] Tokenizer created with provided data.")
	}()

	t := &Tokenizer{
		StringToInt:      stringToInt,
		StringToIntSlice: stringToIntSlice,
		IntToString:      intToString,
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
	return &t, nil
}

type Tokenizer struct {
	StringToInt      map[string]int
	StringToIntSlice []string
	IntToString      map[int][]byte
}

func (t *Tokenizer) Encode(input string) []int {
	slog.Info("[Encode] Start encoding input...")
	defer func() {
		slog.Info("[Encode] End encoding input.")
	}()
	splitted := strings.Split(input, " ")
	res := make([]int, len(splitted))
	for i := range splitted {
		res[i] = t.StringToInt[splitted[i]]
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
	t.StringToInt = map[string]int{}
	t.StringToIntSlice = []string{}
	t.IntToString = map[int][]byte{}
	splitted := strings.Split(input, " ")
	t.StringToIntSlice = splitted
	slog.Debug("Start sorting tokens...")
	t.StringToIntSlice = sort.StringSlice(t.StringToIntSlice)
	slog.Debug("Tokens sorted.")
	errgroup, _ := errgroup.WithContext(context.Background())
	errgroup.SetLimit(10) // Limit concurrency to 10 goroutines

	errgroup.Go(func() error {
		slog.Info("Start creating string to int mappings...")
		for i := range t.StringToIntSlice {
			currentWord := t.StringToIntSlice[i]
			t.StringToInt[currentWord] = i
		}
		slog.Debug("string to int mappings created.")
		return nil
	})
	errgroup.Go(func() error {
		slog.Info("Start creating int to string mappings...")
		for i := range t.StringToIntSlice {
			currentWord := t.StringToIntSlice[i]
			t.IntToString[i] = []byte(currentWord)
		}
		slog.Debug("int to string mappings created.")
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
