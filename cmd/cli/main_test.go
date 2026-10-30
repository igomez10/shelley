package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/igomez10/shelley/pkg/dataset"
)

// run executes the CLI with args and stdin, returning what it wrote to stdout.
func run(t *testing.T, stdin string, args ...string) (string, error) {
	t.Helper()
	cmd := newRootCmd()
	var out bytes.Buffer
	cmd.Reader = strings.NewReader(stdin)
	cmd.Writer = &out
	cmd.ErrWriter = &out
	err := cmd.Run(context.Background(), append([]string{"shelley"}, args...))
	return out.String(), err
}

func TestInitThenTokenize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.gob")

	out, err := run(t, "", "init", "--tokenizer", path, "cherry", "apple", "banana", "apple")
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if !strings.Contains(out, "7 tokens") {
		t.Errorf("init output = %q; expected it to mention 7 tokens (4 special + 3 words)", out)
	}

	tests := []struct {
		name     string
		command  string
		stdin    string
		args     []string
		expected string
	}{
		{"args", "tokenize", "", []string{"apple", "cherry"}, "Tokenized output: [4 6]"},
		{"stdin", "tokenize", "banana\napple dog\n", nil, "Tokenized output: [5 4 0]"},
		{"alias", "to", "", []string{"banana"}, "Tokenized output: [5]"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			args := append([]string{test.command, "--tokenizer", path}, test.args...)
			out, err := run(t, test.stdin, args...)
			if err != nil {
				t.Fatalf("tokenize failed: %v", err)
			}
			if got := strings.TrimSpace(out); got != test.expected {
				t.Errorf("output = %q; expected %q", got, test.expected)
			}
		})
	}
}

func TestInitFromStdin(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.gob")

	if _, err := run(t, "the quick brown fox\njumps over the lazy dog\n", "init", "--tokenizer", path); err != nil {
		t.Fatalf("init failed: %v", err)
	}

	out, err := run(t, "", "tokenize", "--tokenizer", path, "the", "fox")
	if err != nil {
		t.Fatalf("tokenize failed: %v", err)
	}
	// vocabulary: 4 special tokens, then brown dog fox jumps lazy over quick the
	if got := strings.TrimSpace(out); got != "Tokenized output: [11 6]" {
		t.Errorf("output = %q; expected %q", got, "Tokenized output: [11 6]")
	}
}

func TestVocab(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.gob")
	if _, err := run(t, "", "init", "--tokenizer", path, "b", "a", ",", `"q"`, "a"); err != nil {
		t.Fatalf("init failed: %v", err)
	}

	// Sorted words: "q" (with quotes) < , < a < b. Tokens containing commas or
	// quotes are quoted, so the CSV stays parseable.
	expected := `token,id,count
<unk>,0,0
<bos>,1,0
<eos>,2,0
<pad>,3,0
"""q""",4,1
",",5,1
a,6,2
b,7,1
`

	tests := []struct {
		name string
		args []string
	}{
		{"stdout", nil},
		{"output file", []string{"--output", filepath.Join(dir, "vocab.csv")}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			out, err := run(t, "", append([]string{"vocab", "--tokenizer", path}, test.args...)...)
			if err != nil {
				t.Fatalf("vocab failed: %v", err)
			}

			got := out
			if test.args != nil {
				if !strings.Contains(out, "8 tokens") {
					t.Errorf("output = %q; expected it to mention 8 tokens", out)
				}
				b, err := os.ReadFile(test.args[1])
				if err != nil {
					t.Fatal(err)
				}
				got = string(b)
			}
			if got != expected {
				t.Errorf("CSV =\n%s\nexpected\n%s", got, expected)
			}
		})
	}
}

func TestInitVocabSize(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.gob")

	// 4 special tokens + the 2 most frequent words: the (3) and cat (2).
	if _, err := run(t, "", "init", "--tokenizer", path, "--vocab-size", "6", "the", "cat", "the", "sat", "cat", "the", "mat"); err != nil {
		t.Fatalf("init failed: %v", err)
	}

	out, err := run(t, "", "tokenize", "--tokenizer", path, "the", "cat", "sat", "mat")
	if err != nil {
		t.Fatalf("tokenize failed: %v", err)
	}
	if got := strings.TrimSpace(out); got != "Tokenized output: [5 4 0 0]" {
		t.Errorf("output = %q; expected %q", got, "Tokenized output: [5 4 0 0]")
	}

	csvPath := filepath.Join(dir, "vocab.csv")
	if _, err := run(t, "", "vocab", "--tokenizer", path, "--output", csvPath); err != nil {
		t.Fatalf("vocab failed: %v", err)
	}
	b, err := os.ReadFile(csvPath)
	if err != nil {
		t.Fatal(err)
	}
	// sat and mat are dropped, so <unk> counts their 2 occurrences.
	expected := "token,id,count\n<unk>,0,2\n<bos>,1,0\n<eos>,2,0\n<pad>,3,0\ncat,4,2\nthe,5,3\n"
	if string(b) != expected {
		t.Errorf("CSV =\n%s\nexpected\n%s", b, expected)
	}

	if _, err := run(t, "", "init", "--tokenizer", path, "--vocab-size", "2", "a"); err == nil {
		t.Error("expected an error for a vocabulary smaller than the special tokens")
	}
}

func TestPrepare(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.gob")
	if _, err := run(t, "", "init", "--tokenizer", path, "the", "cat", "sat", "on", "the", "mat"); err != nil {
		t.Fatalf("init failed: %v", err)
	}

	// vocabulary: 4 special tokens, then cat mat on sat the
	datasetPath := filepath.Join(dir, "train.txt")
	out, err := run(t, "the cat sat on the mat\nthe dog sat on the cat\n", "prepare", "--tokenizer", path, "--output", datasetPath)
	if err != nil {
		t.Fatalf("prepare failed: %v", err)
	}
	if !strings.Contains(out, "Encoded 12 tokens (1 unknown, 8.3%)") {
		t.Errorf("output = %q; expected it to report 12 tokens with 1 unknown", out)
	}
	if !strings.Contains(out, "inputs:  [8 4 7 6 8 5 8 0]  the cat sat on the mat the <unk>") ||
		!strings.Contains(out, "targets: [4 7 6 8 5 8 0 7]  cat sat on the mat the <unk> sat") {
		t.Errorf("output = %q; expected an example window and its shifted targets", out)
	}

	f, err := os.Open(datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	d, err := dataset.Read(f)
	if err != nil {
		t.Fatal(err)
	}
	expected := []int{8, 4, 7, 6, 8, 5, 8, 0, 7, 6, 8, 4}
	if !slices.Equal(d.IDs, expected) {
		t.Errorf("dataset = %v; expected %v", d.IDs, expected)
	}
}

func TestPrepareRequiresOutput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.gob")
	if _, err := run(t, "", "init", "--tokenizer", path, "a"); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if _, err := run(t, "a", "prepare", "--tokenizer", path); err == nil {
		t.Error("expected an error when --output is missing")
	}
}

func TestCommandsCreateOutputDirectories(t *testing.T) {
	dir := t.TempDir()
	// None of these directories exist yet.
	tokenizerPath := filepath.Join(dir, "tokenizer", "tokenizer.gob")
	csvPath := filepath.Join(dir, "tokenizer", "nested", "vocab.csv")
	datasetPath := filepath.Join(dir, "preparation", "train.txt")

	steps := [][]string{
		{"init", "--tokenizer", tokenizerPath, "the", "cat"},
		{"vocab", "--tokenizer", tokenizerPath, "--output", csvPath},
		{"prepare", "--tokenizer", tokenizerPath, "--output", datasetPath, "the", "cat"},
	}
	for _, args := range steps {
		if _, err := run(t, "", args...); err != nil {
			t.Fatalf("%s failed: %v", args[0], err)
		}
	}

	for _, path := range []string{tokenizerPath, csvPath, datasetPath} {
		if _, err := os.Stat(path); err != nil {
			t.Errorf("expected %s to exist: %v", path, err)
		}
	}
}

func TestVocabMissingTokenizerFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.gob")
	if _, err := run(t, "", "vocab", "--tokenizer", path); err == nil {
		t.Error("expected an error when the tokenizer file does not exist")
	}
}

func TestTokenizeMissingTokenizerFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.gob")
	if _, err := run(t, "", "tokenize", "--tokenizer", path, "hello"); err == nil {
		t.Error("expected an error when the tokenizer file does not exist")
	}
}
