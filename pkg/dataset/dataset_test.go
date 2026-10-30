package dataset

import (
	"bytes"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func TestWriteRead(t *testing.T) {
	tests := []struct {
		name string
		ids  []int
	}{
		{"empty", []int{}},
		{"small", []int{0, 1, 2, 3}},
		{"large IDs", []int{9999, 65535, 65536, 1<<32 - 1, 1 << 40}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := Write(&buf, test.ids); err != nil {
				t.Fatal(err)
			}

			d, err := Read(&buf)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(d.IDs, test.ids) {
				t.Errorf("Read = %v; expected %v", d.IDs, test.ids)
			}
		})
	}
}

func TestWrite_OneIDPerLine(t *testing.T) {
	var buf bytes.Buffer
	if err := Write(&buf, []int{467, 3397, 0}); err != nil {
		t.Fatal(err)
	}
	if got, expected := buf.String(), "467\n3397\n0\n"; got != expected {
		t.Errorf("file = %q; expected %q", got, expected)
	}
}

func TestWrite_NegativeID(t *testing.T) {
	if err := Write(&bytes.Buffer{}, []int{1, -1}); err == nil {
		t.Error("expected an error for a negative ID")
	}
}

func TestRead_Tolerant(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"no trailing newline", "1\n2\n3"},
		{"blank lines", "1\n\n2\n3\n\n"},
		{"surrounding spaces", " 1\n2 \n\t3\n"},
		{"windows line endings", "1\r\n2\r\n3\r\n"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			d, err := Read(strings.NewReader(test.input))
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(d.IDs, []int{1, 2, 3}) {
				t.Errorf("Read = %v; expected [1 2 3]", d.IDs)
			}
		})
	}
}

func TestRead_Invalid(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"word", "1\nhello\n"},
		{"negative", "1\n-5\n"},
		{"two IDs on one line", "1 2\n"},
		{"decimal", "1.5\n"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := Read(strings.NewReader(test.input)); err == nil {
				t.Errorf("Read(%q): expected an error", test.input)
			}
		})
	}
}

func TestExample(t *testing.T) {
	d := &Dataset{IDs: []int{10, 11, 12, 13, 14, 15}}

	tests := []struct {
		start, contextLen int
		inputs, targets   []int
	}{
		{0, 3, []int{10, 11, 12}, []int{11, 12, 13}},
		{2, 3, []int{12, 13, 14}, []int{13, 14, 15}}, // last valid start
		{0, 5, []int{10, 11, 12, 13, 14}, []int{11, 12, 13, 14, 15}},
		{4, 1, []int{14}, []int{15}},
	}

	for _, test := range tests {
		inputs, targets, err := d.Example(test.start, test.contextLen)
		if err != nil {
			t.Fatalf("Example(%d, %d): %v", test.start, test.contextLen, err)
		}
		if !slices.Equal(inputs, test.inputs) || !slices.Equal(targets, test.targets) {
			t.Errorf("Example(%d, %d) = %v, %v; expected %v, %v",
				test.start, test.contextLen, inputs, targets, test.inputs, test.targets)
		}
	}
}

func TestExample_OutOfRange(t *testing.T) {
	d := &Dataset{IDs: []int{10, 11, 12, 13, 14, 15}}

	tests := []struct {
		start, contextLen int
	}{
		{3, 3},  // targets would run past the end
		{-1, 2}, // negative start
		{0, 6},  // needs 7 tokens
		{0, 0},  // empty window
	}

	for _, test := range tests {
		if _, _, err := d.Example(test.start, test.contextLen); err == nil {
			t.Errorf("Example(%d, %d): expected an error", test.start, test.contextLen)
		}
	}
}

func TestBatch(t *testing.T) {
	ids := make([]int, 100)
	for i := range ids {
		ids[i] = i // each token is its own position, so windows are easy to check
	}
	d := &Dataset{IDs: ids}
	rng := rand.New(rand.NewPCG(1, 2))

	inputs, targets, err := d.Batch(rng, 32, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) != 32 || len(targets) != 32 {
		t.Fatalf("got %d inputs and %d targets; expected 32 each", len(inputs), len(targets))
	}

	for b := range inputs {
		if len(inputs[b]) != 8 || len(targets[b]) != 8 {
			t.Fatalf("example %d has %d inputs and %d targets; expected 8 each", b, len(inputs[b]), len(targets[b]))
		}
		for i := range inputs[b] {
			// Windows are contiguous, and each target is the next token.
			if i > 0 && inputs[b][i] != inputs[b][i-1]+1 {
				t.Errorf("example %d is not a contiguous window: %v", b, inputs[b])
			}
			if targets[b][i] != inputs[b][i]+1 {
				t.Errorf("example %d: target %d is not the token after input %d", b, targets[b][i], inputs[b][i])
			}
		}
	}
}

func TestBatch_Deterministic(t *testing.T) {
	d := &Dataset{IDs: make([]int, 1000)}
	for i := range d.IDs {
		d.IDs[i] = i
	}

	first, _, err := d.Batch(rand.New(rand.NewPCG(7, 7)), 4, 5)
	if err != nil {
		t.Fatal(err)
	}
	second, _, err := d.Batch(rand.New(rand.NewPCG(7, 7)), 4, 5)
	if err != nil {
		t.Fatal(err)
	}
	for b := range first {
		if !slices.Equal(first[b], second[b]) {
			t.Errorf("same seed gave different batches: %v vs %v", first, second)
		}
	}
}

func TestBatch_CoversWholeDataset(t *testing.T) {
	// With 5 tokens and context length 2 there are 3 valid starts: 0, 1 and 2.
	d := &Dataset{IDs: []int{0, 1, 2, 3, 4}}
	rng := rand.New(rand.NewPCG(3, 4))

	seen := map[int]bool{}
	inputs, _, err := d.Batch(rng, 200, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, in := range inputs {
		seen[in[0]] = true
	}
	for start := range 3 {
		if !seen[start] {
			t.Errorf("start %d never sampled", start)
		}
	}
	if len(seen) != 3 {
		t.Errorf("sampled starts %v; expected only 0, 1 and 2", seen)
	}
}

func TestBatch_Invalid(t *testing.T) {
	d := &Dataset{IDs: []int{0, 1, 2, 3}}
	rng := rand.New(rand.NewPCG(1, 1))

	tests := []struct {
		name                  string
		batchSize, contextLen int
	}{
		{"zero batch", 0, 2},
		{"zero context", 2, 0},
		{"context too long", 2, 4},
	}

	for _, test := range tests {
		if _, _, err := d.Batch(rng, test.batchSize, test.contextLen); err == nil {
			t.Errorf("%s: expected an error", test.name)
		}
	}
}
