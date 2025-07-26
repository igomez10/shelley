package tokenizer

import (
	"fmt"
	"testing"
)

func TestTokenizer(t *testing.T) {
	input := "Hello World"
	tkn := New()
	e := tkn.Encode(input)
	if len(e) == 0 {
		t.Error("unexpected lenght returned", e)
	}
	t.Log(e)
	fmt.Println(e)

	d := tkn.Decode(e)
	if d != input {
		t.Error("unexpected decoded message")
		t.Error(d)
	}
}
