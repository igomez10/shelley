package tokenize

import (
	"context"
	"fmt"
	"io"

	"github.com/igomez10/shelley/cmd/cli/flags"
	"github.com/igomez10/shelley/pkg/tokenizer"
	"github.com/urfave/cli/v3"
)

func GetCmd() *cli.Command {
	return &cli.Command{
		Name:      "tokenize",
		Aliases:   []string{"to"},
		Category:  "motion",
		Usage:     "tokenize the input",
		UsageText: "tokenize - does the tokenizing",
		Flags: []cli.Flag{
			flags.VerboseFlag,
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			fmt.Println("Tokenizing input...")
			if cmd.Reader == nil {
				return fmt.Errorf("no input provided")
			}
			tkn := tokenizer.New()
			input, err := io.ReadAll(cmd.Reader)
			if err != nil {
				return err
			}
			tkn.Encode(string(input))

			if cmd.Bool(flags.VerboseFlag.Name) {
				encoded := tkn.Encode(string(input))
				decoded := tkn.Decode(encoded)
				fmt.Printf("Encoded: \n%v\n", encoded)
				fmt.Printf("Decoded: \n%s\n", decoded)
			}

			return nil
		},
	}
}
