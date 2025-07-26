package tokenize

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"

	"github.com/igomez10/shelley/cmd/cli/flags"
	"github.com/igomez10/shelley/pkg/tokenizer"
	"github.com/urfave/cli/v3"
)

func GetCmd() *cli.Command {
	return &cli.Command{
		Name:      "tokenize",
		Aliases:   []string{"to"},
		Usage:     "tokenize the input",
		UsageText: "tokenize - does the tokenizing",
		Flags: []cli.Flag{
			flags.VerboseFlag,
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			slog.Info("Tokenizing input...")
			var cmdInput []byte
			if cmd.NArg() > 0 {
				b := strings.Builder{}
				for i, arg := range cmd.Args().Slice() {
					b.WriteString(arg + " ")
					if i%1000 == 0 {
						slog.Info("Processed arguments...", slog.Int("count", i+1))
					}
				}
				cmdInput = []byte(b.String()[:b.Len()-1]) // Remove trailing space
			} else {
				if cmd.Reader == nil {
					return fmt.Errorf("no input provided")
				}
				input, err := io.ReadAll(cmd.Reader)
				if err != nil {
					return err
				}
				cmdInput = input
			}

			slog.Info("Creating new tokenizer...")
			gobfile, err := os.Open("/tmp/tokenizer.gob")
			if err != nil {
				return fmt.Errorf("failed to open tokenizer file: %w", err)
			}
			tkn, err := tokenizer.NewFromFile(gobfile)
			if err != nil {
				return fmt.Errorf("failed to load tokenizer: %w", err)
			}
			slog.Info("Encoding input...")
			res := tkn.Encode(string(cmdInput))
			slog.Info("Tokenization complete.")
			fmt.Println("Tokenized output:", res)

			return nil
		},
	}
}
