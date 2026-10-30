package tokenize

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/igomez10/shelley/cmd/cli/flags"
	"github.com/igomez10/shelley/cmd/cli/input"
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
			flags.TokenizerFlag,
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			slog.Info("Tokenizing input...")
			cmdInput, err := input.Read(cmd)
			if err != nil {
				return fmt.Errorf("failed to read input: %w", err)
			}

			slog.Info("Loading tokenizer...")
			gobfile, err := os.Open(cmd.String(flags.TokenizerFlag.Name))
			if err != nil {
				return fmt.Errorf("failed to open tokenizer file (run `init` first?): %w", err)
			}
			defer gobfile.Close()
			tkn, err := tokenizer.NewFromFile(gobfile)
			if err != nil {
				return fmt.Errorf("failed to load tokenizer: %w", err)
			}
			slog.Info("Encoding input...")
			res := tkn.Encode(string(cmdInput))
			slog.Info("Tokenization complete.")
			fmt.Fprintln(cmd.Root().Writer, "Tokenized output:", res)

			return nil
		},
	}
}
