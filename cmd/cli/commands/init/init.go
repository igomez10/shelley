package initcmd

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/igomez10/shelley/cmd/cli/flags"
	"github.com/igomez10/shelley/cmd/cli/input"
	"github.com/igomez10/shelley/cmd/cli/output"
	"github.com/igomez10/shelley/pkg/tokenizer"
	"github.com/urfave/cli/v3"
)

var vocabSizeFlag = &cli.IntFlag{
	Name:  "vocab-size",
	Usage: "keep only the most frequent words, special tokens included (0 = no limit)",
}

func GetCmd() *cli.Command {
	return &cli.Command{
		Name:      "init",
		Usage:     "build a tokenizer from a corpus",
		UsageText: "init [--tokenizer path] [--vocab-size n] [corpus words...]  (reads stdin when no words are given)",
		Flags: []cli.Flag{
			flags.VerboseFlag,
			flags.TokenizerFlag,
			vocabSizeFlag,
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			slog.Info("Reading corpus...")
			corpus, err := input.Read(cmd)
			if err != nil {
				return fmt.Errorf("failed to read corpus: %w", err)
			}

			path := cmd.String(flags.TokenizerFlag.Name)
			out, err := output.Create(path)
			if err != nil {
				return fmt.Errorf("failed to create tokenizer file: %w", err)
			}
			defer out.Close()

			tkn := tokenizer.New()
			if err := tkn.Init(string(corpus), cmd.Int(vocabSizeFlag.Name), out); err != nil {
				return fmt.Errorf("failed to build tokenizer: %w", err)
			}
			if err := out.Close(); err != nil {
				return fmt.Errorf("failed to write tokenizer file: %w", err)
			}

			fmt.Fprintf(cmd.Root().Writer, "Tokenizer with %d tokens saved to %s\n", len(tkn.StringToIntSlice), path)
			return nil
		},
	}
}
