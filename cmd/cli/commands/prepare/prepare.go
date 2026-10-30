package prepare

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/igomez10/shelley/cmd/cli/flags"
	"github.com/igomez10/shelley/cmd/cli/input"
	"github.com/igomez10/shelley/cmd/cli/output"
	"github.com/igomez10/shelley/pkg/dataset"
	"github.com/igomez10/shelley/pkg/tokenizer"
	"github.com/urfave/cli/v3"
)

var outputFlag = &cli.StringFlag{
	Name:     "output",
	Aliases:  []string{"o"},
	Usage:    "file to write the encoded dataset to, e.g. data/preparation/train.txt",
	Required: true,
}

// exampleLen is the context length of the example printed after encoding.
const exampleLen = 8

func GetCmd() *cli.Command {
	return &cli.Command{
		Name:      "prepare",
		Usage:     "encode a corpus into a dataset of token IDs (one per line) for training",
		UsageText: "prepare [--tokenizer path] --output file.txt < corpus.txt",
		Flags: []cli.Flag{
			flags.VerboseFlag,
			flags.TokenizerFlag,
			outputFlag,
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			slog.Info("Reading corpus...")
			corpus, err := input.Read(cmd)
			if err != nil {
				return fmt.Errorf("failed to read corpus: %w", err)
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

			slog.Info("Encoding corpus...")
			ids := tkn.Encode(string(corpus))

			path := cmd.String(outputFlag.Name)
			out, err := output.Create(path)
			if err != nil {
				return fmt.Errorf("failed to create output file: %w", err)
			}
			defer out.Close()
			if err := dataset.Write(out, ids); err != nil {
				return err
			}
			if err := out.Close(); err != nil {
				return fmt.Errorf("failed to write output file: %w", err)
			}

			w := cmd.Root().Writer
			unknown := 0
			for _, id := range ids {
				if id == tokenizer.UnkID {
					unknown++
				}
			}
			unknownPct := 0.0
			if len(ids) > 0 {
				unknownPct = 100 * float64(unknown) / float64(len(ids))
			}
			fmt.Fprintf(w, "Encoded %d tokens (%d unknown, %.1f%%) to %s\n", len(ids), unknown, unknownPct, path)

			// Show what one training example looks like.
			d := &dataset.Dataset{IDs: ids}
			if inputs, targets, err := d.Example(0, exampleLen); err == nil {
				fmt.Fprintf(w, "Example with context length %d:\n", exampleLen)
				fmt.Fprintf(w, "  inputs:  %v  %s\n", inputs, tkn.Decode(inputs))
				fmt.Fprintf(w, "  targets: %v  %s\n", targets, tkn.Decode(targets))
			}
			return nil
		},
	}
}
