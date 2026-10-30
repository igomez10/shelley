package vocab

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strconv"

	"github.com/igomez10/shelley/cmd/cli/flags"
	"github.com/igomez10/shelley/cmd/cli/output"
	"github.com/igomez10/shelley/pkg/tokenizer"
	"github.com/urfave/cli/v3"
)

var outputFlag = &cli.StringFlag{
	Name:    "output",
	Aliases: []string{"o"},
	Usage:   "file to write the CSV to (default: stdout)",
}

func GetCmd() *cli.Command {
	return &cli.Command{
		Name:      "vocab",
		Usage:     "export the tokenizer vocabulary as CSV (token,id,count)",
		UsageText: "vocab [--tokenizer path] [--output file.csv]",
		Flags: []cli.Flag{
			flags.VerboseFlag,
			flags.TokenizerFlag,
			outputFlag,
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
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

			path := cmd.String(outputFlag.Name)
			if path == "" {
				return writeCSV(cmd.Root().Writer, tkn)
			}

			out, err := output.Create(path)
			if err != nil {
				return fmt.Errorf("failed to create output file: %w", err)
			}
			defer out.Close()
			if err := writeCSV(out, tkn); err != nil {
				return err
			}
			if err := out.Close(); err != nil {
				return fmt.Errorf("failed to write output file: %w", err)
			}
			fmt.Fprintf(cmd.Root().Writer, "Vocabulary with %d tokens written to %s\n", len(tkn.StringToIntSlice), path)
			return nil
		},
	}
}

// writeCSV writes one row per token, ordered by ID, with how many times the
// token appeared in the corpus. encoding/csv quotes tokens that contain commas
// or quotes, such as the "," token itself.
func writeCSV(w io.Writer, tkn *tokenizer.Tokenizer) error {
	cw := csv.NewWriter(w)
	if err := cw.Write([]string{"token", "id", "count"}); err != nil {
		return fmt.Errorf("failed to write CSV: %w", err)
	}
	// The ID of a token is its position in the vocabulary slice.
	for id, token := range tkn.StringToIntSlice {
		if err := cw.Write([]string{token, strconv.Itoa(id), strconv.Itoa(tkn.Counts[id])}); err != nil {
			return fmt.Errorf("failed to write CSV: %w", err)
		}
	}
	cw.Flush()
	if err := cw.Error(); err != nil {
		return fmt.Errorf("failed to write CSV: %w", err)
	}
	return nil
}
