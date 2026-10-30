package input

import (
	"io"
	"log/slog"
	"strings"

	"github.com/urfave/cli/v3"
)

// Read returns the command arguments joined by spaces, or the whole of stdin
// when no arguments were given.
func Read(cmd *cli.Command) ([]byte, error) {
	if cmd.NArg() > 0 {
		b := strings.Builder{}
		for i, arg := range cmd.Args().Slice() {
			if i > 0 {
				b.WriteString(" ")
			}
			b.WriteString(arg)
			if i%1000 == 0 {
				slog.Debug("Processed arguments...", slog.Int("count", i+1))
			}
		}
		return []byte(b.String()), nil
	}

	return io.ReadAll(cmd.Root().Reader)
}
