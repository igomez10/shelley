package main

import (
	"context"
	"fmt"
	"os"

	"github.com/igomez10/shelley/cmd/cli/commands/tokenize"
	"github.com/urfave/cli/v3"
)

func main() {
	ctx := context.Background()
	cmd := &cli.Command{
		Name:      "shelley",
		Usage:     "cli to handle llm tasks",
		UsageText: "cli [global options] command [command options] [arguments...]",
		ArgsUsage: "[args and such]",
		Commands: []*cli.Command{
			tokenize.GetCmd(),
		},
	}

	if err := cmd.Run(ctx, os.Args); err != nil {
		fmt.Println("Error running command:", err)
		os.Exit(1)
	}
}
