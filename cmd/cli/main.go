package main

import (
	"context"
	"fmt"
	"os"

	initcmd "github.com/igomez10/shelley/cmd/cli/commands/init"
	"github.com/igomez10/shelley/cmd/cli/commands/prepare"
	"github.com/igomez10/shelley/cmd/cli/commands/tokenize"
	"github.com/igomez10/shelley/cmd/cli/commands/vocab"
	"github.com/urfave/cli/v3"
)

func main() {
	ctx := context.Background()
	cmd := newRootCmd()

	if err := cmd.Run(ctx, os.Args); err != nil {
		fmt.Println("Error running command:", err)
		os.Exit(1)
	}
}

func newRootCmd() *cli.Command {
	return &cli.Command{
		Name:      "shelley",
		Usage:     "cli to handle llm tasks",
		UsageText: "cli [global options] command [command options] [arguments...]",
		ArgsUsage: "[args and such]",
		Commands: []*cli.Command{
			tokenize.GetCmd(),
			initcmd.GetCmd(),
			vocab.GetCmd(),
			prepare.GetCmd(),
		},
	}
}
