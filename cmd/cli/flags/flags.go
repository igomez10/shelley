package flags

import "github.com/urfave/cli/v3"

var VerboseFlag = &cli.BoolFlag{
	Name:     "verbose",
	Aliases:  []string{"v"},
	Usage:    "enable verbose output",
	Required: false,
}
