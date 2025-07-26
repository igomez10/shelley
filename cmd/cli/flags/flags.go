package flags

import "github.com/urfave/cli/v3"

var VerboseFlag = &cli.StringFlag{
	Name:     "verbose",
	Aliases:  []string{"v"},
	Usage:    "enable verbose output",
	Required: false,
}
