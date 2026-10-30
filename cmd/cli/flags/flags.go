package flags

import "github.com/urfave/cli/v3"

var VerboseFlag = &cli.BoolFlag{
	Name:     "verbose",
	Aliases:  []string{"v"},
	Usage:    "enable verbose output",
	Required: false,
}

const DefaultTokenizerPath = "/tmp/tokenizer.gob"

var TokenizerFlag = &cli.StringFlag{
	Name:    "tokenizer",
	Aliases: []string{"t"},
	Usage:   "path to the tokenizer file",
	Value:   DefaultTokenizerPath,
}
