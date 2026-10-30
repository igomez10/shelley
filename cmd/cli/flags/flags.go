package flags

import "github.com/urfave/cli/v3"

var VerboseFlag = &cli.BoolFlag{
	Name:     "verbose",
	Aliases:  []string{"v"},
	Usage:    "enable verbose output",
	Required: false,
}

// DefaultTokenizerPath is relative to the directory the CLI runs from, which
// is normally the repository root.
const DefaultTokenizerPath = "data/tokenizer/tokenizer.gob"

var TokenizerFlag = &cli.StringFlag{
	Name:    "tokenizer",
	Aliases: []string{"t"},
	Usage:   "path to the tokenizer file",
	Value:   DefaultTokenizerPath,
}
