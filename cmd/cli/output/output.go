package output

import (
	"os"
	"path/filepath"
)

// Create creates or truncates the file at path, first creating its parent
// directories, so commands can write straight into data/<step>/ folders that
// do not exist yet.
func Create(path string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return os.Create(path)
}
