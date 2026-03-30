package odrl

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
)

// DIDData represents the structure of ~/.odrl/did.json
type DIDData struct {
	DID string `json:"did"`
}

// GetDID retrieves the current user's DID from ~/.odrl/did.json
func GetDID() string {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Debug("failed to get user home directory", "error", err)
		return ""
	}

	didPath := filepath.Join(home, ".odrl", "did.json")
	didFile, err := os.ReadFile(didPath)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Debug("failed to read ODRL did.json", "path", didPath, "error", err)
		}
		return ""
	}

	var data DIDData
	if err := json.Unmarshal(didFile, &data); err != nil {
		slog.Debug("failed to unmarshal ODRL did.json", "error", err)
		return ""
	}

	return data.DID
}
