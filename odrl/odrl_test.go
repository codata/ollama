package odrl

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGetDID(t *testing.T) {
	// Setup a temporary HOME directory
	tmpHome := t.TempDir()
	t.Setenv("HOME", tmpHome)

	odrlDir := filepath.Join(tmpHome, ".odrl")
	if err := os.MkdirAll(odrlDir, 0755); err != nil {
		t.Fatal(err)
	}
	didPath := filepath.Join(odrlDir, "did.json")

	testDID := "did:test:123"
	data := DIDData{DID: testDID}
	dataBytes, _ := json.Marshal(data)
	if err := os.WriteFile(didPath, dataBytes, 0644); err != nil {
		t.Fatal(err)
	}

	did := GetDID()
	if did != testDID {
		t.Errorf("expected %s, got %s", testDID, did)
	}
}

func TestGetDIDEmpty(t *testing.T) {
	// Setup a temporary HOME directory
	tmpHome := t.TempDir()
	t.Setenv("HOME", tmpHome)

	odrlDir := filepath.Join(tmpHome, ".odrl")
	if err := os.MkdirAll(odrlDir, 0755); err != nil {
		t.Fatal(err)
	}
	didPath := filepath.Join(odrlDir, "did.json")

	// Test missing file
	did := GetDID()
	if did != "" {
		t.Errorf("expected empty string for missing file, got %s", did)
	}

	// Test invalid JSON
	if err := os.WriteFile(didPath, []byte("invalid json"), 0644); err != nil {
		t.Fatal(err)
	}
	did = GetDID()
	if did != "" {
		t.Errorf("expected empty string for invalid json, got %s", did)
	}
}
