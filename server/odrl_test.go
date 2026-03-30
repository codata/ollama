package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDIDHandler(t *testing.T) {
	// Setup a temporary HOME directory
	tmpHome := t.TempDir()
	t.Setenv("HOME", tmpHome)

	odrlDir := filepath.Join(tmpHome, ".odrl")
	if err := os.MkdirAll(odrlDir, 0755); err != nil {
		t.Fatal(err)
	}
	didPath := filepath.Join(odrlDir, "did.json")

	testDID := "did:test:server:123"
	if err := os.WriteFile(didPath, []byte(`{"did":"`+testDID+`"}`), 0644); err != nil {
		t.Fatal(err)
	}

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatalf("failed to generate routes: %v", err)
	}

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/api/did", nil)
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status code 200, got %d", w.Code)
	}

	var resp map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}

	if resp["did"] != testDID {
		t.Errorf("expected %s, got %s", testDID, resp["did"])
	}
}
