package llm

import (
	"testing"
)

func TestWeightIndexBypass(t *testing.T) {
	idx := &WeightIndex{
		Method:        "bit-signature",
		ShortcutCache: make(map[uint64]int32),
		Metadata:      make(map[string]any),
	}

	// Mock sequence of tokens
	// "how to measure temperature?" -> [100, 200, 300, 400]
	window := []uint64{100, 200, 300, 400}
	predictedToken := int32(500)

	// Test 1: Prediction on empty cache should fail
	_, _, ok := idx.Predict(window)
	if ok {
		t.Error("Prediction should have failed on empty cache")
	}

	// Test 2: Learn a shortcut
	idx.Learn(window, predictedToken)
	if len(idx.ShortcutCache) != 1 {
		t.Errorf("Cache size should be 1, got %d", len(idx.ShortcutCache))
	}

	// Test 3: Predict the shortcut
	token, sim, ok := idx.Predict(window)
	if !ok {
		t.Error("Prediction should have succeeded")
	}
	if token != predictedToken {
		t.Errorf("Expected token %d, got %d", predictedToken, token)
	}
	if sim < 0.99 {
		t.Errorf("Expected high similarity, got %f", sim)
	}

	// Test 4: Sliding window shift
	// Next sequence: [200, 300, 400, 500] -> [600]
	newWindow := make([]uint64, 4)
	copy(newWindow, window[1:])
	newWindow[3] = uint64(500)
	
	idx.Learn(newWindow, 600)
	
	token, _, ok = idx.Predict(newWindow)
	if !ok || token != 600 {
		t.Errorf("Prediction failed after window shift")
	}
}

func TestHammingSimilarity(t *testing.T) {
	a := []uint64{0xFFFFFFFFFFFFFFFF, 0x0}
	b := []uint64{0xFFFFFFFFFFFFFFFF, 0x0}
	
	sim := hammingSimilarity(a, b)
	if sim != 1.0 {
		t.Errorf("Expected similarity 1.0 for identical bits, got %f", sim)
	}

	c := []uint64{0x0, 0x0}
	sim = hammingSimilarity(a, c)
	if sim != 0.5 {
		t.Errorf("Expected similarity 0.5 for half bits different, got %f", sim)
	}
}
