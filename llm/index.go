package llm

import (
	"fmt"
	"log/slog"
	"math"
	"encoding/gob"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"math/bits"

	"github.com/ollama/ollama/fs/ggml"
)

// WeightIndex provides efficient lookups into model weights.
// It is intended to speed up semantic operations by shortcutting transformer passes.
type WeightIndex struct {
	mu           sync.RWMutex
	ModelPath    string
	Architecture string
	
	// VectorMap stores normalized embedding vectors for specific semantic units
	VectorMap    map[string][]float32
	
	// BitMap stores high-fidelity binary signatures for specific semantic units
	BitMap       map[string][]uint64
	
	// ShortcutCache stores previously computed results for specific semantic signatures
	shortcutMu    sync.RWMutex
	ShortcutCache map[uint64]int32
	
	// Metadata contains additional indexed information about layers or weights
	Metadata     map[string]any

	Method       string
	dirty        bool
	learnCount   int
}

// BuildWeightIndex creates a new index from a model's GGML structure.
// It now supports persistence by checking the ./index folder.
func BuildWeightIndex(modelPath string, f *ggml.GGML) (*WeightIndex, error) {
	indexDir := "index"
	modelName := filepath.Base(modelPath)
	indexPath := filepath.Join(indexDir, modelName+".index")

	method := os.Getenv("OLLAMA_INDEX_METHOD")
	if method == "" {
		method = "statistical" // default
	}

	// Ensure directory exists first
	if err := os.MkdirAll(indexDir, 0755); err != nil {
		slog.Error("CRITICAL: could not create index directory", "dir", indexDir, "error", err)
	}

	// Try to load existing index first
	if idx, err := LoadWeightIndex(indexPath); err == nil {
		if idx.Method == method || method == "statistical" {
			slog.Info("Found existing semantic index on disk", "model", modelName, "method", idx.Method)
			return idx, nil
		}
		slog.Info("Method mismatch in existing index, rebuilding...", "old", idx.Method, "new", method)
	}

	slog.Info("BUILDING NEW SEMANTIC INDEX", "model", modelPath, "method", method)
	
	// Open the model file to read actual weights for fingerprints
	file, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model for deep indexing: %w", err)
	}
	defer file.Close()

	idx := &WeightIndex{
		ModelPath:     modelPath,
		Architecture:  f.KV().Architecture(),
		VectorMap:     make(map[string][]float32),
		BitMap:        make(map[string][]uint64),
		ShortcutCache: make(map[uint64]int32),
		Metadata:      make(map[string]any),
		Method:        method,
	}

	// Index essential metadata
	idx.Metadata["parameter_count"] = f.KV().ParameterCount()
	idx.Metadata["block_count"] = f.KV().BlockCount()
	idx.Metadata["embedding_length"] = f.KV().EmbeddingLength()

	// Extract actual weight signatures
	dataOffset := f.Tensors().Offset
	tensors := f.Tensors().Items()
	slog.Info("STARTING DEEP SCAN", "data_offset", dataOffset, "tensors_total", len(tensors))

	// Debug: Print first few tensor names to verify naming convention
	for i := 0; i < 10 && i < len(tensors); i++ {
		slog.Info("Debug: Tensor Sample", "index", i, "name", tensors[i].Name)
	}

	indexedCount := 0
	for _, t := range tensors {
		nameLower := strings.ToLower(t.Name)
		// More inclusive matching
		if strings.Contains(nameLower, "weight") || 
		   strings.Contains(nameLower, "embd") || 
		   strings.Contains(nameLower, "blk") || 
		   strings.Contains(nameLower, "token") {
			
			// Calculate a signature by reading a sample of the tensor data
			absOffset := int64(dataOffset) + int64(t.Offset)
			
			// Sample 4KB
			sampleSize := int64(4096)
			if int64(t.Size()) < sampleSize {
				sampleSize = int64(t.Size())
			}
			
			buf := make([]byte, sampleSize)
			n, err := file.ReadAt(buf, absOffset)
			if err != nil || n == 0 {
				slog.Warn("Skipping tensor: read failure", "tensor", t.Name, "error", err)
				continue
			}

			if idx.Method == "bit-signature" {
				// High-fidelity bitmask of signs
				signature := make([]uint64, 8) // 512 bits
				for i := 0; i < len(buf) && i < 512; i++ {
					if buf[i] > 127 { // Simple sign-bit heuristic on raw bytes
						signature[i/64] |= (1 << (uint(i) % 64))
					}
				}
				idx.BitMap[t.Name] = signature
			} else {
				// Traditional statistical moments
				var sum, sumSq float64
				for _, b := range buf {
					val := float64(b)
					sum += val
					sumSq += val * val
				}
				
				count := float64(len(buf))
				mean := float32(sum / count)
				variance := (sumSq / count) - (sum*sum)/(count*count)
				stdDev := float32(math.Sqrt(max(0, variance)))

				idx.VectorMap[t.Name] = []float32{mean, stdDev, float32(t.Size())}
			}

			indexedCount++
			
			if indexedCount % 50 == 0 {
				slog.Info("Extraction Progress", "indexed", indexedCount, "current", t.Name, "method", idx.Method)
			}
		}
	}

	slog.Info("DEEP SCAN COMPLETE", "architecture", idx.Architecture, "indexed_tensors", indexedCount)
	
	// Save the newly built index
	if err := idx.Save(indexPath); err != nil {
		slog.Error("failed to save built index", "path", indexPath, "error", err)
	}

	return idx, nil
}

// Query performs a proximity search in the weight index to find similar activations or weights.
func (idx *WeightIndex) Query(vector []any, threshold float32) (string, float32, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var bestKey string
	var bestSim float32 = -1.0

	if idx.Method == "bit-signature" {
		targetSig := vector[0].([]uint64)
		for key, sig := range idx.BitMap {
			sim := hammingSimilarity(targetSig, sig)
			if sim > threshold && sim > bestSim {
				bestSim = sim
				bestKey = key
			}
		}
	} else {
		targetVec := vector[0].([]float32)
		for key, vec := range idx.VectorMap {
			sim := cosineSimilarity(targetVec, vec)
			if sim > threshold && sim > bestSim {
				bestSim = sim
				bestKey = key
			}
		}
	}

	if bestKey != "" {
		return bestKey, bestSim, true
	}
	return "", 0, false
}

// BuildSigKey attempts to find a shortcut token for the given signature
func BuildSigKey(sig []uint64) uint64 {
	var h uint64 = 0x811c9dc5
	for _, x := range sig {
		h ^= x
		h *= 0x01000193
	}
	return h
}

// Predict attempts to find a shortcut token for the given signature
func (idx *WeightIndex) Predict(sig []uint64) (int32, float32, bool) {
	idx.shortcutMu.RLock()
	defer idx.shortcutMu.RUnlock()

	key := BuildSigKey(sig)
	if token, ok := idx.ShortcutCache[key]; ok {
		return token, 1.0, true
	}
	return 0, 0, false
}

// Learn registers a new successful shortcut in the index
func (idx *WeightIndex) Learn(sig []uint64, token int32) {
	idx.shortcutMu.Lock()
	defer idx.shortcutMu.Unlock()

	key := BuildSigKey(sig)
	if _, exists := idx.ShortcutCache[key]; !exists {
		idx.ShortcutCache[key] = token
		idx.dirty = true
		idx.learnCount++
		
		// Auto-save every 500 new shortcuts to prevent data loss on crash/pkill
		if idx.learnCount >= 500 {
			idx.learnCount = 0
			// Use a goroutine to avoid blocking the inference loop
			go func(mpath string, i *WeightIndex) {
				modelName := filepath.Base(mpath)
				indexPath := filepath.Join("index", modelName+".index")
				i.Save(indexPath)
			}(idx.ModelPath, idx)
		}
	}
}

// Reset clears the dynamic shortcuts and metadata
func (idx *WeightIndex) Reset() {
	idx.mu.Lock()
	idx.shortcutMu.Lock()
	defer idx.mu.Unlock()
	defer idx.shortcutMu.Unlock()
	idx.ShortcutCache = make(map[uint64]int32)
	idx.Metadata["last_reset"] = time.Now()
}

func hammingSimilarity(a, b []uint64) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var matches int
	totalBits := len(a) * 64
	for i := range a {
		diff := a[i] ^ b[i]
		matches += (64 - bits.OnesCount64(diff))
	}
	return float32(matches) / float32(totalBits)
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func (idx *WeightIndex) Lookup(key string) (any, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	
	if val, ok := idx.Metadata[key]; ok {
		return val, true
	}
	if vec, ok := idx.VectorMap[key]; ok {
		return vec, true
	}
	if sig, ok := idx.BitMap[key]; ok {
		return sig, true
	}
	return nil, false
}

func (idx *WeightIndex) StatusText() string {
	idx.mu.RLock()
	idx.shortcutMu.RLock()
	defer idx.mu.RUnlock()
	defer idx.shortcutMu.RUnlock()
	count := len(idx.VectorMap)
	if idx.Method == "bit-signature" {
		count = len(idx.BitMap)
	}
	return fmt.Sprintf("Index: %s [%s] | Vectors: %d | Shortcuts: %d", 
		idx.Architecture, idx.Method, count, len(idx.ShortcutCache))
}

func (idx *WeightIndex) String() string {
	count := len(idx.VectorMap)
	if idx.Method == "bit-signature" {
		count = len(idx.BitMap)
	}
	return fmt.Sprintf("WeightIndex(%s, method=%s, tensors=%d)", idx.Architecture, idx.Method, count)
}

func (idx *WeightIndex) Save(path string) error {
	idx.mu.RLock()
	idx.shortcutMu.RLock()
	defer idx.mu.RUnlock()
	defer idx.shortcutMu.RUnlock()

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	// Encode specific fields to avoid issues with sync.RWMutex
	if err := enc.Encode(idx.ModelPath); err != nil { return err }
	if err := enc.Encode(idx.Architecture); err != nil { return err }
	if err := enc.Encode(idx.Method); err != nil { return err }
	if err := enc.Encode(idx.VectorMap); err != nil { return err }
	if err := enc.Encode(idx.BitMap); err != nil { return err }
	if err := enc.Encode(idx.ShortcutCache); err != nil { return err }
	if err := enc.Encode(idx.Metadata); err != nil { return err }
	
	slog.Info("weight index saved successfully", "path", path)
	return nil
}

func LoadWeightIndex(path string) (*WeightIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	idx := &WeightIndex{
		VectorMap:     make(map[string][]float32),
		ShortcutCache: make(map[uint64]int32),
		Metadata:      make(map[string]any),
	}

	dec := gob.NewDecoder(f)
	if err := dec.Decode(&idx.ModelPath); err != nil { return nil, err }
	if err := dec.Decode(&idx.Architecture); err != nil { return nil, err }
	if err := dec.Decode(&idx.Method); err != nil { return nil, err }
	if err := dec.Decode(&idx.VectorMap); err != nil { return nil, err }
	if err := dec.Decode(&idx.BitMap); err != nil { return nil, err }
	if err := dec.Decode(&idx.ShortcutCache); err != nil { return nil, err }
	if err := dec.Decode(&idx.Metadata); err != nil { return nil, err }

	slog.Info("weight index loaded successfully", "path", path)
	return idx, nil
}
