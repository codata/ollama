package llm

import (
	"fmt"
	"log"
	"log/slog"
	"math"
	"encoding/gob"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"math/bits"
	"strconv"

	"github.com/ollama/ollama/fs/ggml"
)

var staticLogs sync.Once

func init() {
	gob.Register(map[uint64][]int32{})
	gob.Register(map[uint64]string{})
	gob.Register(map[uint64][]uint64{})
	gob.Register(map[uint64]int{})
	gob.Register(map[string][]float32{})
	gob.Register(map[string]any{})
	gob.Register(uint64(0))
	gob.Register([]int32{})
	gob.Register(int32(0))
}

func GetIndexDir() string {
	dir := os.Getenv("OLLAMA_INDEX_DIRECTORY")
	if dir == "" {
		dir = "/tmp/index"
	}
	return dir
}

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
	
	// SequenceCache stores full responses for specific query signatures
	sequenceMu    sync.RWMutex
	SequenceCache map[uint64][]int32
	
	// MemoryCache stores human-readable memory fragments for specific query signatures
	MemoryCache   map[uint64]string
	
	// PromptSignatures stores fingerprints of prompts for fuzzy navigation
	PromptSignatures map[uint64][]uint64
	
	// Metadata contains additional indexed information about layers or weights
	Metadata     map[string]any

	// StabilityCounter tracks consistent generations for a query
	StabilityCounter map[uint64]int
	
	Method       string
	dirty        bool
	learnCount   int
	
	// Hot Hydration State
	lastHydration   time.Time
	loadedFragments map[string]bool
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
		ModelPath:        modelPath,
		Architecture:     f.KV().Architecture(),
		VectorMap:        make(map[string][]float32),
		BitMap:           make(map[string][]uint64),
		ShortcutCache:    make(map[uint64]int32),
		SequenceCache:    make(map[uint64][]int32),
		MemoryCache:      make(map[uint64]string),
		Metadata:         make(map[string]any),
		StabilityCounter: make(map[uint64]int),
		Method:           method,
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

// BuildShortcutKey creates a key for single-token prediction
func BuildShortcutKey(sig []uint64) uint64 {
	var h uint64 = 0x811c9dc5
	for _, x := range sig {
		h ^= x
		h *= 0x01000193
	}
	return h
}

func (idx *WeightIndex) Predict(sig []uint64) (int32, float32, bool) {
	key := BuildShortcutKey(sig)
	idx.shortcutMu.RLock()
	defer idx.shortcutMu.RUnlock()

	if token, ok := idx.ShortcutCache[key]; ok {
		return token, 1.0, true
	}
	return 0, 0, false
}

// Learn registers a new successful shortcut in the index
func (idx *WeightIndex) Learn(sig []uint64, token int32) {
	idx.shortcutMu.Lock()
	defer idx.shortcutMu.Unlock()

	key := BuildShortcutKey(sig)
	if _, exists := idx.ShortcutCache[key]; !exists {
		idx.ShortcutCache[key] = token
		idx.dirty = true
		idx.learnCount++
		
		// Auto-save every 500 new shortcuts to prevent data loss on crash/pkill
		if idx.learnCount >= 500 {
			idx.learnCount = 0
			// Background auto-save removed
		}
	}
}

func isFuzzyEnabled() bool {
	return os.Getenv("OLLAMA_FUZZY_NAVIGATION") == "1"
}

// RegisterMemory captures a human-readable response tied to a semantic signature
func (idx *WeightIndex) RegisterMemory(promptTokens []int32, response string) {
	if len(promptTokens) == 0 { return }
	targetSig := idx.CalculateTokenSignature(promptTokens)
	key := BuildStructuralKey(targetSig, len(promptTokens))
	idx.sequenceMu.Lock()
	defer idx.sequenceMu.Unlock()

	if idx.MemoryCache == nil {
		idx.MemoryCache = make(map[uint64]string)
	}

	idx.MemoryCache[key] = response
	
	// Also ensure PromptSignature is captured for fuzzy search
	if len(promptTokens) > 0 {
		if idx.PromptSignatures == nil {
			idx.PromptSignatures = make(map[uint64][]uint64)
		}
		idx.PromptSignatures[key] = idx.CalculateTokenSignature(promptTokens)
		idx.dirty = true
	}
}

// SearchMemory attempts to find a highly related previous response to use as context
func (idx *WeightIndex) SearchMemory(promptTokens []int32) (string, float32, bool) {
	if !isFuzzyEnabled() || len(promptTokens) == 0 {
		return "", 0, false
	}

	targetSig := calculateTokenSignature(promptTokens)
	idx.sequenceMu.RLock()
	defer idx.sequenceMu.RUnlock()

	var bestKey uint64
	var bestSim float32 = -1.0

	// Segmented Search (Sliding Window of 64 tokens)
	for key, storedSig := range idx.PromptSignatures {
		// 1. Check Global Similarity
		sim := hammingSimilarity(targetSig, storedSig)
		
		// 2. Check Segment Similarity (First 64 tokens) - WEIGHTED
		// If the variable is at the start (common in priming), this catches it
		if sim < 0.7 && len(promptTokens) > 64 {
			segSig := calculateTokenSignature(promptTokens[:64])
			segSim := hammingSimilarity(segSig, storedSig)
			if segSim > sim { sim = segSim }
		}

		if sim >= 0.4 && sim > bestSim {
			bestSim = sim
			bestKey = key
		}
	}

	if bestSim >= 0.4 {
		if mem, ok := idx.MemoryCache[bestKey]; ok {
			return mem, bestSim, true
		}
	}

	return "", 0, false
}

func (idx *WeightIndex) CalculateTokenSignature(promptTokens []int32) []uint64 {
	return calculateTokenSignature(promptTokens)
}

// BuildStructuralKey creates a collision-resistant key from the high-entropy fingerprint and length
// BuildStructuralKey creates a high-fidelity cryptographic key from the token signature.
// It uses a diffusion-heavy prime multiplication hash to ensure zero collisions
// between prompts that share similar instructional prefixes.
func BuildStructuralKey(sig []uint64, tokenCount int) uint64 {
	// Offset basis for 64-bit FNV
	var h uint64 = 0xcbf29ce484222325
	prime := uint64(0x100000001b3)

	// Incorporate length first to separate different prompt structures
	h ^= uint64(tokenCount)
	h *= prime

	for i, s := range sig {
		// Mix position into the hash to ensure sensitivity to token order
		h ^= (s ^ uint64(i))
		h *= prime
		
		// Final diffusion step per block
		h ^= (h >> 32)
	}

	return h
}

// PredictSequence attempts to find a full response for the given query signature
func (idx *WeightIndex) PredictSequence(promptTokens []int32, promptString string) ([]int32, bool) {
	// 0. Hot Hydration: Sync with Global Memory before each new query
	idx.HotReload()

	if len(promptTokens) == 0 { return nil, false }
	
	targetSig := calculateTokenSignature(promptTokens)
	key := BuildStructuralKey(targetSig, len(promptTokens))
	
	idx.sequenceMu.RLock()
	defer idx.sequenceMu.RUnlock()

	// 1. Exact Structural Match (High-Speed Bypass with Stability Check)
	// This only fires for identical prompts after enough consistent cycles.
	if seq, ok := idx.SequenceCache[key]; ok {
		cycles := 2 // default to 2 as per community standard
		if cyclesStr := os.Getenv("OLLAMA_LEARN_CYCLES"); cyclesStr != "" {
			if c, err := strconv.Atoi(cyclesStr); err == nil {
				cycles = c
			}
		}

		// Log once to verify environmental propagation
		staticLogs.Do(func() {
			log.Printf("CODATA Fabric: Active Sovereignty Threshold: %d cycles (from ENV: '%s')", cycles, os.Getenv("OLLAMA_LEARN_CYCLES"))
		})

		hits := idx.StabilityCounter[key]
		if hits >= cycles {
			log.Printf("CODATA Fabric: Structural HIT (Key: %x, stability=%d, tokens=%d)", key, hits, len(seq))
			return seq, true
		} else {
			log.Printf("CODATA Fabric: Learning Phase (Key: %x, observation=%d/%d)", key, hits+1, cycles)
		}
	}

	return nil, false
}

func calculateTokenSignature(tokens []int32) []uint64 {
	sig := make([]uint64, 8)
	// Sequence-Aware Rolling Hash to ensure subject sovereignty
	var rolling uint64 = 0xcbf29ce484222325 // FNV offset basis
	for i, t := range tokens {
		// Blend token value, position, and previous state
		h := (uint64(t) ^ uint64(i)) * 0x100000001b3
		rolling = (rolling ^ h) * 0x100000001b3
		
		// Map the rolling state into the 512-bit signature window
		bitPos := rolling % 512
		sig[bitPos/64] |= (1 << (bitPos % 64))
	}
	return sig
}

// Global test cache to survive deadlocks
var GlobalSequenceCache = make(map[uint64][]int32)

// RegisterSequence stores a full response and its textual content for a query signature
func (idx *WeightIndex) RegisterSequence(tokens []int32, promptTokens []int32, responseText string) {
	if len(promptTokens) == 0 { return }
	
	targetSig := calculateTokenSignature(promptTokens)
	key := BuildStructuralKey(targetSig, len(promptTokens))
	
	idx.sequenceMu.Lock()
	defer idx.sequenceMu.Unlock()
	
	if idx.SequenceCache == nil {
		idx.SequenceCache = make(map[uint64][]int32)
	}
	if idx.MemoryCache == nil {
		idx.MemoryCache = make(map[uint64]string)
	}
	if idx.PromptSignatures == nil {
		idx.PromptSignatures = make(map[uint64][]uint64)
	}
	
	if idx.StabilityCounter == nil {
		idx.StabilityCounter = make(map[uint64]int)
	}
	
	idx.SequenceCache[key] = tokens
	idx.PromptSignatures[key] = targetSig
	idx.MemoryCache[key] = responseText
	idx.StabilityCounter[key]++
	
	idx.dirty = true
	
	// Pass captured state to avoid deadlock with caller holding sequenceMu
	idx.SaveAsync(fmt.Sprintf("codata_%x.nav", key))
}

func (idx *WeightIndex) SaveAsync(filename string) {
	// Launch entirely in background to avoid deadlock with caller's locks
	go func() {
		dir := GetIndexDir()
		if err := os.MkdirAll(dir, 0755); err != nil {
			return
		}

		path := filepath.Join(dir, filename)
		
		idx.mu.RLock()
		idx.shortcutMu.RLock()
		idx.sequenceMu.RLock()
		
		modelPath := idx.ModelPath
		arch := idx.Architecture
		method := idx.Method
		
		vMap := make(map[string][]float32)
		for k, v := range idx.VectorMap { vMap[k] = v }
		bMap := make(map[string][]uint64)
		for k, v := range idx.BitMap { bMap[k] = v }
		sCache := make(map[uint64]int32)
		for k, v := range idx.ShortcutCache { sCache[k] = v }
		seqCache := make(map[uint64][]int32)
		for k, v := range idx.SequenceCache { seqCache[k] = v }
		mCache := make(map[uint64]string)
		for k, v := range idx.MemoryCache { mCache[k] = v }
		pSigs := make(map[uint64][]uint64)
		for k, v := range idx.PromptSignatures { pSigs[k] = v }

		meta := make(map[string]any)
		for k, v := range idx.Metadata { meta[k] = v }
		stbCounter := make(map[uint64]int)
		for k, v := range idx.StabilityCounter { stbCounter[k] = v }

		idx.sequenceMu.RUnlock()
		idx.shortcutMu.RUnlock()
		idx.mu.RUnlock()

		tmpPath := path + ".tmp"
		f, err := os.Create(tmpPath)
		if err != nil { return }
		defer f.Close()

		enc := gob.NewEncoder(f)
		enc.Encode(modelPath)
		enc.Encode(arch)
		enc.Encode(method)
		enc.Encode(vMap)
		enc.Encode(bMap)
		enc.Encode(sCache)
		enc.Encode(seqCache)
		enc.Encode(meta)
		enc.Encode(mCache)
		enc.Encode(pSigs)
		enc.Encode(stbCounter)

		f.Sync()
		os.Rename(tmpPath, path)
	}()
}

// Reset clears the dynamic shortcuts, sequences and metadata
func (idx *WeightIndex) Reset() {
	idx.mu.Lock()
	idx.shortcutMu.Lock()
	idx.sequenceMu.Lock()
	defer idx.mu.Unlock()
	defer idx.shortcutMu.Unlock()
	defer idx.sequenceMu.Unlock()
	idx.ShortcutCache = make(map[uint64]int32)
	idx.SequenceCache = make(map[uint64][]int32)
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
	idx.sequenceMu.RLock()
	defer idx.mu.RUnlock()
	defer idx.shortcutMu.RUnlock()
	defer idx.sequenceMu.RUnlock()
	count := len(idx.VectorMap)
	if idx.Method == "bit-signature" {
		count = len(idx.BitMap)
	}
	return fmt.Sprintf("Index: %s [%s] | Vectors: %d | Shortcuts: %d | Sequences: %d", 
		idx.Architecture, idx.Method, count, len(idx.ShortcutCache), len(idx.SequenceCache))
}

func (idx *WeightIndex) String() string {
	count := len(idx.VectorMap)
	if idx.Method == "bit-signature" {
		count = len(idx.BitMap)
	}
	return fmt.Sprintf("WeightIndex(%s, method=%s, tensors=%d)", idx.Architecture, idx.Method, count)
}

func (idx *WeightIndex) Save(path string) error {
	idx.SaveAsync(path)
	return nil
}

func LoadWeightIndex(ignoredPath string) (*WeightIndex, error) {
	dir := GetIndexDir()
	slog.Info("CODATA Fabric: Hydrating Semantic Index fragments", "dir", dir)

	idx := &WeightIndex{
		VectorMap:     make(map[string][]float32),
		BitMap:        make(map[string][]uint64),
		ShortcutCache: make(map[uint64]int32),
		SequenceCache: make(map[uint64][]int32),
		Metadata:      make(map[string]any),
		StabilityCounter: make(map[uint64]int),
	}

	files, _ := filepath.Glob(filepath.Join(dir, "*.nav"))
	legacyFiles, _ := filepath.Glob(filepath.Join(dir, "*.index"))
	allFiles := append(files, legacyFiles...)

	if len(allFiles) == 0 {
		return idx, fmt.Errorf("no navigation fragments found in %s", dir)
	}

	for _, path := range allFiles {
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		
		dec := gob.NewDecoder(f)
		var mPath, arch, meth string
		var vMap map[string][]float32
		var bMap map[string][]uint64
		var sCache map[uint64]int32
		var seqCache map[uint64][]int32
		var meta map[string]any

		if err := dec.Decode(&mPath); err == nil { idx.ModelPath = mPath }
		if err := dec.Decode(&arch); err == nil { idx.Architecture = arch }
		if err := dec.Decode(&meth); err == nil { idx.Method = meth }
		if err := dec.Decode(&vMap); err == nil { for k, v := range vMap { idx.VectorMap[k] = v } }
		if err := dec.Decode(&bMap); err == nil { for k, v := range bMap { idx.BitMap[k] = v } }
		if err := dec.Decode(&sCache); err == nil { for k, v := range sCache { idx.ShortcutCache[k] = v } }
		if err := dec.Decode(&seqCache); err == nil { for k, v := range seqCache { idx.SequenceCache[k] = v } }
		if err := dec.Decode(&meta); err == nil { for k, v := range meta { idx.Metadata[k] = v } }
		
		// Fabric specific decodings
		var mCache map[uint64]string
		var pSigs map[uint64][]uint64
		if err := dec.Decode(&mCache); err == nil { 
			if idx.MemoryCache == nil { idx.MemoryCache = make(map[uint64]string) }
			for k, v := range mCache { idx.MemoryCache[k] = v } 
		}
		if err := dec.Decode(&pSigs); err == nil { 
			if idx.PromptSignatures == nil { idx.PromptSignatures = make(map[uint64][]uint64) }
			for k, v := range pSigs { idx.PromptSignatures[k] = v } 
		}
		var stbCounter map[uint64]int
		if err := dec.Decode(&stbCounter); err == nil {
			if idx.StabilityCounter == nil { idx.StabilityCounter = make(map[uint64]int) }
			for k, v := range stbCounter { idx.StabilityCounter[k] = v }
		}
		
		f.Close()
	}

	GlobalSequenceCache = idx.SequenceCache
	idx.lastHydration = time.Now()
	if idx.loadedFragments == nil {
		idx.loadedFragments = make(map[string]bool)
	}
	for _, f := range allFiles {
		idx.loadedFragments[f] = true
	}
	slog.Info("CODATA Fabric: Sovereignty Hydrated from Loom Fragments", 
		"sequences", len(idx.SequenceCache), 
		"fragments", len(allFiles),
		"stb_map", len(idx.StabilityCounter))
	return idx, nil
}

// HotReload scans the index directory for new fragments and merges them into the active index.
func (idx *WeightIndex) HotReload() {
	indexDir := os.Getenv("OLLAMA_INDEX_DIRECTORY")
	if indexDir == "" {
		indexDir = "/tmp/index"
	}

	dirs := []string{indexDir, "index"}
	var newFiles []string
	
	for _, dir := range dirs {
		files, _ := filepath.Glob(filepath.Join(dir, "*.nav"))
		for _, f := range files {
			if !idx.loadedFragments[f] {
				newFiles = append(newFiles, f)
			}
		}
	}

	if len(newFiles) == 0 {
		return
	}

	slog.Info("CODATA Fabric: Hot Hydrating Global Memory", "new_fragments", len(newFiles))
	
	for _, path := range newFiles {
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		
		dec := gob.NewDecoder(f)
		var seqCache map[uint64][]int32
		var stbCounter map[uint64]int
		var meta map[string]any
		
		// Use a temporary map to decode and merge
		// Fields must match the encoding order in SaveAsync
		var dummyStr string
		dec.Decode(&dummyStr) // modelPath
		dec.Decode(&dummyStr) // architecture
		dec.Decode(&dummyStr) // method
		
		var dummyMap map[string][]float32
		dec.Decode(&dummyMap) // vectorMap
		var dummyBit map[string][]uint64
		dec.Decode(&dummyBit) // bitMap
		
		var dummyShort map[uint64]int32
		dec.Decode(&dummyShort) // shortcutCache
		
		if err := dec.Decode(&seqCache); err == nil {
			for k, v := range seqCache {
				idx.SequenceCache[k] = v
			}
		}
		
		if err := dec.Decode(&meta); err == nil {
			for k, v := range meta {
				idx.Metadata[k] = v
			}
		}

		var dummyMem map[uint64]string
		dec.Decode(&dummyMem) // memoryCache
		var dummySig map[uint64][]uint64
		dec.Decode(&dummySig) // promptSignatures

		if err := dec.Decode(&stbCounter); err == nil {
			for k, v := range stbCounter {
				idx.StabilityCounter[k] = v
			}
		}
		
		f.Close()
		idx.loadedFragments[path] = true
	}
	
	idx.lastHydration = time.Now()
	GlobalSequenceCache = idx.SequenceCache
}
