package mlxrunner

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
)

type Request struct {
	TextCompletionsRequest
	Responses chan CompletionResponse
	Pipeline  func(Request) error

	Ctx context.Context

	Sampler *sample.Sampler
}

type Options struct {
	Temperature     float32 `json:"temperature"`
	TopP            float32 `json:"top_p"`
	MinP            float32 `json:"min_p"`
	TopK            int     `json:"top_k"`
	RepeatLastN     int     `json:"repeat_last_n"`
	PresencePenalty float32 `json:"presence_penalty"`
	MaxTokens       int     `json:"max_tokens"`
	Optimization    string  `json:"optimization"`
	IndexCommand    string  `json:"index_command"`

	// Deprecated: use MaxTokens instead
	NumPredict int `json:"num_predict"`
}

type TextCompletionsRequest struct {
	Prompt  string  `json:"prompt"`
	Options Options `json:"options"`
}

type Runner struct {
	Model         base.Model
	Tokenizer     *tokenizer.Tokenizer
	Requests      chan Request
	cache         kvCache
	contextLength int
	Index         *llm.WeightIndex
}

func (r *Runner) Load(modelName string) error {
	root, err := model.Open(modelName)
	if err != nil {
		return err
	}
	defer root.Close()

	m, err := base.New(root)
	if err != nil {
		return err
	}

	// Load all tensor blobs from manifest
	tensors, err := loadTensorsFromManifest(root)
	if err != nil {
		return err
	}

	// Assign weights to model (model-specific logic)
	loadWeights := base.Weights(m)
	if err := loadWeights(tensors); err != nil {
		return err
	}

	r.Model = m
	r.Tokenizer = m.Tokenizer()
	r.contextLength = m.MaxContextLength()

	// Handle semantic index persistence using the blob digest for shared caching
	indexDir := "/Users/vyacheslavtykhonov/projects/dev/ollama/index"
	var identifier string
	layers := root.Manifest.GetTensorLayers("")
	if len(layers) > 0 {
		identifier = layers[0].Digest
	} else {
		identifier = modelName
	}
	
	indexPath := filepath.Join(indexDir, identifier+".index")
	absPath, _ := filepath.Abs(indexPath)
	
	// Create index directory explicitly
	if err := os.MkdirAll(indexDir, 0755); err != nil {
		slog.Error("Could not create index directory", "dir", indexDir, "error", err)
	}

	idx, err := llm.LoadWeightIndex(indexPath)
	if err == nil {
		r.Index = idx
		slog.Info("Loaded persistent semantic index", "identifier", identifier, "path", absPath)
	} else {
		method := os.Getenv("OLLAMA_INDEX_METHOD")
		if method == "" {
			method = "statistical"
		}
		
		r.Index = &llm.WeightIndex{
			ModelPath:     identifier,
			Architecture:  "llama",
			VectorMap:     make(map[string][]float32),
			BitMap:        make(map[string][]uint64),
			ShortcutCache: make(map[uint64]int32),
			Metadata:      make(map[string]any),
			Method:        method,
		}
		// Index tensors by name
		for name := range tensors {
			if strings.Contains(name, "embd") || strings.Contains(name, "weight") {
				if method == "bit-signature" {
					// Use identity-hash for weight pointers during initial bootstrap
					r.Index.BitMap[name] = make([]uint64, 8)
				} else {
					r.Index.VectorMap[name] = []float32{1.0, 0.0, 1.0}
				}
			}
		}
		// PERSIST LEARNED SHORTCUTS ON TERMINATION
		if r.Index != nil {
			indexDir := "/Users/vyacheslavtykhonov/projects/dev/ollama/index"
			indexPath := filepath.Join(indexDir, r.Index.ModelPath+".index")
			if err := r.Index.Save(indexPath); err != nil {
				slog.Error("failed to auto-save learned index", "path", indexPath, "error", err)
			} else {
				slog.Info("auto-saved learned index", "path", indexPath, "shortcuts", len(r.Index.ShortcutCache))
			}
		}
	}

	return nil
}

// loadTensorsFromManifest loads all tensor blobs from the manifest into a
// flat map, deduplicating by digest and remapping safetensors key suffixes.
//
// Uses a two-phase approach: first loads all raw tensors, then remaps
// .bias → _qbias with complete knowledge of which base names have .scale
// entries. This avoids a race condition where Go map iteration order could
// cause .bias to be processed before .scale within the same blob.
func loadTensorsFromManifest(root *model.Root) (map[string]*mlx.Array, error) {
	// Phase 1: Load all tensors raw from all blobs
	rawTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		blobPath := root.Manifest.BlobPath(layer.Digest)
		for name, arr := range mlx.Load(blobPath) {
			rawTensors[name] = arr
		}
	}

	// Phase 2: Identify all base names that have .scale tensors and remap them
	scaleBaseNames := make(map[string]bool)
	allTensors := make(map[string]*mlx.Array, len(rawTensors))
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			baseName := strings.TrimSuffix(name, ".scale")
			allTensors[baseName+"_scale"] = arr
			scaleBaseNames[baseName] = true
		}
	}

	// Phase 3: Process remaining tensors with complete scale knowledge
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			continue // already handled
		}
		if strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias") {
			baseName := strings.TrimSuffix(name, ".bias")
			if scaleBaseNames[baseName] {
				allTensors[baseName+"_qbias"] = arr
			} else {
				allTensors[name] = arr
			}
		} else {
			allTensors[name] = arr
		}
	}

	slog.Info("Loaded tensors from manifest", "count", len(allTensors))
	return allTensors, nil
}

func (r *Runner) Run(host, port string, mux http.Handler) error {
	g, ctx := errgroup.WithContext(context.Background())

	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case request := <-r.Requests:
				if err := request.Pipeline(request); err != nil {
					slog.Info("Request terminated", "error", err)
					var statusErr api.StatusError
					if !errors.As(err, &statusErr) {
						statusErr = api.StatusError{
							StatusCode:   http.StatusInternalServerError,
							ErrorMessage: err.Error(),
						}
					}
					select {
					case request.Responses <- CompletionResponse{Error: &statusErr}:
					case <-request.Ctx.Done():
					}
				}

				close(request.Responses)
			}
		}
	})

	g.Go(func() error {
		slog.Info("Starting HTTP server", "host", host, "port", port)
		return http.ListenAndServe(net.JoinHostPort(host, port), mux)
	})

	return g.Wait()
}
