package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func prefillChunkSize() int {
	return 2 << 10
}

func (r *Runner) TextGenerationPipeline(request Request) error {
	ctx := request.Ctx
	
	// Handle index commands immediately
	if request.Options.IndexCommand != "" {
		if r.Index == nil {
			select {
			case request.Responses <- CompletionResponse{Content: "Index not available for this model.\n", Done: true}:
				return nil
			default:
				return nil
			}
		}

		if request.Options.IndexCommand == "status" {
			select {
			case request.Responses <- CompletionResponse{Content: r.Index.StatusText() + "\n", Done: true}:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		if request.Options.IndexCommand == "rebuild" {
			status := r.Index.StatusText()
			// Send initial status
			select {
			case request.Responses <- CompletionResponse{Content: "Triggering rebuild. Current state: " + status + "\n"}:
			case <-ctx.Done():
				return ctx.Err()
			}

			r.Index.Reset()
			
			// Save the reset index immediately
			indexPath := filepath.Join("index", r.Index.ModelPath+".index")
			if err := r.Index.Save(indexPath); err != nil {
				slog.Error("failed to save reset index", "error", err)
			}

			// Send completion message
			select {
			case request.Responses <- CompletionResponse{Content: "Index reset completed. New state: " + r.Index.StatusText() + "\n", Done: true}:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	if r.Model == nil {
		return errors.New("model not loaded")
	}

	enableCompile := true
	if modelCompile, ok := r.Model.(interface{ EnableCompile() bool }); ok {
		enableCompile = modelCompile.EnableCompile()
	}
	if enableCompile {
		mlx.EnableCompile()
	} else {
		mlx.DisableCompile()
	}
	mlx.ResetPeakMemory()
	ctx = request.Ctx
	var (
		sample, logprobs         *mlx.Array
		nextSample, nextLogprobs *mlx.Array
	)
	defer func() {
		if request.Sampler != nil {
			request.Sampler.Free()
		}
		mlx.Unpin(sample, logprobs)
		mlx.Unpin(nextSample, nextLogprobs)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.dumpTree()
		}
		slog.Info("peak memory", "size", mlx.PrettyBytes(mlx.PeakMemory()))

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
	}()

	if r.Index != nil {
		slog.Info("Starting pipeline with WeightIndex", "method", r.Index.Method, "shortcuts", len(r.Index.ShortcutCache))
	} else {
		slog.Warn("Starting pipeline WITHOUT WeightIndex")
	}

	inputs := r.Tokenizer.Encode(request.Prompt, r.Tokenizer.AddBOS())
	if len(inputs) == 0 {
		return errors.New("empty prompt")
	}

	if len(inputs) >= r.contextLength {
		return api.StatusError{
			StatusCode:   http.StatusBadRequest,
			ErrorMessage: fmt.Sprintf("input length (%d tokens) exceeds the model's maximum context length (%d tokens)", len(inputs), r.contextLength),
		}
	}

	// Cap generation to stay within the model's context length
	maxGenerate := r.contextLength - len(inputs)
	if request.Options.MaxTokens <= 0 {
		request.Options.MaxTokens = maxGenerate
	} else {
		request.Options.MaxTokens = min(request.Options.MaxTokens, maxGenerate)
	}

	request.Sampler.ResetHistory(inputs)

	session := r.cache.begin(r.Model, inputs)
	defer session.close()

	caches := session.caches
	tokens := session.remaining
	prefillChunk := prefillChunkSize()

	// Request periodic snapshots during prefill and near the end of the
	// prompt so that long prompts can be partially restored and
	// thinking/generation can be retried without full reprocessing.
	const snapshotInterval = 8192
	for offset := snapshotInterval; offset < len(inputs); offset += snapshotInterval {
		session.requestSnapshot(offset)
	}

	const preThinking = 4
	if end := len(inputs) - preThinking; end > 0 {
		session.requestSnapshot(end)
	}

	materializeCaches := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			state = append(state, c.State()...)
		}
		if len(state) == 0 {
			return
		}
		mlx.Eval(state...)
	}

	now := time.Now()
	total, processed := len(tokens), 0
	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			return err
		}

		n := min(prefillChunk, total-processed-1)

		// If there's a pending snapshot, split the batch so we can
		// capture it at the exact offset.
		if snapOffset := session.nextPendingSnapshot(); snapOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			tokensUntilSnapshot := snapOffset - (baseOffset + processed)
			if tokensUntilSnapshot > 0 && tokensUntilSnapshot < n {
				n = tokensUntilSnapshot
			}
		}

		r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		mlx.Sweep()
		materializeCaches()
		processed += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)

		// Create snapshot if we've reached a pending offset.
		if snapOffset := session.nextPendingSnapshot(); snapOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			if baseOffset+processed >= snapOffset {
				session.snapshot()
			}
		}

		mlx.ClearCache()
	}

	window := make([]uint64, 4) // 4-token semantic window
	// Seed window from the last 4 tokens of the prompt
	for i := 0; i < 4; i++ {
		idx := len(inputs) - 4 + i
		if idx >= 0 {
			window[i] = uint64(inputs[idx])
		}
	}

	stepCount := 0
	step := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		// --- DIAGNOSTIC HEARTBEAT ---
		stepCount++
		isSingle := token.Size() == 1
		idxMethod := "nil"
		if r.Index != nil { idxMethod = r.Index.Method }
		
		if stepCount <= 5 {
			slog.Info("Step diagnostic", "step", stepCount, "size", token.Size(), "method", idxMethod, "shortcuts", len(r.Index.ShortcutCache))
		}

		// --- BYPASS LOGIC START ---
		idxActive := r.Index != nil && strings.TrimSpace(r.Index.Method) == "bit-signature"
		
		if idxActive && isSingle {
			val := uint64(token.Int())
			// Shift window and add new token
			copy(window, window[1:])
			window[3] = val
			
			if predictedToken, sim, ok := r.Index.Predict(window); ok && sim > 0.99 {
				slog.Info("SEMANTIC BYPASS TRIGGERED", "token", predictedToken, "window", window)
				sample := mlx.FromValue(int(predictedToken))
				logprobs := mlx.FromValues([]float32{0.0}, 1) 
				mlx.Pin(sample, logprobs)
				return sample, logprobs
			}
		}

		fwd := r.Model.Forward(token.ExpandDims(0), caches)
		logits := r.Model.Unembed(fwd)
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		logprobs := logits.Subtract(logits.Logsumexp(true))
		sample := request.Sampler.Sample(logprobs)

		// --- LEARN LOGIC START ---
		if r.Index != nil && r.Index.Method == "bit-signature" && token.Size() == 1 {
			r.Index.Learn(window, int32(sample.Int()))
			// Force visibility for the first few learned items
			sz := len(r.Index.ShortcutCache)
			if sz < 10 || sz%10 == 0 {
				slog.Info("shortcut cache status", "learned", sz)
			}
		}
		// --- LEARN LOGIC END ---

		mlx.Pin(sample, logprobs)
		mlx.Sweep()
		mlx.AsyncEval(sample, logprobs)

		return sample, logprobs
	}

	sample, logprobs = step(mlx.FromValues(tokens[processed:], total-processed))

	var b bytes.Buffer

	final := CompletionResponse{Done: true, PromptEvalCount: len(inputs), EvalCount: request.Options.MaxTokens, DoneReason: 1}
	for i := range request.Options.MaxTokens {
		if err := ctx.Err(); err != nil {
			return err
		}

		request.Sampler.AppendToken(sample)
		nextSample, nextLogprobs = step(sample)

		if i == 0 {
			mlx.Eval(sample)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		output := int32(sample.Int())
		session.outputs = append(session.outputs, output)

		if r.Tokenizer.IsEOS(output) {
			final.DoneReason = 0
			final.EvalCount = i
			break
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case request.Responses <- CompletionResponse{
			Content: r.Decode(output, &b),
		}:
		}

		mlx.Unpin(sample, logprobs)
		sample, logprobs = nextSample, nextLogprobs
		nextSample, nextLogprobs = nil, nil

		if i%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalDuration = time.Since(now)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

func (r Runner) Decode(sample int32, b *bytes.Buffer) string {
	token := r.Tokenizer.Decode([]int32{sample})

	if _, err := b.WriteString(token); err != nil {
		slog.Error("Failed to write token to buffer", "error", err)
		return ""
	}

	return flushValidUTF8Prefix(b)
}
