# Nitro Semantic Bypass

The Nitro Semantic Bypass is a high-performance optimization layer for Ollama that uses a probabilistic "bit-signature" index to skip redundant token generation.

## 🚀 Overview

LLMs often generate repetitive patterns or blocks of text that have already been generated in previous contexts (e.g., boilerplate code, common definitions, or structured JSON). Nitro indexes these associations and "fast-forwards" the generation when a known semantic pattern is detected.

## 🛠️ How it Works

1.  **Bit-Signature Indexing**:
    *   As the model generates tokens, Nitro maintains an 8-gram sliding window of contexts.
    *   Each 8-gram is hashed into a bitmask (signature) and indexed against the resulting token.
    *   This index is model-specific and persisted across restarts.

2.  **The Bypass Loop**:
    *   During new requests, Nitro checks if the current context matches a high-confidence signature in the index.
    *   If a match is found (>0.9 similarity), Nitro predicts the next token immediately without calling the GPU.
    *   This loop continues as long as predictions remain in the index, allowing hits of 100+ tokens in a single millisecond.

3.  **Semantic Verification**: 
    *   Even when a token is bypassed, the system ensures it is mathematically sound by verifying the prediction against the index's bitmask probability cloud.

## ⚙️ Configuration

| Environment Variable | Description |
| :--- | :--- |
| `OLLAMA_INDEX_METHOD=bit-signature` | Enables the Nitro indexing mechanism. |
| `OLLAMA_DEBUG=1` | Prints Nitro trigger events and token counts to the logs. |

## 📈 Performance Impact

*   **Latency**: First-token latency for repeated content is reduced to near-zero.
*   **Throughput**: Content that exists in the index can be "generated" at a rate exceeding 1000 tokens/second (internal), blocked only by the asynchronous GPU verification sync.
