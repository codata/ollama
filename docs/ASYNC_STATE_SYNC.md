# Asynchronous State Synchronization

Asynchronous State Sync (ASS) is the mechanism that allows Nitro to deliver bypassed tokens at wire-speed without being blocked by the GPU's memory bandwidth limits.

## 🧱 The Problem

Even when tokens are predicted by Nitro, the GPU must still update its internal **KV Cache** to maintain consistency for the *next* token generation. On high-parameter models (20B+), this "Fast-Forwarding" (Sync) is bound by the GPU's prefill throughput (usually 50-70 tokens/s).

Without ASS, the user would receive the tokens instantly, but the request would hang until the GPU finished the sync, dragging the `eval_rate` down to hardware speeds.

## ⚡ The Solution: Asynchronous Detaching

1.  **Detached Syncing**:
    *   If Nitro predicts an **EOG (End of Generation)** token, the request is immediately "detached."
    *   The HTTP connection is closed, and the final JSON response is sent to the user.
    *   The GPU Sync continues in the background within the runner's main loop.

2.  **Early Semaphore Release**:
    *   As soon as a sequence is detached, it releases its slot in the **Global Request Semaphore**.
    *   New incoming requests can start immediately, even while the previous request's background sync is still processing.

3.  **Ghost Buffer**: 
    *   The runner maintains a `parallel + 32` sequence buffer. This allows up to 32 "ghost" sequences to perform background syncing without starving the active slots for primary generation.

## 📉 Aggregate Benefits

In high-concurrency scenarios, ASYNC State Sync allows for **Interleaved Throughput**. The total time to process a batch of 10 requests is significantly lower because the "handoff" latency between requests is minimized.
