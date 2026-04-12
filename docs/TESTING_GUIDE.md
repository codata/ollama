# Testing Nitro Performance

Three specialized test suites are provided to validate stability and performance.

## 🏃 1. `test_bypass.sh` (Integration)
The primary functional test.
*   **Purpose**: Validates that a specific prompt can be learned in Run 1 and bypassed in Run 2.
*   **Usage**: `./test_bypass.sh`
*   **Success Criteria**: Log reports "Semantic bypass is active!".

## 🏎️ 2. `test_parallel.sh` (Stability)
Stress tests the scheduler and detached syncing.
*   **Purpose**: Fires 10 simultaneous requests to the same underlying model.
*   **Usage**: `./test_parallel.sh`
*   **Success Criteria**: All 10 requests finish successfully with concurrent Nitro triggers and no server panics.

## 🌍 3. `test_diverse.sh` (Benchmark)
Evaluates efficacy across different content types.
*   **Purpose**: Runs 5 diverse prompts (Physics, Coding, History, Health, Computer Science).
*   **Usage**: `./test_diverse.sh`
*   **Success Criteria**: Demonstrates wall-clock speedups across multiple domains (e.g., 3x speedup on Python code).

## 💡 Pro Tip
When testing for absolute reproducibility, set **`temperature: 0.0`** in your API request to maximize Nitro cache hits.
