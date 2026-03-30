# CODATA Ollama: Optimized Inference & ODRL Integration

This document outlines the custom modifications and optimization procedures for the **CODATA-enhanced Ollama** distribution. This version focuses on maximum performance for professional NVIDIA hardware (A6000/A100) and native data sovereignty via **ODRL (Open Digital Rights Language)**.

---

## 🛠️ Manual Build Instructions

### 1. Prerequisites
- **Go**: 1.22 or higher
- **CMake**: 3.21 or higher
- **GCC/G++**: 11 or higher
- **CUDA Toolkit**: 12.x or 13.0

### 2. Environment Setup
If you are installing CUDA to a non-standard path (e.g., a RAID drive), set your paths accordingly:
```bash
export CUDA_HOME=/nas/C19M/cuda-toolkit/13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3. Critical Fixes for Custom Filesystems
If building on **NAS, RAID, or exFAT** volumes that do not support symbolic links:
1.  **Disable Versioned SONAME**: Add `set(CMAKE_PLATFORM_NO_VERSIONED_SONAME 1)` to the top of `CMakeLists.txt`.
2.  **Avoid VCS Errors**: Use the `-buildvcs=false` flag during the Go build phase.

### 4. Build Commands

#### **For CUDA 12.x**
```bash
find . -name "CMakeCache.txt" -delete
cmake -S . -B build -DGGML_CUDA=ON -DCUDAToolkit_ROOT=$CUDA_HOME
cmake --build build --parallel $(nproc)
go build -buildvcs=false -o ollama_codata .
```

#### **For CUDA 13.0 (Recommended)**
CUDA 13 provides superior register management for Ampere/Lovelace architectures.
```bash
# Clean metadata first to avoid "binary directory already used" errors
find build/ -name "CMakeFiles" -type d -exec rm -rf {} +
cmake -S . -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_VARIANTS=ON
cmake --build build --parallel $(nproc)
go build -buildvcs=false -o ollama_rebuilt .
```

---

## ⚡ Inference Engine Optimizations

The CODATA distribution implements several TRIZ-inspired optimizations to achieve **up to 10x speedup** over generic deployments.

### 1. Hardware Isolation (GPU Dedication)
To avoid bottlenecks caused by slower secondary GPUs (like the T400 4GB) or desktop UI overhead, the inference engine should be isolated to the primary compute card:
```bash
export CUDA_VISIBLE_DEVICES=1  # Target the RTX A6000
```

### 2. Flash Attention & Kernel Fusion
- **Enabled by Default**: The build process specifically targets native instruction sets.
- **Result**: Prompt evaluation speeds (preprocessing) exceed **4300 tokens/s** on A6000 hardware.
- **Environment**: `export OLLAMA_FLASH_ATTENTION=1`.

### 3. High-Throughput Threading
Configured to utilize **128+ concurrent threads** for the compute graph, ensuring that the CPU never bottlenecks the GPU during complex MoE (Mixture of Experts) routing.

---

## 🛡️ ODRL & Decentralized Identity

This version integrates the **ODRL framework** for verifiable AI provenance.

### 1. Identity Discovery
The server identifies itself via a **Decentralized Identifier (DID)** stored in `~/.odrl/did.json`.
- **Endpoint**: `GET /api/did` returns the instance identity.

### 2. Automatic Metadata Attribution
Every generation and chat response includes the serving DID in the JSON metadata:
```json
{
  "model": "gpt-oss:20b",
  "response": "...",
  "did": "did:oyd:zQmcVHWDMe..."
}
```

### 3. Implementation Flow
- **Identification**: Resolved via the local ODRL wallet upon startup.
- **Verification**: Facilitates auditability and compliance with ODRL Usage Agreements in distributed compute environments.

---

### Performance Metrics (Verified on RTX A6000)
| Model | Type | Prompt Eval | Token Gen |
| :--- | :--- | :--- | :--- |
| **Gemma 3 4B** | Dense | 4320 t/s | 151 t/s |
| **GPT-OSS 20B**| MoE/Quant | 4357 t/s | 127 t/s |

---
**CODATA Project**: *Enabling high-performance, sovereign AI infrastructure.*
