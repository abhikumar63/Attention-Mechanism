# C++ Transformer & Attention Engine (Practice Project)

A from-scratch, low-level C++ implementation of a complete Transformer Block. This project bypasses high-level wrappers like PyTorch to explicitly manage contiguous memory matrices, compute Scaled Dot-Product Attention, and execute multi-dimensional tensor math natively on Apple Silicon.

## Core Features

* **Custom Tensor Infrastructure:** A ground-up `Tensor` class separating logical shape/stride metadata from physical memory, allowing for $O(1)$ zero-copy transpositions.
* **Hardware Acceleration:** Direct bindings to Apple's **Accelerate Framework** (`cblas_sgemm`) to offload matrix multiplications (GEMM) from the CPU to native silicon.
* **Multi-Head Attention (MHA):** Parallelized attention heads with precise multi-dimensional splitting, concatenation, and safe-softmax numerical stabilization.
* **Complete Transformer Block:** Includes Residual Connections, standard deviation-based Layer Normalization, and a Feed-Forward Network (FFN).
* **PyTorch Numerical Parity:** Validated against a reference PyTorch implementation. The C++ engine achieves exact floating-point parity within a strict tolerance limit.
* **Memory Safe:** Fully profiled and stress-tested using LLVM AddressSanitizer (ASan) to guarantee zero memory leaks or dangling pointers during massive iteration loops.

## Project Structure

```text
.
├── CMakeLists.txt         # ARM64 & Accelerate-bound build configuration
├── build.sh               # Execution pipeline script
├── validate.py            # PyTorch ground-truth validation script
├── include/
│   ├── Tensor.hpp             # Contiguous memory management & GEMM math
│   ├── Attention.hpp          # Scaled dot-product attention equation
│   ├── MultiHeadAttention.hpp # Head splitting and tensor merging
│   └── TransformerBlock.hpp   # Residuals, LayerNorm, and FFN wrapper
└── src/
    └── main.cpp               # Execution entry point and memory stress tests
```

## Environment & Prerequisites

This engine is optimized for macOS development and natively targets the ARM64 architecture. 

* **Hardware:** Apple Silicon (M1/M2/M3/M4) recommended.
* **Build System:** CMake (v3.20+) and an Apple Clang C++17 compiler.
* **Environment:** Verified using Ghostty / Zsh, managed via `pyenv` for the Python validation scripts. 
* **IDE Compatibility:** Configurations naturally support standard IDEs like Google Antigravity.

## Build and Execution

To compile the engine, link the Accelerate framework, and run the memory stress tests, execute the automated build script:

```bash
chmod +x build.sh
./build.sh
```

To verify numerical parity against the PyTorch ground truth, ensure you have `torch` installed via your Python environment and run:

```bash
python3 validate.py
```

## Upcoming Phases

* **Autoregressive Causal Masking:** Implementing lower-triangular masking to prevent future-token visibility during generative inference.
* *(Future)* Ingestion streams and dynamic vector spaces.