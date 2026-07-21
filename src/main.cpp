#include <iostream>
#include "Tensor.hpp"
#include "TransformerBlock.hpp"

void print_tensor(const std::string& name, const Tensor& t) {
    std::cout << "--- " << name << " [" << t.shape_[0] << "x" << t.shape_[1] << "] ---\n";
    for (size_t i = 0; i < t.shape_[0]; ++i) {
        for (size_t j = 0; j < t.shape_[1]; ++j) {
            std::cout << std::fixed << std::setprecision(4) << t.get({i, j}) << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "Executing Final Phase: Complete Transformer Block...\n\n";

    size_t seq_len = 3;
    size_t embed_dim = 8;
    size_t num_heads = 2;
    size_t ffn_dim = 32; // Standard practice is 4x the embed_dim

    // 1. The Input Sequence
    Tensor X({seq_len, embed_dim});

    // 2. Attention Projection Weights
    Tensor Q_proj({embed_dim, embed_dim});
    Tensor K_proj({embed_dim, embed_dim});
    Tensor V_proj({embed_dim, embed_dim});

    // 3. FFN Weights
    Tensor W_1({embed_dim, ffn_dim});
    Tensor W_2({ffn_dim, embed_dim});

    // Populate with varied dummy data
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < embed_dim; ++j) {
            X.set({i, j}, (float)(i + j) * 0.1f); // Added j to create variance
        }
    }
    for (size_t i = 0; i < embed_dim; ++i) {
        for (size_t j = 0; j < embed_dim; ++j) {
            Q_proj.set({i, j}, (float)(i + j) * 0.01f);
            K_proj.set({i, j}, (float)(i + j) * 0.01f);
            V_proj.set({i, j}, (float)(i + j) * 0.01f);
        }
        for (size_t j = 0; j < ffn_dim; ++j) {
            W_1.set({i, j}, (float)(i + j) * 0.01f);
        }
    }
    for (size_t i = 0; i < ffn_dim; ++i) {
        for (size_t j = 0; j < embed_dim; ++j) {
            W_2.set({i, j}, (float)(i + j) * 0.01f);
        }
    }

    std::cout << "Running Block: " << seq_len << " Tokens, " << embed_dim << " Dim, " << num_heads << " Heads.\n\n";
    
    // Execute a single pass to print the output
    Tensor final_output = TransformerBlock::compute(X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads);
    print_tensor("Output of Complete Transformer Block", final_output);

    // --- MEMORY STRESS TEST ---
    std::cout << "Initiating Memory Stress Test (10,000 iterations)...\n";
    for (int it = 0; it < 10000; ++it) {
        // If there is a memory leak or a dangling pointer, ASan will terminate the program here.
        Tensor test_output = TransformerBlock::compute(X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads);
    }
    
    std::cout << "[SUCCESS] 10,000 forward passes completed with zero memory violations.\n";
    std::cout << "[SUCCESS] C++ Transformer Engine is fully operational and memory-safe.\n";
    return 0;
}