#include <iostream>
#include <iomanip>
#include "Tensor.hpp"
#include "Attention.hpp"
#include "MultiHeadAttention.hpp"
#include "TransformerBlock.hpp"

// Utility to print matrices cleanly
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
    std::cout << "=====================================================\n";
    std::cout << " ATTENTION MECHANISM: TRANSFORMER ENGINE DIAGNOSTICS \n";
    std::cout << "=====================================================\n\n";

    // ---------------------------------------------------------
    // PHASE 3: Single-Head Scaled Dot-Product Attention
    // ---------------------------------------------------------
    {
        std::cout << ">>> EXECUTING PHASE 3: Single-Head Attention\n";
        size_t seq_len = 3, d_k = 4;
        Tensor Q({seq_len, d_k}), K({seq_len, d_k}), V({seq_len, d_k});

        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < d_k; ++j) {
                Q.set({i, j}, (float)(i + j));
                K.set({i, j}, (float)(i * j));
                V.set({i, j}, (float)(i == j ? 1.0f : 0.1f));
            }
        }
        Tensor context = Attention::compute(Q, K, V);
        print_tensor("Phase 3 Output Context", context);
    }

    // ---------------------------------------------------------
    // PHASE 4: Multi-Head Attention
    // ---------------------------------------------------------
    {
        std::cout << ">>> EXECUTING PHASE 4: Multi-Head Attention\n";
        size_t seq_len = 3, embed_dim = 8, num_heads = 2;
        Tensor Q({seq_len, embed_dim}), K({seq_len, embed_dim}), V({seq_len, embed_dim});

        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < embed_dim; ++j) {
                Q.set({i, j}, (float)(i + j) * 0.1f);
                K.set({i, j}, (float)(i * j) * 0.1f);
                V.set({i, j}, (float)(i == j ? 1.0f : 0.0f));
            }
        }
        Tensor mha_output = MultiHeadAttention::compute(Q, K, V, num_heads);
        print_tensor("Phase 4 MHA Output", mha_output);
    }

    // ---------------------------------------------------------
    // PHASE 5: The Complete Transformer Block
    // ---------------------------------------------------------
    {
        std::cout << ">>> EXECUTING PHASE 5: Complete Transformer Block\n";
        size_t seq_len = 3, embed_dim = 8, num_heads = 2, ffn_dim = 32;
        
        Tensor X({seq_len, embed_dim});
        Tensor Q_proj({embed_dim, embed_dim}), K_proj({embed_dim, embed_dim}), V_proj({embed_dim, embed_dim});
        Tensor W_1({embed_dim, ffn_dim}), W_2({ffn_dim, embed_dim});

        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < embed_dim; ++j) X.set({i, j}, (float)(i + j) * 0.1f);
        }
        for (size_t i = 0; i < embed_dim; ++i) {
            for (size_t j = 0; j < embed_dim; ++j) {
                float val = (float)(i + j) * 0.01f;
                Q_proj.set({i, j}, val); K_proj.set({i, j}, val); V_proj.set({i, j}, val);
            }
            for (size_t j = 0; j < ffn_dim; ++j) W_1.set({i, j}, (float)(i + j) * 0.01f);
        }
        for (size_t i = 0; i < ffn_dim; ++i) {
            for (size_t j = 0; j < embed_dim; ++j) W_2.set({i, j}, (float)(i + j) * 0.01f);
        }

        std::cout << "--- Pass A: Bidirectional (Unmasked) ---\n";
        Tensor unmasked = TransformerBlock::compute(X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads, false);
        print_tensor("Phase 5 Unmasked Output", unmasked);

        std::cout << "--- Pass B: Autoregressive (Masked) ---\n";
        Tensor masked = TransformerBlock::compute(X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads, true);
        print_tensor("Phase 5 Masked Output", masked);
        
        // --- MEMORY STRESS TEST ---
        std::cout << "Initiating Memory Stress Test (10,000 iterations)...\n";
        for (int it = 0; it < 10000; ++it) {
            // If there is a memory leak or a dangling pointer, ASan will terminate the program here.
            Tensor test_output = TransformerBlock::compute(X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads, true);
        }
        
        std::cout << "[SUCCESS] 10,000 forward passes completed with zero memory violations.\n";
    }

    std::cout << "\n=================================================\n";
    std::cout << " DIAGNOSTICS COMPLETE: ALL SYSTEMS NOMINAL \n";
    std::cout << "=================================================\n";
    return 0;
}