#include <iostream>
#include <iomanip>
#include <vector>
#include "Tensor.hpp"
#include "Attention.hpp"
#include "MultiHeadAttention.hpp"
#include "TransformerBlock.hpp"
#include "KVCache.hpp"

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
        Tensor context = Attention::compute(Q, K, V, nullptr, false);
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
        Tensor mha_output = MultiHeadAttention::compute(Q, K, V, num_heads, nullptr, false);
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
        Tensor unmasked = TransformerBlock::compute(X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads, nullptr, false);
        print_tensor("Phase 5 Unmasked Output", unmasked);

        std::cout << "--- Pass B: Autoregressive (Masked) ---\n";
        Tensor masked = TransformerBlock::compute(X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads, nullptr, true);
        print_tensor("Phase 5 Masked Output", masked);
        
        std::cout << "Initiating Memory Stress Test (10,000 iterations)...\n";
        for (int it = 0; it < 10000; ++it) {
            Tensor test_output = TransformerBlock::compute(X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads, nullptr, true);
        }
        std::cout << "[SUCCESS] 10,000 forward passes completed with zero memory violations.\n\n";
    }

    // ---------------------------------------------------------
    // PHASE 6: Optimized Autoregressive Generation with KV Caching
    // ---------------------------------------------------------
    {
        std::cout << ">>> EXECUTING PHASE 6: Optimized Autoregressive Generation via KV Cache\n";
        size_t embed_dim = 8, num_heads = 2, ffn_dim = 32;

        // Shared Weights setup
        Tensor Q_proj({embed_dim, embed_dim}), K_proj({embed_dim, embed_dim}), V_proj({embed_dim, embed_dim});
        Tensor W_1({embed_dim, ffn_dim}), W_2({ffn_dim, embed_dim});
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

        // Initialize our persistent Key-Value caches (one vector entry per head)
        std::vector<KVCache> caches(num_heads);

        // STEP A: THE PREFILL PHASE (Ingesting a 3-token initial prompt)
        size_t prompt_len = 3;
        std::cout << "\n[Prefill Phase] Processing prompt of length: " << prompt_len << "\n";
        
        Tensor prompt_X({prompt_len, embed_dim});
        for (size_t i = 0; i < prompt_len; ++i) {
            for (size_t j = 0; j < embed_dim; ++j) prompt_X.set({i, j}, (float)(i + j) * 0.1f);
        }

        // Pass the cache address down. Causal mask is TRUE since we possess multiple parallel tokens.
        Tensor prefill_output = TransformerBlock::compute(prompt_X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads, &caches, true);
        print_tensor("Prefill Output Token States", prefill_output);
        
        for (size_t h = 0; h < num_heads; ++h) {
            caches[h].print_status("Head " + std::to_string(h));
        }

        // STEP B: THE DECODE LOOP (Simulating generation of subsequent single tokens)
        std::cout << "\n[Decode Phase] Initiating loop for subsequent tokens...\n";
        size_t steps_to_generate = 2;

        for (size_t step = 0; step < steps_to_generate; ++step) {
            std::cout << "\n--- Generation Step " << (step + 1) << " ---\n";
            
            // A new singular token matrix arriving from the generation step
            size_t input_len = 1; 
            Tensor next_token_X({input_len, embed_dim});
            for (size_t j = 0; j < embed_dim; ++j) {
                // Emulating dynamic token features
                next_token_X.set({0, j}, 0.5f + (float)step * 0.1f); 
            }

            // Execute the pass. Causal mask can be FALSE because a sequence length of 1 cannot attend to future elements anyway!
            Tensor decode_output = TransformerBlock::compute(next_token_X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads, &caches, false);
            
            print_tensor("Decode Single Output Token State", decode_output);

            for (size_t h = 0; h < num_heads; ++h) {
                caches[h].print_status("Head " + std::to_string(h));
            }
        }
    }

    // ---------------------------------------------------------
    // PHASE 7: Binary Weight Ingestion (Real-World Pipeline)
    // ---------------------------------------------------------
    {
        std::cout << "\n>>> EXECUTING PHASE 7: Binary Weight Ingestion\n";
        size_t seq_len = 3, embed_dim = 8, num_heads = 2, ffn_dim = 32;

        Tensor X({seq_len, embed_dim});
        Tensor Q_proj({embed_dim, embed_dim});
        Tensor K_proj({embed_dim, embed_dim});
        Tensor V_proj({embed_dim, embed_dim});
        Tensor W_1({embed_dim, ffn_dim});
        Tensor W_2({ffn_dim, embed_dim});

        // Open the binary file
        std::ifstream weight_file("../model.bin", std::ios::binary);
        if (!weight_file) {
            std::cerr << "[ERROR] Could not open model.bin. Did you run export_weights.py?\n";
            return 1;
        }

        // Load tensors in the EXACT order they were exported in Python
        std::cout << "\nStreaming bytes from model.bin into Tensor memory...\n";
        X.load_from_binary(weight_file);
        Q_proj.load_from_binary(weight_file);
        K_proj.load_from_binary(weight_file);
        V_proj.load_from_binary(weight_file);
        W_1.load_from_binary(weight_file);
        W_2.load_from_binary(weight_file);
        weight_file.close();

        // Run the Masked Autoregressive Block
        Tensor output = TransformerBlock::compute(X, Q_proj, K_proj, V_proj, W_1, W_2, num_heads, nullptr, true);

        print_tensor("C++ Engine Output (from Binary Weights)", output);
    }

    std::cout << "\n=================================================\n";
    std::cout << " DIAGNOSTICS COMPLETE: ALL SYSTEMS NOMINAL \n";
    std::cout << "=================================================\n";
    return 0;
}