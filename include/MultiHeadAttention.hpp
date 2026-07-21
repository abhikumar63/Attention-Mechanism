#pragma once
#include "Tensor.hpp"
#include "Attention.hpp"
#include <vector>
#include <stdexcept>

class MultiHeadAttention {
public:
    static Tensor compute(const Tensor& Q, const Tensor& K, const Tensor& V, size_t num_heads) {
        size_t embed_dim = Q.shape_[1];
        
        if (embed_dim % num_heads != 0) {
            throw std::invalid_argument("Embedding dimension must be perfectly divisible by number of heads.");
        }

        size_t head_dim = embed_dim / num_heads;
        std::vector<Tensor> head_outputs;

        // 1. Split and compute Attention for each head independently
        for (size_t h = 0; h < num_heads; ++h) {
            size_t start_col = h * head_dim;
            
            Tensor q_head = Q.slice_cols(start_col, head_dim);
            Tensor k_head = K.slice_cols(start_col, head_dim);
            Tensor v_head = V.slice_cols(start_col, head_dim);

            // Route through your Phase 3 hardware-accelerated engine
            Tensor head_context = Attention::compute(q_head, k_head, v_head);
            head_outputs.push_back(head_context);
        }

        // 2. Concatenate all heads back into the original embedding dimension
        Tensor final_context = head_outputs[0];
        for (size_t h = 1; h < num_heads; ++h) {
            final_context = final_context.concat_cols(head_outputs[h]);
        }

        return final_context;
    }
};