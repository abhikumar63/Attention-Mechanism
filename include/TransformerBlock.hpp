#pragma once
#include "Tensor.hpp"
#include "MultiHeadAttention.hpp"

class TransformerBlock {
public:
    static Tensor compute(const Tensor& X, 
                          const Tensor& Q_proj, const Tensor& K_proj, const Tensor& V_proj,
                          const Tensor& W_1, const Tensor& W_2, 
                          size_t num_heads) {
        
        // --- SUB-LAYER 1: Multi-Head Attention ---
        // Project input X into Q, K, V
        Tensor Q = X.matmul(Q_proj);
        Tensor K = X.matmul(K_proj);
        Tensor V = X.matmul(V_proj);

        Tensor mha_output = MultiHeadAttention::compute(Q, K, V, num_heads);

        // Residual Connection & Layer Normalization
        Tensor norm_1 = X.add(mha_output);
        norm_1.layer_norm();

        // --- SUB-LAYER 2: Feed-Forward Network ---
        Tensor ffn_hidden = norm_1.matmul(W_1);
        
        // Non-linear activation
        ffn_hidden.relu();
        
        // Project back to original dimension
        Tensor ffn_output = ffn_hidden.matmul(W_2);

        // Residual Connection & Layer Normalization
        Tensor final_output = norm_1.add(ffn_output);
        final_output.layer_norm();

        return final_output;
    }
};