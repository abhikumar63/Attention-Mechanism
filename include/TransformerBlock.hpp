#ifndef TRANSFORMER_BLOCK_HPP
#define TRANSFORMER_BLOCK_HPP

#include "Tensor.hpp"
#include "MultiHeadAttention.hpp"
#include "KVCache.hpp"
#include <vector>

class TransformerBlock {
public:
    static Tensor compute(const Tensor& X, 
                          const Tensor& Q_proj, const Tensor& K_proj, const Tensor& V_proj,
                          const Tensor& W_1, const Tensor& W_2, 
                          size_t num_heads, std::vector<KVCache>* head_caches = nullptr, 
                          bool use_causal_mask = false) {
        
        // 1. Compute projections
        Tensor Q = X.matmul(Q_proj);
        Tensor K = X.matmul(K_proj);
        Tensor V = X.matmul(V_proj);

        // 2. Multi-Head Attention Routing
        Tensor mha_output = MultiHeadAttention::compute(Q, K, V, num_heads, head_caches, use_causal_mask);

        // 3. First Residual & LayerNorm
        Tensor norm1 = X.add(mha_output);
        norm1.layer_norm();

        // 4. Feed Forward Network (Linear -> ReLU -> Linear)
        Tensor ffn1 = norm1.matmul(W_1);
        ffn1.relu();
        Tensor ffn2 = ffn1.matmul(W_2);

        // 5. Second Residual & LayerNorm
        Tensor output = norm1.add(ffn2);
        output.layer_norm();

        return output;
    }
};

#endif // TRANSFORMER_BLOCK_HPP