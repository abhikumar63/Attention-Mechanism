#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include "Tensor.hpp"
#include "KVCache.hpp"
#include <cmath>
#include <stdexcept>

class Attention {
public:
    static Tensor compute(const Tensor& Q, const Tensor& K, const Tensor& V, 
                          KVCache* cache = nullptr, bool use_causal_mask = false) {
        
        // Dynamic pointers to hold whatever matrices will be used for the final dot product
        const Tensor* final_k = &K;
        const Tensor* final_v = &V;

        // If a KV cache is provided, update it and point to the aggregated historical tensors
        if (cache != nullptr) {
            cache->update(K, V);
            final_k = &(*(cache->k_cache));
            final_v = &(*(cache->v_cache));
        }

        // Integrity checks based on the chosen operational state
        if (Q.shape_[1] != final_k->shape_[1]) {
            throw std::invalid_argument("Dimension mismatch: Q and K head dimensions must match.");
        }
        if (final_k->shape_[0] != final_v->shape_[0]) {
            throw std::invalid_argument("Dimension mismatch: Cached K and V sequence lengths must match.");
        }

        float d_k = static_cast<float>(final_k->shape_[1]);

        // 1. O(1) Zero-Copy Transpose (Strides updated natively)
        Tensor K_T = final_k->transpose(0, 1);

        // 1. Attention Scores = Q * K^T
        // If caching: [1, head_dim] * [head_dim, current_seq_len] -> [1, current_seq_len]
        Tensor scores = Q.matmul(K_T);

        // 2. Scale scores
        scores.scale(1.0f / std::sqrt(d_k));

        // 3. Causal Masking (Only relevant if we are processing multiple tokens at once)
        if (use_causal_mask && scores.shape_[0] > 1) {
            scores.apply_causal_mask();
        }

        // 4. Softmax normalization over the row
        scores.softmax();

        // 5. Context Output = Scores * V
        // If caching: [1, current_seq_len] * [current_seq_len, head_dim] -> [1, head_dim]
        Tensor context = scores.matmul(*final_v);
        return context;
    }
};

#endif // ATTENTION_HPP