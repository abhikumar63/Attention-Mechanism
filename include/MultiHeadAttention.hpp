#ifndef MULTI_HEAD_ATTENTION_HPP
#define MULTI_HEAD_ATTENTION_HPP

#include "Tensor.hpp"
#include "Attention.hpp"
#include "KVCache.hpp"
#include <vector>

class MultiHeadAttention {
public:
    static Tensor compute(const Tensor& Q, const Tensor& K, const Tensor& V, 
                          size_t num_heads, std::vector<KVCache>* head_caches = nullptr, 
                          bool use_causal_mask = false) {
        
        size_t seq_len = Q.shape_[0];
        size_t embed_dim = Q.shape_[1];
        
        if (embed_dim % num_heads != 0) {
            throw std::invalid_argument("Embedding dimension must be perfectly divisible by number of heads.");
        }
        
        size_t head_dim = embed_dim / num_heads;
        Tensor combined_output({seq_len, embed_dim});

        // Ensure cache vector matches head count if provided
        if (head_caches != nullptr && head_caches->size() != num_heads) {
            throw std::invalid_argument("Provided KV cache vector size must equal num_heads.");
        }

        for (size_t h = 0; h < num_heads; ++h) {
            Tensor q_head({seq_len, head_dim});
            Tensor k_head({seq_len, head_dim});
            Tensor v_head({seq_len, head_dim});

            // Extract contiguous blocks for this head
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < head_dim; ++j) {
                    size_t src_col = h * head_dim + j;
                    q_head.set({i, j}, Q.get({i, src_col}));
                    k_head.set({i, j}, K.get({i, src_col}));
                    v_head.set({i, j}, V.get({i, src_col}));
                }
            }

            // Route execution down to individual head attention calculators
            KVCache* head_cache = (head_caches != nullptr) ? &((*head_caches)[h]) : nullptr;
            Tensor head_context = Attention::compute(q_head, k_head, v_head, head_cache, use_causal_mask);

            // Write back into consolidated matrix array
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < head_dim; ++j) {
                    size_t dest_col = h * head_dim + j;
                    combined_output.set({i, dest_col}, head_context.get({i, j}));
                }
            }
        }

        return combined_output;
    }
};

#endif // MULTI_HEAD_ATTENTION_HPP