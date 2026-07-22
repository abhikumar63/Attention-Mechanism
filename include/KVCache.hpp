#ifndef KV_CACHE_HPP
#define KV_CACHE_HPP

#include "Tensor.hpp"
#include <optional>

struct KVCache {
    std::optional<Tensor> k_cache;
    std::optional<Tensor> v_cache;

    // Resets the cache buffers completely (e.g., between different prompts)
    void clear() {
        k_cache.reset();
        v_cache.reset();
    }

    // Updates the cache with new incoming token projections
    void update(const Tensor& new_k, const Tensor& new_v) {
        if (!k_cache.has_value() || !v_cache.has_value()) {
            // Prefill Phase: Cache is empty, initialize it with the prompt projections
            k_cache = new_k;
            v_cache = new_v;
        } else {
            // Decode Phase: Dynamically append the single new token projection along Dim 0
            k_cache = k_cache->concatenate(new_k);
            v_cache = v_cache->concatenate(new_v);
        }
    }

    // Inspection helper to verify cache state shapes
    void print_status(const std::string& label) const {
        if (k_cache.has_value() && v_cache.has_value()) {
            std::cout << "[" << label << "] Cache Active - Size: " 
                      << k_cache->shape_[0] << " tokens (Dim: " << k_cache->shape_[1] << ")\n";
        } else {
            std::cout << "[" << label << "] Cache is currently empty.\n";
        }
    }
};

#endif // KV_CACHE_HPP