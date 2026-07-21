#pragma once
#include "Tensor.hpp"
#include <cmath>
#include <stdexcept>

class Attention {
public:
    // Computes Scaled Dot-Product Attention
    // Q, K, V are expected to be 2D tensors of shape [SeqLen, HeadDim]
    static Tensor compute(const Tensor& Q, const Tensor& K, const Tensor& V) {
        if (Q.shape_.size() != 2 || K.shape_.size() != 2 || V.shape_.size() != 2) {
            throw std::invalid_argument("Attention requires 2D matrices [SeqLen, HeadDim]");
        }

        // 1. Get dimension of the key vector (d_k)
        float d_k = static_cast<float>(K.shape_[1]);

        // 2. Transpose K to K^T
        // This is O(1) time, zero memory copied.
        Tensor K_T = K.transpose(0, 1);

        // 3. Score = Q * K^T
        Tensor scores = Q.matmul(K_T);

        // 4. Scale by 1 / sqrt(d_k)
        scores.scale(1.0f / std::sqrt(d_k));

        // 5. Apply Softmax to normalize scores into probabilities
        scores.softmax();

        // 6. Context = Scores * V
        Tensor context = scores.matmul(V);

        return context;
    }
};