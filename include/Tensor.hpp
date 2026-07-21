#pragma once
#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <Accelerate/Accelerate.h>

class Tensor {
public:
    // The physical 1D memory buffer, shared across any "views" (like transposed versions)
    std::shared_ptr<std::vector<float>> data_;
    
    // The logical dimensions (e.g., [Batch, Heads, SeqLen, HeadDim])
    std::vector<size_t> shape_;
    
    // The step-size needed to move 1 position along each dimension
    std::vector<size_t> strides_;

    // 1. Primary Constructor: Allocates new contiguous memory
    Tensor(const std::vector<size_t>& shape) : shape_(shape) {
        size_t total_size = 1;
        strides_.resize(shape.size());
        
        // Calculate Row-Major strides (C-style contiguous)
        // A shape of [3, 4] will have strides [4, 1]
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides_[i] = total_size;
            total_size *= shape[i];
        }
        
        // Allocate zero-initialized memory
        data_ = std::make_shared<std::vector<float>>(total_size, 0.0f);
    }

    // 2. View Constructor: Creates a new perspective on existing memory
    Tensor(std::shared_ptr<std::vector<float>> data, 
           const std::vector<size_t>& shape, 
           const std::vector<size_t>& strides) 
        : data_(data), shape_(shape), strides_(strides) {}

    // 3. Zero-Copy Transpose
    Tensor transpose(size_t dim0, size_t dim1) const {
        if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
            throw std::out_of_range("Transpose dimensions out of bounds");
        }

        std::vector<size_t> new_shape = shape_;
        std::vector<size_t> new_strides = strides_;
        
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);
        
        return Tensor(data_, new_shape, new_strides);
    }

    // Utility: Calculate the flat index in the 1D array from multi-dimensional coordinates
    size_t get_flat_index(const std::vector<size_t>& indices) const {
        size_t flat_idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            flat_idx += indices[i] * strides_[i];
        }
        return flat_idx;
    }

    // Utility: Set a value (mostly for testing setup)
    void set(const std::vector<size_t>& indices, float val) {
        (*data_)[get_flat_index(indices)] = val;
    }

    // Utility: Get a value
    float get(const std::vector<size_t>& indices) const {
        return (*data_)[get_flat_index(indices)];
    }

    Tensor matmul(const Tensor& other) const {
        if (shape_.size() != 2 || other.shape_.size() != 2) {
            throw std::invalid_argument("Matmul currently supports 2D tensors only.");
        }
        
        int M = shape_[0];
        int K = shape_[1];
        int K2 = other.shape_[0];
        int N = other.shape_[1];

        if (K != K2) {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
        }

        // Create the output tensor with shape [M, N]
        Tensor result({(size_t)M, (size_t)N});

        // Determine if our zero-copy transpose was used by checking strides
        // A standard row-major matrix has stride[1] == 1.
        // If stride[0] == 1, it means the matrix was transposed.
        CBLAS_TRANSPOSE transA = (strides_[1] == 1) ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE transB = (other.strides_[1] == 1) ? CblasNoTrans : CblasTrans;

        // Leading dimensions (stride of the major axis in physical memory)
        int lda = (transA == CblasNoTrans) ? strides_[0] : strides_[1];
        int ldb = (transB == CblasNoTrans) ? other.strides_[0] : other.strides_[1];
        int ldc = result.strides_[0];

        // Execute hardware-accelerated GEMM (General Matrix Multiply)
        // C = alpha * A * B + beta * C
        cblas_sgemm(CblasRowMajor, transA, transB, 
                    M, N, K, 
                    1.0f,                         // alpha
                    this->data_->data(), lda,     // Matrix A and leading dim
                    other.data_->data(), ldb,     // Matrix B and leading dim
                    0.0f,                         // beta
                    result.data_->data(), ldc);   // Matrix C and leading dim

        return result;
    }

    void softmax() {
        if (shape_.size() != 2) {
            throw std::invalid_argument("Softmax currently supports 2D tensors.");
        }

        int rows = shape_[0];
        int cols = shape_[1];

        for (int i = 0; i < rows; ++i) {
            // 1. Find the max value in the current row for numerical stability
            float max_val = -INFINITY;
            for (int j = 0; j < cols; ++j) {
                float val = get({(size_t)i, (size_t)j});
                if (val > max_val) max_val = val;
            }

            // 2. Compute exponents and track the sum
            float sum_exp = 0.0f;
            for (int j = 0; j < cols; ++j) {
                float val = get({(size_t)i, (size_t)j});
                float e_val = std::exp(val - max_val); // Safe exponent
                set({(size_t)i, (size_t)j}, e_val);
                sum_exp += e_val;
            }

            // 3. Normalize to create a probability distribution (sum = 1.0)
            for (int j = 0; j < cols; ++j) {
                float e_val = get({(size_t)i, (size_t)j});
                set({(size_t)i, (size_t)j}, e_val / sum_exp);
            }
        }
    }

    // Element-wise scalar multiplication (used for scaling by 1/sqrt(d_k))
    void scale(float factor) {
        for (size_t i = 0; i < data_->size(); ++i) {
            (*data_)[i] *= factor;
        }
    }

    // Extract a sub-matrix along the columns (used to split into Attention Heads)
    Tensor slice_cols(size_t start_col, size_t num_cols) const {
        if (shape_.size() != 2) throw std::invalid_argument("Slice requires 2D tensor.");
        if (start_col + num_cols > shape_[1]) throw std::out_of_range("Slice out of bounds.");

        Tensor result({shape_[0], num_cols});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < num_cols; ++j) {
                result.set({i, j}, get({i, start_col + j}));
            }
        }
        return result;
    }

    // Merge another tensor horizontally (used to concatenate Heads back together)
    Tensor concat_cols(const Tensor& other) const {
        if (shape_.size() != 2 || other.shape_.size() != 2) throw std::invalid_argument("Concat requires 2D.");
        if (shape_[0] != other.shape_[0]) throw std::invalid_argument("Row counts must match.");

        Tensor result({shape_[0], shape_[1] + other.shape_[1]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            // Copy left tensor
            for (size_t j = 0; j < shape_[1]; ++j) {
                result.set({i, j}, get({i, j}));
            }
            // Copy right tensor
            for (size_t j = 0; j < other.shape_[1]; ++j) {
                result.set({i, j + shape_[1]}, other.get({i, j}));
            }
        }
        return result;
    }

    // Element-wise Addition (Residual Connection)
    Tensor add(const Tensor& other) const {
        if (shape_ != other.shape_) throw std::invalid_argument("Shapes must match for addition.");
        Tensor result(shape_);
        for (size_t i = 0; i < data_->size(); ++i) {
            result.data_->at(i) = data_->at(i) + other.data_->at(i);
        }
        return result;
    }

    // ReLU Activation Function (in-place)
    void relu() {
        for (size_t i = 0; i < data_->size(); ++i) {
            if (data_->at(i) < 0.0f) data_->at(i) = 0.0f;
        }
    }

    // Layer Normalization (in-place along the last dimension)
    void layer_norm() {
        if (shape_.size() != 2) throw std::invalid_argument("LayerNorm requires 2D tensor.");
        int rows = shape_[0];
        int cols = shape_[1];

        for (int i = 0; i < rows; ++i) {
            // 1. Calculate Mean
            float mean = 0.0f;
            for (int j = 0; j < cols; ++j) mean += get({(size_t)i, (size_t)j});
            mean /= cols;

            // 2. Calculate Variance
            float var = 0.0f;
            for (int j = 0; j < cols; ++j) {
                float diff = get({(size_t)i, (size_t)j}) - mean;
                var += diff * diff;
            }
            var /= cols;
            
            // 3. Normalize (add epsilon 1e-5 to prevent division by zero)
            float std_dev = std::sqrt(var + 1e-5f);
            for (int j = 0; j < cols; ++j) {
                float normalized = (get({(size_t)i, (size_t)j}) - mean) / std_dev;
                set({(size_t)i, (size_t)j}, normalized);
            }
        }
    }
};