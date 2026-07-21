import torch
import torch.nn.functional as F

# Dimensions (Matching your C++ main.cpp)
seq_len, embed_dim, num_heads, ffn_dim = 3, 8, 2, 32
head_dim = embed_dim // num_heads

# 1. Recreate the EXACT dummy data loops from C++
X = torch.zeros(seq_len, embed_dim)
for i in range(seq_len):
    for j in range(embed_dim):
        X[i, j] = (i + j) * 0.1

Q_proj = torch.zeros(embed_dim, embed_dim)
K_proj = torch.zeros(embed_dim, embed_dim)
V_proj = torch.zeros(embed_dim, embed_dim)
for i in range(embed_dim):
    for j in range(embed_dim):
        val = (i + j) * 0.01
        Q_proj[i, j] = val; K_proj[i, j] = val; V_proj[i, j] = val  # noqa: E702

W1 = torch.zeros(embed_dim, ffn_dim)
for i in range(embed_dim):
    for j in range(ffn_dim):
        W1[i, j] = (i + j) * 0.01

W2 = torch.zeros(ffn_dim, embed_dim)
for i in range(ffn_dim):
    for j in range(embed_dim):
        W2[i, j] = (i + j) * 0.01

# --- SUB-LAYER 1: Multi-Head Attention ---
Q = X @ Q_proj
K = X @ K_proj
V = X @ V_proj

# Split into heads and transpose: [seq_len, heads, head_dim] -> [heads, seq_len, head_dim]
Q_heads = Q.view(seq_len, num_heads, head_dim).transpose(0, 1)
K_heads = K.view(seq_len, num_heads, head_dim).transpose(0, 1)
V_heads = V.view(seq_len, num_heads, head_dim).transpose(0, 1)

# Scaled Dot-Product Attention
scores = Q_heads @ K_heads.transpose(-2, -1) / (head_dim ** 0.5)
probs = F.softmax(scores, dim=-1)
head_outputs = probs @ V_heads

# Concatenate heads back together
mha_out = head_outputs.transpose(0, 1).contiguous().view(seq_len, embed_dim)

# Residual + LayerNorm 1 (eps=1e-5 matches our C++ implementation)
norm1 = F.layer_norm(X + mha_out, [embed_dim], eps=1e-5)

# --- SUB-LAYER 2: Feed-Forward Network ---
ffn_hidden = F.relu(norm1 @ W1)
ffn_out = ffn_hidden @ W2

# Residual + LayerNorm 2
final_output = F.layer_norm(norm1 + ffn_out, [embed_dim], eps=1e-5)

# Display the ground truth
print("--- PyTorch Ground Truth [3x8] ---")
for row in final_output:
    print("  ".join([f"{val:.4f}" for val in row]))
print("\n")