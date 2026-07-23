import torch
import torch.nn.functional as F
import numpy as np
import math

print("Generating realistic weights and exporting to model.bin...")

# Network Dimensions
seq_len, embed_dim, num_heads, ffn_dim = 3, 8, 2, 32
head_dim = embed_dim // num_heads

# 1. Generate "Trained" Weights (Normal distribution, simulating a real model)
torch.manual_seed(42) # Lock seed for reproducibility
X = torch.randn(seq_len, embed_dim)
Q_proj = torch.randn(embed_dim, embed_dim) / math.sqrt(embed_dim)
K_proj = torch.randn(embed_dim, embed_dim) / math.sqrt(embed_dim)
V_proj = torch.randn(embed_dim, embed_dim) / math.sqrt(embed_dim)
W1 = torch.randn(embed_dim, ffn_dim) / math.sqrt(embed_dim)
W2 = torch.randn(ffn_dim, embed_dim) / math.sqrt(ffn_dim)

# 2. Dump raw bytes to a binary file
# Order is strictly maintained: X, Q, K, V, W1, W2
with open("model.bin", "wb") as f:
    X.numpy().astype(np.float32).tofile(f)
    Q_proj.numpy().astype(np.float32).tofile(f)
    K_proj.numpy().astype(np.float32).tofile(f)
    V_proj.numpy().astype(np.float32).tofile(f)
    W1.numpy().astype(np.float32).tofile(f)
    W2.numpy().astype(np.float32).tofile(f)

print("[SUCCESS] Wrote contiguous float32 bytes to model.bin")

# 3. Calculate PyTorch Ground Truth to verify C++ later
Q = X @ Q_proj
K = X @ K_proj
V = X @ V_proj

Q_heads = Q.view(seq_len, num_heads, head_dim).transpose(0, 1)
K_heads = K.view(seq_len, num_heads, head_dim).transpose(0, 1)
V_heads = V.view(seq_len, num_heads, head_dim).transpose(0, 1)

scores = Q_heads @ K_heads.transpose(-2, -1) / math.sqrt(head_dim)
# Using causal mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))

probs = F.softmax(scores, dim=-1)
head_outputs = probs @ V_heads
mha_out = head_outputs.transpose(0, 1).contiguous().view(seq_len, embed_dim)

norm1 = F.layer_norm(X + mha_out, [embed_dim], eps=1e-5)
ffn_out = F.relu(norm1 @ W1) @ W2
final_output = F.layer_norm(norm1 + ffn_out, [embed_dim], eps=1e-5)

print("\n--- PyTorch Ground Truth Output ---")
for row in final_output:
    print("  ".join([f"{float(val):.4f}" for val in row]))