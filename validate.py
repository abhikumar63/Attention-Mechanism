import torch
import torch.nn.functional as F
import math

def print_tensor(name, t):
    print(f"--- {name} [{t.shape[0]}x{t.shape[1]}] ---")
    for row in t:
        print("  ".join([f"{float(val):.4f}" for val in row]))
    print("")

print("=================================================")
print(" PYTORCH GROUND TRUTH DIAGNOSTICS ")
print("=================================================\n")

# ---------------------------------------------------------
# PHASE 3: Single-Head Scaled Dot-Product Attention
# ---------------------------------------------------------
print(">>> EXECUTING PHASE 3: Single-Head Attention")
seq_len, d_k = 3, 4
Q = torch.zeros(seq_len, d_k)
K = torch.zeros(seq_len, d_k)
V = torch.zeros(seq_len, d_k)

for i in range(seq_len):
    for j in range(d_k):
        Q[i, j] = i + j
        K[i, j] = i * j
        V[i, j] = 1.0 if i == j else 0.1

scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
probs = F.softmax(scores, dim=-1)
context = probs @ V
print_tensor("Phase 3 Output Context", context)

# ---------------------------------------------------------
# PHASE 4: Multi-Head Attention
# ---------------------------------------------------------
print(">>> EXECUTING PHASE 4: Multi-Head Attention")
seq_len, embed_dim, num_heads = 3, 8, 2
head_dim = embed_dim // num_heads
Q = torch.zeros(seq_len, embed_dim)
K = torch.zeros(seq_len, embed_dim)
V = torch.zeros(seq_len, embed_dim)

for i in range(seq_len):
    for j in range(embed_dim):
        Q[i, j] = (i + j) * 0.1
        K[i, j] = (i * j) * 0.1
        V[i, j] = 1.0 if i == j else 0.0

Q_heads = Q.view(seq_len, num_heads, head_dim).transpose(0, 1)
K_heads = K.view(seq_len, num_heads, head_dim).transpose(0, 1)
V_heads = V.view(seq_len, num_heads, head_dim).transpose(0, 1)

scores = Q_heads @ K_heads.transpose(-2, -1) / math.sqrt(head_dim)
probs = F.softmax(scores, dim=-1)
head_outputs = probs @ V_heads
mha_output = head_outputs.transpose(0, 1).contiguous().view(seq_len, embed_dim)
print_tensor("Phase 4 MHA Output", mha_output)

# ---------------------------------------------------------
# PHASE 5: The Complete Transformer Block
# ---------------------------------------------------------
print(">>> EXECUTING PHASE 5: Complete Transformer Block")
seq_len, embed_dim, num_heads, ffn_dim = 3, 8, 2, 32
head_dim = embed_dim // num_heads

X = torch.zeros(seq_len, embed_dim)
Q_proj = torch.zeros(embed_dim, embed_dim)
K_proj = torch.zeros(embed_dim, embed_dim)
V_proj = torch.zeros(embed_dim, embed_dim)
W1 = torch.zeros(embed_dim, ffn_dim)
W2 = torch.zeros(ffn_dim, embed_dim)

for i in range(seq_len):
    for j in range(embed_dim):
        X[i, j] = (i + j) * 0.1
for i in range(embed_dim):
    for j in range(embed_dim):
        val = (i + j) * 0.01
        Q_proj[i, j] = val; K_proj[i, j] = val; V_proj[i, j] = val  # noqa: E702
    for j in range(ffn_dim):
        W1[i, j] = (i + j) * 0.01
for i in range(ffn_dim):
    for j in range(embed_dim):
        W2[i, j] = (i + j) * 0.01

def run_transformer_block(X, is_masked):
    Q = X @ Q_proj
    K = X @ K_proj
    V = X @ V_proj

    Q_heads = Q.view(seq_len, num_heads, head_dim).transpose(0, 1)
    K_heads = K.view(seq_len, num_heads, head_dim).transpose(0, 1)
    V_heads = V.view(seq_len, num_heads, head_dim).transpose(0, 1)

    scores = Q_heads @ K_heads.transpose(-2, -1) / math.sqrt(head_dim)
    
    if is_masked:
        # Apply lower triangular mask by filling the upper triangle with -inf
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
    probs = F.softmax(scores, dim=-1)
    head_outputs = probs @ V_heads
    mha_out = head_outputs.transpose(0, 1).contiguous().view(seq_len, embed_dim)

    norm1 = F.layer_norm(X + mha_out, [embed_dim], eps=1e-5)
    ffn_out = F.relu(norm1 @ W1) @ W2
    final_output = F.layer_norm(norm1 + ffn_out, [embed_dim], eps=1e-5)
    return final_output

print("--- Pass A: Bidirectional (Unmasked) ---")
print_tensor("Phase 5 Unmasked Output", run_transformer_block(X, is_masked=False))

print("--- Pass B: Autoregressive (Masked) ---")
print_tensor("Phase 5 Masked Output", run_transformer_block(X, is_masked=True))

print("=================================================")
print(" DIAGNOSTICS COMPLETE ")
print("=================================================")