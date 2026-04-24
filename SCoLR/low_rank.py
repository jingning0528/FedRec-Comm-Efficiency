"""
Low-rank building blocks for CoLR
(Communication-efficient Low-Rank Federated Recommendation).

Reference: "Towards Efficient Communication and Secure Federated
            Recommendation System via Low-rank Training"

Key idea
--------
Instead of communicating a full matrix W ∈ R^{m×n}, represent it as
    W  =  A @ B.T,   A ∈ R^{m×r},  B ∈ R^{n×r},  r << min(m, n).
Only the low-rank factors are communicated/aggregated, saving
    (m*n) / (m*r + n*r)   ≈   min(m,n) / r   ×  in communication.
"""

import torch
import torch.nn.functional as F


class LowRankEmbedding(torch.nn.Module):
    """
    Embedding table approximated as  E = A @ B.T
        A : Embedding(num_embeddings, rank)   ← per-item / per-user factors
        B : Parameter(embedding_dim, rank)    ← shared basis

    Forward output is identical in shape to a regular nn.Embedding lookup.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.rank           = rank

        self.A = torch.nn.Embedding(num_embeddings, rank)   # |·| × r
        self.B = torch.nn.Parameter(torch.empty(embedding_dim, rank))  # d × r

        torch.nn.init.normal_(self.A.weight, std=0.01)
        torch.nn.init.normal_(self.B,        std=0.01)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # (batch, rank) @ (rank, d) → (batch, d)
        return self.A(idx) @ self.B.T

    @property
    def weight(self) -> torch.Tensor:
        """Full (num_embeddings, embedding_dim) matrix — used for compatibility."""
        return self.A.weight @ self.B.T


class LowRankLinear(torch.nn.Module):
    """
    Linear layer with weight matrix  W = A @ B.T
        A : Parameter(out_features, rank)
        B : Parameter(in_features,  rank)

    Drop-in replacement for nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank

        self.A    = torch.nn.Parameter(torch.empty(out_features, rank))
        self.B    = torch.nn.Parameter(torch.empty(in_features,  rank))
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None

        torch.nn.init.xavier_uniform_(self.A)
        torch.nn.init.xavier_uniform_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.A @ self.B.T           # (out, in)
        return F.linear(x, W, self.bias)