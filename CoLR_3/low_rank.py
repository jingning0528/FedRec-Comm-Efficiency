"""
Low-rank building blocks for CoLR
(Communication-efficient Low-Rank Federated Recommendation).

Reference: "Towards Efficient Communication and Secure Federated
            Recommendation System via Low-rank Training" — Algorithm 1

Key idea
--------
Each round t:
  B(t) ~ D_B   (re-sampled each round, frozen on client)
  Q(t,0)_u = Q(t-1) + B(t-1) @ A(t)   ← merge previous delta into dense Q
  A(t,0)_u = 0                          ← re-initialized to zero
  Upload: A_u (full, all items)
  Aggregate: A(t+1) = Σ (N_u/N) * A_u
"""

import math
import torch
import torch.nn.functional as F


class CoLREmbedding(torch.nn.Module):
    """
    CoLR item embedding (Algorithm 1):
        effective_emb(i) = Q_base[i]  +  A[i] @ B.T

    Q_base  : (num_embeddings, embedding_dim)  dense, frozen on client (merged Q)
    A       : (num_embeddings, rank)            trainable delta, init 0 each round
    B       : (embedding_dim,  rank)            random each round, frozen on client
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.rank           = rank

        # Dense merged Q — received from server, frozen on client
        self.Q_base = torch.nn.Embedding(num_embeddings, embedding_dim)
        torch.nn.init.normal_(self.Q_base.weight, std=0.01)
        self.Q_base.weight.requires_grad_(False)

        # Low-rank delta A — trained each round, init to 0
        self.A = torch.nn.Embedding(num_embeddings, rank)
        torch.nn.init.zeros_(self.A.weight)

        # Shared basis B — re-sampled by server, frozen on client
        self.B = torch.nn.Parameter(torch.empty(embedding_dim, rank))
        torch.nn.init.normal_(self.B, std=1.0 / math.sqrt(rank))
        self.B.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.Q_base(idx) + self.A(idx) @ self.B.T

    def reset_A(self):
        """Re-initialize A to zero at start of each round (Alg 1, line 10)."""
        torch.nn.init.zeros_(self.A.weight)

    def freeze_for_local_train(self):
        """Freeze Q_base and B; only A is trainable (Alg 1, line 11)."""
        self.Q_base.weight.requires_grad_(False)
        self.B.requires_grad_(False)
        self.A.weight.requires_grad_(True)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)


class LowRankLinear(torch.nn.Module):
    """
    Linear layer  W = A @ B.T   (used for MLP tower).
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
        W = self.A @ self.B.T
        return F.linear(x, W, self.bias)