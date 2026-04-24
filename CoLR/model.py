import torch
from .low_rank import LowRankEmbedding, LowRankLinear


class NeuralCollaborativeFiltering(torch.nn.Module):
    """
    CoLR-NCF: item embeddings and shared MLP weights are low-rank factorised.
    User embeddings remain full-rank and stay on the client (never uploaded).
    """

    def __init__(self, user_num: int, item_num: int,
                 predictive_factor: int = 32, rank: int = 16):
        super().__init__()
        pf = predictive_factor
        self.rank = rank

        # ── User embeddings: full-rank, private, never communicated ──────────
        self.mlp_user_embeddings = torch.nn.Embedding(user_num, 2 * pf)
        self.gmf_user_embeddings = torch.nn.Embedding(user_num, 2 * pf)

        # ── Item embeddings: LOW-RANK, aggregated on server (CoLR) ───────────
        self.mlp_item_embeddings = LowRankEmbedding(item_num, 2 * pf, rank)
        self.gmf_item_embeddings = LowRankEmbedding(item_num, 2 * pf, rank)

        # ── MLP tower: LOW-RANK linear layers (CoLR) ─────────────────────────
        self.mlp = torch.nn.Sequential(
            LowRankLinear(4 * pf, 2 * pf, rank), torch.nn.ReLU(),
            LowRankLinear(2 * pf, pf,     rank), torch.nn.ReLU(),
            LowRankLinear(pf,     pf // 2, rank), torch.nn.ReLU(),
        )

        # ── Output heads: regular linear (small, not a bottleneck) ───────────
        self.gmf_out      = torch.nn.Linear(2 * pf,   1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, 2 * pf))
        self.mlp_out      = torch.nn.Linear(pf // 2,  1)
        self.output_logits = torch.nn.Linear(pf,      1)
        self.model_blending = 0.5

        self.initialize_weights()
        self.join_output_weights()

    # ── Weight init ───────────────────────────────────────────────────────────

    def initialize_weights(self):
        torch.nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        # LowRankEmbedding / LowRankLinear self-initialise in __init__
        torch.nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output  = self.mlp_forward(user_id, item_id)
        return self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)

    def gmf_forward(self, user_id, item_id):
        return torch.mul(self.gmf_user_embeddings(user_id),
                         self.gmf_item_embeddings(item_id))   # LowRankEmbedding

    def mlp_forward(self, user_id, item_id):
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_item_embeddings(item_id)          # LowRankEmbedding
        return self.mlp(torch.cat([user_emb, item_emb], dim=1))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    def layer_setter(self, src, dst):
        for s, d in zip(src.parameters(), dst.parameters()):
            d.data[:] = s.data[:]

    def load_server_weights(self, server_model):
        """Copy aggregated low-rank item factors from server to this client."""
        self.layer_setter(server_model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(server_model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(server_model.mlp,                 self.mlp)
        self.layer_setter(server_model.gmf_out,             self.gmf_out)
        self.layer_setter(server_model.mlp_out,             self.mlp_out)
        self.layer_setter(server_model.output_logits,       self.output_logits)


if __name__ == '__main__':
    ncf = NeuralCollaborativeFiltering(100, 100, predictive_factor=64, rank=16)
    total  = sum(p.numel() for p in ncf.parameters())
    item   = (sum(p.numel() for p in ncf.mlp_item_embeddings.parameters()) +
              sum(p.numel() for p in ncf.gmf_item_embeddings.parameters()))
    print(ncf)
    print(f"Total params : {total:,}")
    print(f"Item  params : {item:,}  (low-rank, communicated)")