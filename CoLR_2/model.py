import torch
from .low_rank import LowRankEmbedding, LowRankLinear


class NeuralCollaborativeFiltering(torch.nn.Module):
    """
    CoLR-NCF (paper-faithful):
      - B (shared basis) is broadcast from server, FROZEN on client each round.
      - A_u (item factors) is the only item-side parameter clients train & upload.
      - User embeddings are full-rank, private, never communicated.
    """

    def __init__(self, user_num: int, item_num: int,
                 predictive_factor: int = 32, rank: int = 16):
        super().__init__()
        pf = predictive_factor
        self.rank = rank

        self.mlp_user_embeddings = torch.nn.Embedding(user_num, 2 * pf)
        self.gmf_user_embeddings = torch.nn.Embedding(user_num, 2 * pf)

        self.mlp_item_embeddings = LowRankEmbedding(item_num, 2 * pf, rank)
        self.gmf_item_embeddings = LowRankEmbedding(item_num, 2 * pf, rank)

        self.mlp = torch.nn.Sequential(
            LowRankLinear(4 * pf, 2 * pf, rank), torch.nn.ReLU(),
            LowRankLinear(2 * pf, pf,     rank), torch.nn.ReLU(),
            LowRankLinear(pf,     pf // 2, rank), torch.nn.ReLU(),
        )

        self.gmf_out       = torch.nn.Linear(2 * pf,  1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, 2 * pf))
        self.mlp_out       = torch.nn.Linear(pf // 2, 1)
        self.output_logits  = torch.nn.Linear(pf,     1)
        self.model_blending = 0.5

        self.initialize_weights()
        self.join_output_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        torch.nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output  = self.mlp_forward(user_id, item_id)
        return self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)

    def gmf_forward(self, user_id, item_id):
        return torch.mul(self.gmf_user_embeddings(user_id),
                         self.gmf_item_embeddings(item_id))

    def mlp_forward(self, user_id, item_id):
        return self.mlp(torch.cat([self.mlp_user_embeddings(user_id),
                                   self.mlp_item_embeddings(item_id)], dim=1))

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    def layer_setter(self, src, dst):
        for s, d in zip(src.parameters(), dst.parameters()):
            d.data[:] = s.data[:]

    # ── Paper-faithful CoLR: B is server-shared basis, frozen on client ───────

    def freeze_item_B(self):
        """
        Freeze B (shared basis) in both item embeddings.
        Called BEFORE local training each round — clients only train A_u.
        """
        self.mlp_item_embeddings.B.requires_grad_(False)
        self.gmf_item_embeddings.B.requires_grad_(False)

    def unfreeze_all(self):
        """Restore all parameters to trainable (called after round ends)."""
        for p in self.parameters():
            p.requires_grad_(True)

    def get_item_A(self) -> dict:
        """
        Extract only the A matrices (item factors) for upload.
        Paper: clients upload A_u^(t), NOT B.
        Returns detached CPU tensors.
        """
        return {
            "mlp_A": self.mlp_item_embeddings.A.weight.detach().cpu(),  # (item_num, rank)
            "gmf_A": self.gmf_item_embeddings.A.weight.detach().cpu(),  # (item_num, rank)
        }

    def get_shared_weights(self) -> dict:
        """
        Extract MLP tower + output heads for upload (these ARE averaged).
        B is NOT included — server keeps its own B.
        """
        return {
            "mlp_state":       {k: v.detach().cpu()
                                for k, v in self.mlp.state_dict().items()},
            "gmf_out_weight":  self.gmf_out.weight.detach().cpu(),
            "mlp_out_weight":  self.mlp_out.weight.detach().cpu(),
            "output_logits_w": self.output_logits.weight.detach().cpu(),
            "output_logits_b": self.output_logits.bias.detach().cpu(),
        }

    def load_server_weights(self, server_model):
        """
        Load from server:
          - B (shared basis, will be frozen this round)
          - A (aggregated item factors, client's starting point)
          - MLP tower + output heads (global average)
        """
        # ── Item A: start from global aggregated A ────────────────────────────
        self.mlp_item_embeddings.A.weight.data.copy_(
            server_model.mlp_item_embeddings.A.weight.data)
        self.gmf_item_embeddings.A.weight.data.copy_(
            server_model.gmf_item_embeddings.A.weight.data)
        # ── Item B: copy server basis (will be frozen before training) ────────
        self.mlp_item_embeddings.B.data.copy_(
            server_model.mlp_item_embeddings.B.data)
        self.gmf_item_embeddings.B.data.copy_(
            server_model.gmf_item_embeddings.B.data)
        # ── MLP tower + output heads ──────────────────────────────────────────
        self.layer_setter(server_model.mlp,           self.mlp)
        self.layer_setter(server_model.gmf_out,       self.gmf_out)
        self.layer_setter(server_model.mlp_out,       self.mlp_out)
        self.layer_setter(server_model.output_logits, self.output_logits)