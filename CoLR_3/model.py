import torch
import numpy as np
from .low_rank import CoLREmbedding, LowRankLinear


class NeuralCollaborativeFiltering(torch.nn.Module):
    """
    CoLR-NCF (Algorithm 1 faithful):
      - Q_base: dense merged embedding, received from server, frozen on client.
      - B:      random basis re-sampled each round, frozen on client.
      - A:      low-rank delta, initialized to 0 each round — ONLY trained param (item-side).
      - Upload: full A (all items) each round.
      - User embeddings: private, never communicated.
    """

    def __init__(self, user_num: int, item_num: int,
                 predictive_factor: int = 32, rank: int = 16):
        super().__init__()
        pf        = predictive_factor
        self.rank = rank
        self.pf   = pf

        # ── User embeddings (private, never communicated) ─────────────────────
        self.mlp_user_embeddings = torch.nn.Embedding(user_num, 2 * pf)
        self.gmf_user_embeddings = torch.nn.Embedding(user_num, 2 * pf)

        # ── Item embeddings: CoLR (Q_base + A @ B.T) ─────────────────────────
        self.mlp_item_embeddings = CoLREmbedding(item_num, 2 * pf, rank)
        self.gmf_item_embeddings = CoLREmbedding(item_num, 2 * pf, rank)

        # ── MLP tower (shared, averaged via FedAvg) ───────────────────────────
        self.mlp = torch.nn.Sequential(
            LowRankLinear(4 * pf, 2 * pf, rank), torch.nn.ReLU(),
            LowRankLinear(2 * pf, pf,     rank), torch.nn.ReLU(),
            LowRankLinear(pf,     pf // 2, rank), torch.nn.ReLU(),
        )

        self.gmf_out        = torch.nn.Linear(2 * pf,  1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, 2 * pf))
        self.mlp_out        = torch.nn.Linear(pf // 2, 1)
        self.output_logits  = torch.nn.Linear(pf,      1)
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
        return self.output_logits(
            torch.cat([self.gmf_forward(user_id, item_id),
                       self.mlp_forward(user_id, item_id)], dim=1)
        ).view(-1)

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

    # ── Algorithm 1 per-round hooks ───────────────────────────────────────────

    def prepare_for_local_train(self):
        """
        Called after loading server weights, before local training.
        Reset A=0 and freeze Q_base + B  (Alg 1, lines 10-11).
        Only A_u and user embeddings (p_u) are trainable.
        """
        self.mlp_item_embeddings.reset_A()
        self.gmf_item_embeddings.reset_A()
        self.mlp_item_embeddings.freeze_for_local_train()
        self.gmf_item_embeddings.freeze_for_local_train()

    def unfreeze_all(self):
        """Restore all parameters to trainable."""
        for p in self.parameters():
            p.requires_grad_(True)

    # kept for backwards compat — delegates to prepare_for_local_train
    def freeze_item_B(self):
        self.prepare_for_local_train()

    def get_item_A(self) -> dict:
        """
        Extract full A matrices for upload (Alg 1, line 16).
        Uploads ALL rows (not sparse — that is SCoLR's optimisation).
        Returns detached CPU tensors.
        """
        return {
            "mlp_A": self.mlp_item_embeddings.A.weight.detach().cpu(),  # (item_num, rank)
            "gmf_A": self.gmf_item_embeddings.A.weight.detach().cpu(),  # (item_num, rank)
        }

    def get_shared_weights(self) -> dict:
        """MLP tower + output heads for FedAvg."""
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
        Load from server before local training (Alg 1, lines 9-10):
          - Q_base ← merged dense Q from server
          - B      ← new random basis B(t)
          - MLP tower + output heads
        A is reset to 0 by prepare_for_local_train().
        """
        # ── Q_base: merged dense embedding Q(t) ──────────────────────────────
        self.mlp_item_embeddings.Q_base.weight.data.copy_(server_model.Q_mlp.data)
        self.gmf_item_embeddings.Q_base.weight.data.copy_(server_model.Q_gmf.data)

        # ── B: new random basis B(t) ──────────────────────────────────────────
        self.mlp_item_embeddings.B.data.copy_(server_model.B_mlp)
        self.gmf_item_embeddings.B.data.copy_(server_model.B_gmf)

        # ── MLP tower + output heads ──────────────────────────────────────────
        self.layer_setter(server_model.mlp,           self.mlp)
        self.layer_setter(server_model.gmf_out,       self.gmf_out)
        self.layer_setter(server_model.mlp_out,       self.mlp_out)
        self.layer_setter(server_model.output_logits, self.output_logits)