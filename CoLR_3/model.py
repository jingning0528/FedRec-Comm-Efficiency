import torch
import numpy as np
from .low_rank import CoLREmbedding, LowRankLinear


class NeuralCollaborativeFiltering(torch.nn.Module):
    """
    CoLR-NCF (Algorithm 1 faithful):
      - Q_base: maintained LOCALLY on client, never re-downloaded.
      - Each round: client merges Q = Q + B_prev @ A_downloaded.T  (Alg 1, line 7)
      - B:  new random basis downloaded each round, frozen on client.
      - A:  low-rank delta, reset to 0 each round, only trainable item-side param.
      - Upload: full A (all items) each round.
    """

    def __init__(self, user_num: int, item_num: int,
                 predictive_factor: int = 32, rank: int = 16):
        super().__init__()
        pf        = predictive_factor
        self.rank = rank
        self.pf   = pf

        self.mlp_user_embeddings = torch.nn.Embedding(user_num, 2 * pf)
        self.gmf_user_embeddings = torch.nn.Embedding(user_num, 2 * pf)

        self.mlp_item_embeddings = CoLREmbedding(item_num, 2 * pf, rank)
        self.gmf_item_embeddings = CoLREmbedding(item_num, 2 * pf, rank)

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

    def prepare_for_local_train(self):
        """Reset A=0 and freeze Q_base + B (Alg 1, lines 10-11)."""
        self.mlp_item_embeddings.reset_A()
        self.gmf_item_embeddings.reset_A()
        self.mlp_item_embeddings.freeze_for_local_train()
        self.gmf_item_embeddings.freeze_for_local_train()

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def freeze_item_B(self):
        self.prepare_for_local_train()

    def get_item_A(self) -> dict:
        return {
            "mlp_A": self.mlp_item_embeddings.A.weight.detach().cpu(),
            "gmf_A": self.gmf_item_embeddings.A.weight.detach().cpu(),
        }

    def get_shared_weights(self) -> dict:
        return {
            "mlp_state":       {k: v.detach().cpu()
                                for k, v in self.mlp.state_dict().items()},
            "gmf_out_weight":  self.gmf_out.weight.detach().cpu(),
            "mlp_out_weight":  self.mlp_out.weight.detach().cpu(),
            "output_logits_w": self.output_logits.weight.detach().cpu(),
            "output_logits_b": self.output_logits.bias.detach().cpu(),
        }

    def load_server_weights(self, server_model, is_first_round: bool = False):
        """
        Algorithm 1, lines 5-10:
          t > 0:  Download A(t), merge Q_u = Q_u + B_prev @ A(t).T  (lines 6-7)
          All t:  Download B(t), reset A=0                           (line 10)
          MLP tower + output heads (FedAvg result)

        Dense Q is NOT downloaded — it lives permanently on the client.
        """
        dev = self.mlp_item_embeddings.Q_base.weight.device

        # ── Alg 1, lines 6-7: merge Q locally using downloaded A + prev B ─────
        if not is_first_round:
            A_mlp = server_model.A_mlp_agg.to(dev)   # (item_num, rank)
            A_gmf = server_model.A_gmf_agg.to(dev)
            B_mlp_prev = server_model.B_mlp_prev.to(dev)  # (2*pf, rank)
            B_gmf_prev = server_model.B_gmf_prev.to(dev)

            # Q = Q + B_prev @ A.T  →  (item_num, 2*pf)
            self.mlp_item_embeddings.Q_base.weight.data.add_(
                (B_mlp_prev @ A_mlp.T).T)
            self.gmf_item_embeddings.Q_base.weight.data.add_(
                (B_gmf_prev @ A_gmf.T).T)

        # ── Alg 1, line 10: download new B(t) ────────────────────────────────
        self.mlp_item_embeddings.B.data.copy_(server_model.B_mlp)
        self.gmf_item_embeddings.B.data.copy_(server_model.B_gmf)

        # ── MLP tower + output heads ──────────────────────────────────────────
        self.layer_setter(server_model.mlp,           self.mlp)
        self.layer_setter(server_model.gmf_out,       self.gmf_out)
        self.layer_setter(server_model.mlp_out,       self.mlp_out)
        self.layer_setter(server_model.output_logits, self.output_logits)