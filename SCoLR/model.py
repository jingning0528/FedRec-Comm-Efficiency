import torch
from .low_rank import LowRankEmbedding, LowRankLinear


class NeuralCollaborativeFiltering(torch.nn.Module):
    """
    SCoLR-NCF: same as CoLR but supports extracting a subsampled item subset
    for partial upload (SCoLR).
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

    def load_server_weights(self, server_model):
        """Copy aggregated low-rank item factors from server to this client."""
        self.layer_setter(server_model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(server_model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(server_model.mlp,                 self.mlp)
        self.layer_setter(server_model.gmf_out,             self.gmf_out)
        self.layer_setter(server_model.mlp_out,             self.mlp_out)
        self.layer_setter(server_model.output_logits,       self.output_logits)

    # ── SCoLR: extract partial update (subsampled item rows only) ────────────

    def get_scolr_partial_update(self, item_indices: torch.Tensor) -> dict:
        """
        Return only the subsampled rows of A (item factors) plus full B and
        shared weights. This is the SCoLR partial upload payload.

        item_indices : 1-D LongTensor of sampled item ids, shape (s,)
        """
        return {
            # subsampled A rows — shape (s, rank)
            "item_indices":     item_indices.cpu(),
            "mlp_A_rows":       self.mlp_item_embeddings.A.weight[item_indices].detach().cpu(),
            "gmf_A_rows":       self.gmf_item_embeddings.A.weight[item_indices].detach().cpu(),
            # full shared basis B — shape (2*pf, rank)
            "mlp_B":            self.mlp_item_embeddings.B.detach().cpu(),
            "gmf_B":            self.gmf_item_embeddings.B.detach().cpu(),
            # shared MLP tower low-rank factors
            "mlp_state":        {k: v.detach().cpu()
                                 for k, v in self.mlp.state_dict().items()},
            # output heads
            "gmf_out_weight":   self.gmf_out.weight.detach().cpu(),
            "mlp_out_weight":   self.mlp_out.weight.detach().cpu(),
            "output_logits_w":  self.output_logits.weight.detach().cpu(),
            "output_logits_b":  self.output_logits.bias.detach().cpu(),
        }