import torch
from .low_rank import LowRankEmbedding, LowRankLinear


class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    """
    Server-side item model for SCoLR.
    Supports sparse FedAvg — only subsampled item rows are updated per round.
    """

    def __init__(self, item_num: int, predictive_factor: int = 32, rank: int = 16):
        super().__init__()
        pf = predictive_factor
        self.rank     = rank
        self.item_num = item_num
        self.pf       = pf

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
        torch.nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def layer_setter(self, src, dst):
        for s, d in zip(src.parameters(), dst.parameters()):
            d.data[:] = s.data[:]

    def set_weights(self, client_model):
        """Used by CoLR full-model extraction (kept for compatibility)."""
        self.layer_setter(client_model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(client_model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(client_model.mlp,                 self.mlp)
        self.layer_setter(client_model.gmf_out,             self.gmf_out)
        self.layer_setter(client_model.mlp_out,             self.mlp_out)
        self.layer_setter(client_model.output_logits,       self.output_logits)

    def apply_scolr_aggregate(self, avg_mlp_A: torch.Tensor,
                               avg_gmf_A: torch.Tensor,
                               item_indices: torch.Tensor,
                               avg_mlp_B: torch.Tensor,
                               avg_gmf_B: torch.Tensor,
                               avg_mlp_state: dict,
                               avg_gmf_out_w: torch.Tensor,
                               avg_mlp_out_w: torch.Tensor,
                               avg_output_logits_w: torch.Tensor,
                               avg_output_logits_b: torch.Tensor):
        """
        SCoLR sparse update:
        - Only rows in item_indices of A are overwritten with the averaged values.
        - Rows not covered by any client this round are LEFT UNCHANGED (carry-over).
        - B, MLP, and output heads are fully replaced (global average).
        """
        # ── Sparse A update (only sampled item rows) ──────────────────────────
        self.mlp_item_embeddings.A.weight.data[item_indices] = avg_mlp_A.to(
            self.mlp_item_embeddings.A.weight.device)
        self.gmf_item_embeddings.A.weight.data[item_indices] = avg_gmf_A.to(
            self.gmf_item_embeddings.A.weight.device)

        # ── Full B update ─────────────────────────────────────────────────────
        self.mlp_item_embeddings.B.data = avg_mlp_B.to(self.mlp_item_embeddings.B.device)
        self.gmf_item_embeddings.B.data = avg_gmf_B.to(self.gmf_item_embeddings.B.device)

        # ── MLP tower update ──────────────────────────────────────────────────
        dev = next(self.mlp.parameters()).device
        self.mlp.load_state_dict({k: v.to(dev) for k, v in avg_mlp_state.items()})

        # ── Output heads ──────────────────────────────────────────────────────
        dev = self.gmf_out.weight.device
        self.gmf_out.weight.data      = avg_gmf_out_w.to(dev)
        self.mlp_out.weight.data      = avg_mlp_out_w.to(dev)
        self.output_logits.weight.data = avg_output_logits_w.to(dev)
        self.output_logits.bias.data   = avg_output_logits_b.to(dev)

    def forward(self):
        return torch.tensor(0.0)

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W


if __name__ == '__main__':
    server = ServerNeuralCollaborativeFiltering(3706, predictive_factor=64, rank=16)
    params = sum(p.numel() for p in server.parameters())
    print(server)
    print(f"Server param count (= bits communicated per round): {params:,}")