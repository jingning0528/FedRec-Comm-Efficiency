import torch
from .low_rank import LowRankEmbedding, LowRankLinear


class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    """
    Server-side item model using the same low-rank structure as the client.
    FedAvg averages the low-rank factors (A, B) directly — CoLR aggregation.
    """

    def __init__(self, item_num: int, predictive_factor: int = 32, rank: int = 16):
        super().__init__()
        pf = predictive_factor
        self.rank = rank

        # ── Low-rank item embeddings (CoLR) ───────────────────────────────────
        self.mlp_item_embeddings = LowRankEmbedding(item_num, 2 * pf, rank)
        self.gmf_item_embeddings = LowRankEmbedding(item_num, 2 * pf, rank)

        # ── Low-rank MLP tower (CoLR) ─────────────────────────────────────────
        self.mlp = torch.nn.Sequential(
            LowRankLinear(4 * pf, 2 * pf, rank), torch.nn.ReLU(),
            LowRankLinear(2 * pf, pf,     rank), torch.nn.ReLU(),
            LowRankLinear(pf,     pf // 2, rank), torch.nn.ReLU(),
        )

        # ── Output heads ──────────────────────────────────────────────────────
        self.gmf_out      = torch.nn.Linear(2 * pf,  1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, 2 * pf))
        self.mlp_out      = torch.nn.Linear(pf // 2, 1)
        self.output_logits = torch.nn.Linear(pf,     1)
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
        """Extract item-side low-rank factors from a trained client model."""
        self.layer_setter(client_model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(client_model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(client_model.mlp,                 self.mlp)
        self.layer_setter(client_model.gmf_out,             self.gmf_out)
        self.layer_setter(client_model.mlp_out,             self.mlp_out)
        self.layer_setter(client_model.output_logits,       self.output_logits)

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