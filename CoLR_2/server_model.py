import torch
from .low_rank import LowRankEmbedding, LowRankLinear


class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    """
    Server-side model for paper-faithful CoLR.

    Per-round protocol:
      1. Server broadcasts  B (fixed this round),  A_global,  MLP weights.
      2. Clients freeze B, train A_u and MLP locally.
      3. Clients upload A_u + MLP weights.
      4. Server aggregates:  A_global = Σ (N_u/N) * A_u   (B UNCHANGED).
    """

    def __init__(self, item_num: int, predictive_factor: int = 32, rank: int = 16):
        super().__init__()
        pf            = predictive_factor
        self.rank     = rank
        self.item_num = item_num
        self.pf       = pf

        # A: aggregated item factors (updated every round via FedAvg)
        # B: shared basis            (FIXED within a round, updated slowly)
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

    def aggregate_A(self,
                    client_mlp_As: list,   # list of (item_num, rank) tensors
                    client_gmf_As: list,
                    client_n_items: list,  # list of N_u (interaction counts)
                    client_shared:  list,  # list of shared-weight dicts
                    ):
        """
        Paper CoLR aggregation:
          A^(t+1) = Σ_u (N_u / N) * A_u^(t)   ← only A is aggregated
          B^(t)   unchanged                     ← B stays on server

        Also averages MLP tower + output heads (these are shared, not per-user).
        """
        total_n = sum(client_n_items)
        dev     = self.mlp_item_embeddings.A.weight.device

        # ── Weighted FedAvg on A only ─────────────────────────────────────────
        mlp_A_agg = torch.zeros_like(self.mlp_item_embeddings.A.weight)
        gmf_A_agg = torch.zeros_like(self.gmf_item_embeddings.A.weight)
        for mlp_A, gmf_A, n in zip(client_mlp_As, client_gmf_As, client_n_items):
            w = n / total_n
            mlp_A_agg += w * mlp_A.to(dev)
            gmf_A_agg += w * gmf_A.to(dev)

        self.mlp_item_embeddings.A.weight.data.copy_(mlp_A_agg)
        self.gmf_item_embeddings.A.weight.data.copy_(gmf_A_agg)

        # ── B is NOT touched — it is the fixed shared basis ───────────────────

        # ── Average MLP tower + output heads (standard FedAvg) ────────────────
        num = len(client_shared)
        # MLP
        avg_mlp = {k: torch.zeros_like(v)
                   for k, v in client_shared[0]["mlp_state"].items()}
        for cs in client_shared:
            for k, v in cs["mlp_state"].items():
                avg_mlp[k] += v / num
        mlp_dev = next(self.mlp.parameters()).device
        self.mlp.load_state_dict({k: v.to(mlp_dev) for k, v in avg_mlp.items()})

        # Output heads
        avg_gmf_out_w    = sum(cs["gmf_out_weight"]  for cs in client_shared) / num
        avg_mlp_out_w    = sum(cs["mlp_out_weight"]  for cs in client_shared) / num
        avg_out_logits_w = sum(cs["output_logits_w"] for cs in client_shared) / num
        avg_out_logits_b = sum(cs["output_logits_b"] for cs in client_shared) / num

        d = self.gmf_out.weight.device
        self.gmf_out.weight.data      = avg_gmf_out_w.to(d)
        self.mlp_out.weight.data      = avg_mlp_out_w.to(d)
        self.output_logits.weight.data = avg_out_logits_w.to(d)
        self.output_logits.bias.data   = avg_out_logits_b.to(d)

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