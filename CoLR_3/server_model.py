import math
import torch
from .low_rank import LowRankLinear


class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    """
    Server-side model for CoLR (Algorithm 1).

    Per-round protocol:
      1.  Sample B(t) ~ D_B  (new random basis each round)
      2.  Broadcast A(t+1) [aggregated low-rank], B(t), MLP weights to clients.
          NOTE: dense Q is NOT broadcast — clients maintain Q locally.
      3.  Clients: merge Q_u = Q_u + B(t-1) @ A(t), copy new B, reset A=0.
      4.  Clients upload: A_u (full, all items).
      5.  Server aggregates: A(t+1) = Σ (N_u/N) * A_u
    """

    def __init__(self, item_num: int, predictive_factor: int = 32, rank: int = 16):
        super().__init__()
        pf            = predictive_factor
        self.rank     = rank
        self.item_num = item_num
        self.pf       = pf

        # ── Aggregated A — broadcast to clients each round (NOT dense Q) ──────
        # After round t: A(t+1) = Σ (N_u/N) * A_u
        # Clients use this to merge their local Q: Q = Q + B_prev @ A.T
        self.register_buffer('A_mlp_agg', torch.zeros(item_num, rank))
        self.register_buffer('A_gmf_agg', torch.zeros(item_num, rank))

        # ── Previous-round B — needed by clients for merge (Alg 1, line 7) ───
        # Clients keep B(t-1) locally, but server also tracks it for reference
        self.register_buffer('B_mlp_prev', torch.empty(2 * pf, rank))
        self.register_buffer('B_gmf_prev', torch.empty(2 * pf, rank))

        # ── Current-round B matrices — re-sampled each round (Alg 1, line 3) ─
        self.register_buffer('B_mlp', torch.empty(2 * pf, rank))
        self.register_buffer('B_gmf', torch.empty(2 * pf, rank))
        self._sample_B_inplace(self.B_mlp)
        self._sample_B_inplace(self.B_gmf)
        self.B_mlp_prev.copy_(self.B_mlp)
        self.B_gmf_prev.copy_(self.B_gmf)

        # ── Shared MLP tower + output heads ───────────────────────────────────
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

        torch.nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)
        self.join_output_weights()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sample_B_inplace(B: torch.Tensor):
        """Sample B ~ N(0, 1/sqrt(rank)) — paper D_B distribution."""
        rank = B.shape[1]
        torch.nn.init.normal_(B, std=1.0 / math.sqrt(rank))

    def sample_new_B(self):
        """
        Algorithm 1, line 3: sample B(t) ~ D_B.
        Save current B as B_prev (clients need B(t-1) for merge).
        """
        self.B_mlp_prev.copy_(self.B_mlp)
        self.B_gmf_prev.copy_(self.B_gmf)
        self._sample_B_inplace(self.B_mlp)
        self._sample_B_inplace(self.B_gmf)

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    # ── Aggregation + merge (Algorithm 1, lines 18 + 7) ──────────────────────

    def aggregate(self,
                  client_mlp_As: list,
                  client_gmf_As: list,
                  client_n_items: list,
                  client_shared:  list):
        """
        Algorithm 1, line 18: aggregate A only.
          A(t+1) = Σ_u (N_u/N) * A_u(t)

        Dense Q merge now happens on CLIENT side (Alg 1, line 7).
        Also averages MLP tower + output heads (standard FedAvg).
        """
        total_n = sum(client_n_items)
        dev     = self.A_mlp_agg.device

        # ── Aggregate A: weighted FedAvg (Alg 1, line 18) ────────────────────
        A_mlp_new = torch.zeros(self.item_num, self.rank, device=dev)
        A_gmf_new = torch.zeros(self.item_num, self.rank, device=dev)

        for mlp_A, gmf_A, n in zip(client_mlp_As, client_gmf_As, client_n_items):
            w = n / total_n
            A_mlp_new += w * mlp_A.to(dev)
            A_gmf_new += w * gmf_A.to(dev)

        # Store aggregated A for broadcast to clients next round
        self.A_mlp_agg.copy_(A_mlp_new)
        self.A_gmf_agg.copy_(A_gmf_new)

        # ── FedAvg on MLP tower + output heads ────────────────────────────────
        num     = len(client_shared)
        avg_mlp = {k: torch.zeros_like(v)
                   for k, v in client_shared[0]["mlp_state"].items()}
        for cs in client_shared:
            for k, v in cs["mlp_state"].items():
                avg_mlp[k] += v / num
        mlp_dev = next(self.mlp.parameters()).device
        self.mlp.load_state_dict({k: v.to(mlp_dev) for k, v in avg_mlp.items()})

        avg_gmf_out_w    = sum(cs["gmf_out_weight"]  for cs in client_shared) / num
        avg_mlp_out_w    = sum(cs["mlp_out_weight"]  for cs in client_shared) / num
        avg_out_logits_w = sum(cs["output_logits_w"] for cs in client_shared) / num
        avg_out_logits_b = sum(cs["output_logits_b"] for cs in client_shared) / num

        d = self.gmf_out.weight.device
        self.gmf_out.weight.data       = avg_gmf_out_w.to(d)
        self.mlp_out.weight.data       = avg_mlp_out_w.to(d)
        self.output_logits.weight.data = avg_out_logits_w.to(d)
        self.output_logits.bias.data   = avg_out_logits_b.to(d)

    def forward(self):
        return torch.tensor(0.0)


if __name__ == '__main__':
    server = ServerNeuralCollaborativeFiltering(3706, predictive_factor=64, rank=16)
    params = sum(p.numel() for p in server.parameters())
    print(server)
    print(f"Server param count: {params:,}")