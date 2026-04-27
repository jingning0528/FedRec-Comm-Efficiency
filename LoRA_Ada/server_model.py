import torch
import torch.nn as nn


class ServerNeuralCollaborativeFiltering(nn.Module):
    """
    Server-side model for Adaptive LoRA FedAvg.
    Holds max_rank adapters; distributes rank-sliced payloads to clients.
    """

    def __init__(self, item_num: int, predictive_factor: int = 32, lora_rank: int = 8):
        super().__init__()
        emb_dim = 2 * predictive_factor
        self.max_rank = lora_rank

        self.register_buffer("mlp_item_E0", torch.zeros(item_num, emb_dim))
        self.register_buffer("gmf_item_E0", torch.zeros(item_num, emb_dim))

        self.mlp_lora_A = nn.Parameter(torch.empty(lora_rank, emb_dim))
        self.mlp_lora_B = nn.Parameter(torch.zeros(item_num, lora_rank))
        self.gmf_lora_A = nn.Parameter(torch.empty(lora_rank, emb_dim))
        self.gmf_lora_B = nn.Parameter(torch.zeros(item_num, lora_rank))

        self.mlp = nn.Sequential(
            nn.Linear(4 * predictive_factor, 2 * predictive_factor), nn.ReLU(),
            nn.Linear(2 * predictive_factor, predictive_factor),      nn.ReLU(),
            nn.Linear(predictive_factor, predictive_factor // 2),     nn.ReLU(),
        )
        self.gmf_out       = nn.Linear(emb_dim, 1)
        self.mlp_out       = nn.Linear(predictive_factor // 2, 1)
        self.output_logits = nn.Linear(predictive_factor, 1)
        self.model_blending = 0.5

        self._initialize_weights()
        self.join_output_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.mlp_lora_A, std=0.01)
        nn.init.normal_(self.gmf_lora_A, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def join_output_weights(self):
        W = nn.Parameter(torch.cat([
            self.model_blending       * self.gmf_out.weight,
            (1 - self.model_blending) * self.mlp_out.weight,
        ], dim=1))
        self.output_logits.weight = W

    def get_download_payload(self) -> dict:
        """Full max-rank payload (used internally / for warm-up)."""
        payload = {}
        for name, p in self.named_parameters():
            payload[name] = p.detach().cpu()
        for name, b in self.named_buffers():
            payload[name] = b.detach().cpu()
        return payload

    def get_download_payload_for_rank(self, rank: int) -> dict:
        """
        Slice LoRA adapters to `rank` for a specific client tier.
        lora_A: (max_rank, d) → (rank, d)
        lora_B: (N, max_rank) → (N, rank)
        """
        payload = self.get_download_payload()
        payload["mlp_lora_A"] = payload["mlp_lora_A"][:rank]
        payload["mlp_lora_B"] = payload["mlp_lora_B"][:, :rank]
        payload["gmf_lora_A"] = payload["gmf_lora_A"][:rank]
        payload["gmf_lora_B"] = payload["gmf_lora_B"][:, :rank]
        return payload

    def forward(self):
        return torch.tensor(0.0)


if __name__ == "__main__":
    m = ServerNeuralCollaborativeFiltering(200, predictive_factor=64, lora_rank=8)
    print(m)