import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralCollaborativeFiltering(nn.Module):
    """
    Standard LoRA baseline for federated NCF.

    Item embeddings are decomposed as:
        E_item = E0  +  lora_B @ lora_A
    where E0 is frozen, lora_A ∈ R^{rank × emb_dim} and
    lora_B ∈ R^{item_num × rank} are the trainable low-rank adapters.

    User embeddings are trained fully (private, never uploaded).
    Only lora_A, lora_B (+ shared MLP/output layers) are communicated.
    """

    def __init__(self, user_num: int, item_num: int,
                 predictive_factor: int = 32, lora_rank: int = 8):
        super().__init__()
        emb_dim = 2 * predictive_factor

        self.mlp_user_embeddings = nn.Embedding(user_num, emb_dim)
        self.gmf_user_embeddings = nn.Embedding(user_num, emb_dim)

        # E0: trainable during warm-up, frozen during PEFT
        self.mlp_item_E0 = nn.Parameter(torch.zeros(item_num, emb_dim), requires_grad=False)
        self.gmf_item_E0 = nn.Parameter(torch.zeros(item_num, emb_dim), requires_grad=False)

        self.mlp_lora_A = nn.Parameter(torch.empty(lora_rank, emb_dim))
        self.mlp_lora_B = nn.Parameter(torch.zeros(item_num, lora_rank))
        self.gmf_lora_A = nn.Parameter(torch.empty(lora_rank, emb_dim))
        self.gmf_lora_B = nn.Parameter(torch.zeros(item_num, lora_rank))

        # ── MLP + output (shared, communicated) ───────────────────────────────
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

    # ── Init ──────────────────────────────────────────────────────────────────

    def _initialize_weights(self):
        nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        # LoRA standard init: A ~ N(0, 0.01), B = 0  →  initial delta = 0
        nn.init.normal_(self.mlp_lora_A, std=0.01)
        nn.init.normal_(self.gmf_lora_A, std=0.01)
        # lora_B already zeros from __init__
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

    # ── Effective item embeddings (E0 + B·A) ──────────────────────────────────

    def _mlp_item_emb(self, item_id: torch.Tensor) -> torch.Tensor:
        e0    = self.mlp_item_E0[item_id]                          # (n, d)
        delta = self.mlp_lora_B[item_id] @ self.mlp_lora_A        # (n, d)
        return e0 + delta

    def _gmf_item_emb(self, item_id: torch.Tensor) -> torch.Tensor:
        e0    = self.gmf_item_E0[item_id]
        delta = self.gmf_lora_B[item_id] @ self.gmf_lora_A
        return e0 + delta

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_out = self._gmf_forward(user_id, item_id)
        mlp_out = self._mlp_forward(user_id, item_id)
        return self.output_logits(torch.cat([gmf_out, mlp_out], dim=1)).view(-1)

    def _gmf_forward(self, user_id, item_id):
        return torch.mul(self.gmf_user_embeddings(user_id),
                         self._gmf_item_emb(item_id))

    def _mlp_forward(self, user_id, item_id):
        concat = torch.cat([self.mlp_user_embeddings(user_id),
                             self._mlp_item_emb(item_id)], dim=1)
        return self.mlp(concat)

    # ── Phase control ──────────────────────────────────────────────────────────

    def set_warmup_mode(self):
        """Warm-up: train E0 + user embs + MLP. Freeze LoRA."""
        _warmup_roots = {"mlp_item_E0", "gmf_item_E0",
                         "mlp_user_embeddings", "gmf_user_embeddings",
                         "mlp", "gmf_out", "mlp_out", "output_logits"}
        for name, p in self.named_parameters():
            p.requires_grad = name.split(".")[0] in _warmup_roots

    def set_peft_mode(self):
        """PEFT: train LoRA + user embs. Freeze E0 + MLP."""
        _peft_roots = {"mlp_lora_A", "mlp_lora_B", "gmf_lora_A", "gmf_lora_B",
                       "mlp_user_embeddings", "gmf_user_embeddings"}
        for name, p in self.named_parameters():
            p.requires_grad = name.split(".")[0] in _peft_roots

    def reset_lora(self):
        """Re-initialise LoRA to zero-delta state (called at warm-up → PEFT transition)."""
        nn.init.normal_(self.mlp_lora_A, std=0.01)
        nn.init.zeros_(self.mlp_lora_B)
        nn.init.normal_(self.gmf_lora_A, std=0.01)
        nn.init.zeros_(self.gmf_lora_B)

    # ── Upload payload helpers ─────────────────────────────────────────────────

    def get_warmup_update(self) -> dict:
        """Warm-up upload: E0 + MLP + output heads (no user embs, no LoRA)."""
        skip = {"mlp_user_embeddings", "gmf_user_embeddings",
                "mlp_lora_A", "mlp_lora_B", "gmf_lora_A", "gmf_lora_B"}
        return {name: p.detach().cpu().clone()
                for name, p in self.named_parameters()
                if name.split(".")[0] not in skip}

    def get_lora_update(self) -> dict:
        """PEFT upload: LoRA adapters + shared MLP/output (no user embs, no E0)."""
        skip = {"mlp_user_embeddings", "gmf_user_embeddings",
                "mlp_item_E0", "gmf_item_E0"}
        return {name: p.detach().cpu().clone()
                for name, p in self.named_parameters()
                if name.split(".")[0] not in skip}

    def load_server_weights(self, payload: dict):
        """Load any payload (warmup or PEFT) from server — skips missing keys."""
        own = self.state_dict()
        for k, v in payload.items():
            if k in own:
                own[k].copy_(v)
        self.load_state_dict(own, strict=False)


if __name__ == "__main__":
    ncf = NeuralCollaborativeFiltering(100, 200, predictive_factor=64, lora_rank=8)
    print(ncf)
    x = torch.randint(0, 100, (32, 2))
    x[:, 1] = torch.randint(0, 200, (32,))
    print("output shape:", ncf(x).shape)