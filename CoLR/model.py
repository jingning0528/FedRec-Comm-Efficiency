import torch

class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=32, rank=8):
        super(NeuralCollaborativeFiltering, self).__init__()
        embed_dim = 2 * predictive_factor
        self.rank = rank

        # ── User embeddings (local, never shared) ────────────────────────────
        self.mlp_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=embed_dim)
        self.gmf_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=embed_dim)

        # ── Item embedding base (received from server each round, frozen) ─────
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=embed_dim)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=embed_dim)

        # ── Shared B matrices (sent by server, fixed on client) ───────────────
        # B: (item_num, rank) — projects low-rank space → embedding space
        self.register_buffer('B_mlp', torch.zeros(item_num, rank))
        self.register_buffer('B_gmf', torch.zeros(item_num, rank))

        # ── Per-client trainable A matrices (ONLY thing uploaded to server) ───
        # A: (rank, embed_dim) — small, rank << item_num
        self.A_mlp = torch.nn.Parameter(torch.zeros(rank, embed_dim))
        self.A_gmf = torch.nn.Parameter(torch.zeros(rank, embed_dim))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * predictive_factor, 2 * predictive_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * predictive_factor, predictive_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(predictive_factor, predictive_factor // 2),
            torch.nn.ReLU()
        )
        self.gmf_out = torch.nn.Linear(embed_dim, 1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, embed_dim))
        self.mlp_out = torch.nn.Linear(predictive_factor // 2, 1)
        self.output_logits = torch.nn.Linear(predictive_factor, 1)
        self.model_blending = 0.5
        self.initialize_weights()
        self.join_output_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def forward(self, x):
        user_id, item_id = x[:, 0], x[:, 1]
        # Effective item embeddings: I_eff = I_global + B @ A  (CoLR core)
        mlp_item_eff = self.mlp_item_embeddings.weight + self.B_mlp @ self.A_mlp  # (M, D)
        gmf_item_eff = self.gmf_item_embeddings.weight + self.B_gmf @ self.A_gmf  # (M, D)

        gmf_user = self.gmf_user_embeddings(user_id)
        gmf_item = gmf_item_eff[item_id]
        gmf_product = torch.mul(gmf_user, gmf_item)

        mlp_user = self.mlp_user_embeddings(user_id)
        mlp_item = mlp_item_eff[item_id]
        mlp_output = self.mlp(torch.cat([mlp_user, mlp_item], dim=1))

        return self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    def load_server_weights(self, server_model, prev_A_mlp=None, prev_A_gmf=None):
        """
        Called at start of each round.
        Copies updated item base + shared B from server.
        Warm-starts A from previous round instead of zeroing (faster convergence).
        """
        self.mlp_item_embeddings.weight.data.copy_(server_model.mlp_item_embeddings.weight.data)
        self.gmf_item_embeddings.weight.data.copy_(server_model.gmf_item_embeddings.weight.data)
        self.B_mlp.copy_(server_model.B_mlp)
        self.B_gmf.copy_(server_model.B_gmf)
        self.mlp.load_state_dict(server_model.mlp.state_dict())
        self.gmf_out.load_state_dict(server_model.gmf_out.state_dict())
        self.mlp_out.load_state_dict(server_model.mlp_out.state_dict())
        self.output_logits.load_state_dict(server_model.output_logits.state_dict())
        # Warm-start A from previous round; zero only on first round
        if prev_A_mlp is not None:
            self.A_mlp.data.copy_(prev_A_mlp)
            self.A_gmf.data.copy_(prev_A_gmf)
        else:
            self.A_mlp.data.zero_()
            self.A_gmf.data.zero_()
        self.mlp_item_embeddings.weight.requires_grad_(False)
        self.gmf_item_embeddings.weight.requires_grad_(False)

if __name__ == '__main__':
    ncf = NeuralCollaborativeFiltering(100, 100, 64, rank=8)
    print(ncf)
    print(f"Trainable params: {sum(p.numel() for p in ncf.parameters() if p.requires_grad)}")