import torch

class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, item_num, predictive_factor=32, rank=8):
        super(ServerNeuralCollaborativeFiltering, self).__init__()
        embed_dim = 2 * predictive_factor
        self.rank = rank
        self.item_num = item_num

        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=embed_dim)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=embed_dim)

        # ── Shared B matrices ──
        # Orthogonal init: columns are orthonormal → unit-scale gradient signal to A
        B_mlp_init = torch.zeros(item_num, rank)
        B_gmf_init = torch.zeros(item_num, rank)
        torch.nn.init.orthogonal_(B_mlp_init[:rank] if item_num >= rank else B_mlp_init)
        torch.nn.init.orthogonal_(B_gmf_init[:rank] if item_num >= rank else B_gmf_init)
        self.register_buffer('B_mlp', B_mlp_init)
        self.register_buffer('B_gmf', B_gmf_init)

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
        torch.nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    def update_with_A(self, A_mlp_avg: torch.Tensor, A_gmf_avg: torch.Tensor):
        """
        CoLR aggregation step:
            I_global = I_global + B @ A_avg
        Absorbs the averaged low-rank update into the base embeddings.
        """
        with torch.no_grad():
            self.mlp_item_embeddings.weight.add_(self.B_mlp @ A_mlp_avg)
            self.gmf_item_embeddings.weight.add_(self.B_gmf @ A_gmf_avg)

    def set_weights(self, model):
        """Copy item-side weights from a full client model (used in standard FedAvg path)."""
        def copy_(src, dst):
            for s, d in zip(src.parameters(), dst.parameters()):
                d.data.copy_(s.data)
        copy_(model.mlp_item_embeddings, self.mlp_item_embeddings)
        copy_(model.gmf_item_embeddings, self.gmf_item_embeddings)
        copy_(model.mlp,                 self.mlp)
        copy_(model.gmf_out,             self.gmf_out)
        copy_(model.mlp_out,             self.mlp_out)
        copy_(model.output_logits,       self.output_logits)

    def update_shared_weights(self, averaged_weights: dict):
        """FedAvg the MLP + output layer weights from clients."""
        sd = self.state_dict()
        sd.update(averaged_weights)
        self.load_state_dict(sd)

    def forward(self):
        return torch.tensor(0.0)

if __name__ == '__main__':
    ncf = ServerNeuralCollaborativeFiltering(100, 64, rank=8)
    print(ncf)