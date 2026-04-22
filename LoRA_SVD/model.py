import torch

class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=32, lora_rank=8):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.lora_rank = lora_rank
        self.item_num = item_num
        self.predictive_factor = predictive_factor

        self.mlp_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=2*predictive_factor)
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)
        self.gmf_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=2*predictive_factor)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)

        # All parameters trainable — no freezing
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4*predictive_factor, 2*predictive_factor), torch.nn.ReLU(),
            torch.nn.Linear(2*predictive_factor, predictive_factor),   torch.nn.ReLU(),
            torch.nn.Linear(predictive_factor, predictive_factor//2),  torch.nn.ReLU()
        )
        self.gmf_out = torch.nn.Linear(2*predictive_factor, 1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, 2*predictive_factor))
        self.mlp_out = torch.nn.Linear(predictive_factor//2, 1)
        self.output_logits = torch.nn.Linear(predictive_factor, 1)
        self.model_blending = 0.5

        # Store server item embeddings to compute delta at upload time
        self.register_buffer('server_mlp_item_weight', torch.zeros(item_num, 2*predictive_factor))
        self.register_buffer('server_gmf_item_weight', torch.zeros(item_num, 2*predictive_factor))

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
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output  = self.mlp_forward(user_id, item_id)
        return self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)

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

    def layer_setter(self, model, model_copy):
        for m, mc in zip(model.parameters(), model_copy.parameters()):
            mc.data[:] = m.data[:]

    def svd_compress(self, delta: torch.Tensor, rank: int):
        """Compress delta matrix via truncated SVD. Returns U, S, Vh."""
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        return U[:, :rank], S[:rank], Vh[:rank, :]

    def get_compressed_update(self):
        """
        Compute delta = W_trained - W_server, compress via SVD to rank r.
        Upload U, S, Vh instead of full delta — communication efficient.
        Reconstruction: delta ≈ U @ diag(S) @ Vh
        """
        delta_mlp = self.mlp_item_embeddings.weight.data.cpu() - self.server_mlp_item_weight.cpu()
        delta_gmf = self.gmf_item_embeddings.weight.data.cpu() - self.server_gmf_item_weight.cpu()

        U_mlp, S_mlp, Vh_mlp = self.svd_compress(delta_mlp, self.lora_rank)
        U_gmf, S_gmf, Vh_gmf = self.svd_compress(delta_gmf, self.lora_rank)

        state = self.state_dict()
        return {
            # SVD-compressed item embedding deltas
            'U_mlp': U_mlp, 'S_mlp': S_mlp, 'Vh_mlp': Vh_mlp,
            'U_gmf': U_gmf, 'S_gmf': S_gmf, 'Vh_gmf': Vh_gmf,
            # Shared layers (full — small compared to embeddings)
            'mlp.0.weight':         state['mlp.0.weight'].cpu(),
            'mlp.0.bias':           state['mlp.0.bias'].cpu(),
            'mlp.2.weight':         state['mlp.2.weight'].cpu(),
            'mlp.2.bias':           state['mlp.2.bias'].cpu(),
            'mlp.4.weight':         state['mlp.4.weight'].cpu(),
            'mlp.4.bias':           state['mlp.4.bias'].cpu(),
            'gmf_out.weight':       state['gmf_out.weight'].cpu(),
            'gmf_out.bias':         state['gmf_out.bias'].cpu(),
            'mlp_out.weight':       state['mlp_out.weight'].cpu(),
            'mlp_out.bias':         state['mlp_out.bias'].cpu(),
            'output_logits.weight': state['output_logits.weight'].cpu(),
            'output_logits.bias':   state['output_logits.bias'].cpu(),
        }

    def load_server_weights(self, compressed: dict):
        """
        Load SVD-compressed server download.
        Reconstruct item embeddings: W ≈ U @ diag(S) @ Vh
        Snapshot reconstructed weights to compute upload delta later.
        """
        # Reconstruct item embeddings from SVD
        mlp_item_w = compressed['U_mlp'] @ torch.diag(compressed['S_mlp']) @ compressed['Vh_mlp']
        gmf_item_w = compressed['U_gmf'] @ torch.diag(compressed['S_gmf']) @ compressed['Vh_gmf']

        self.mlp_item_embeddings.weight.data.copy_(mlp_item_w.to(self.mlp_item_embeddings.weight.device))
        self.gmf_item_embeddings.weight.data.copy_(gmf_item_w.to(self.gmf_item_embeddings.weight.device))

        # Load shared layers
        shared_keys = [k for k in compressed.keys()
                       if k not in ('U_mlp','S_mlp','Vh_mlp','U_gmf','S_gmf','Vh_gmf')]
        sd = self.state_dict()
        for k in shared_keys:
            sd[k] = compressed[k].to(self.mlp_item_embeddings.weight.device)
        self.load_state_dict(sd, strict=False)

        # Snapshot reconstructed weights to compute delta after local training
        self.server_mlp_item_weight.copy_(self.mlp_item_embeddings.weight.data.cpu())
        self.server_gmf_item_weight.copy_(self.gmf_item_embeddings.weight.data.cpu())