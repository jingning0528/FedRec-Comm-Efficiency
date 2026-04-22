import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=32, lora_rank=8):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.lora_rank = lora_rank
        self.item_num = item_num
        self.predictive_factor = predictive_factor

        # User embeddings — local, never shared
        self.mlp_user_embeddings = nn.Embedding(user_num, 2*predictive_factor)
        self.gmf_user_embeddings = nn.Embedding(user_num, 2*predictive_factor)

        # Item embeddings as LoRA: emb(i) = B[i] @ A
        # B: (item_num, r)  — item-specific latent factors
        # A: (r, emb_dim)   — shared projection, same for all items
        # No separate W_base — item embedding IS B @ A (pure low-rank factorization)
        self.mlp_lora_B = nn.Embedding(item_num, lora_rank)
        self.mlp_lora_A = nn.Parameter(torch.randn(lora_rank, 2*predictive_factor) * 0.01)
        self.gmf_lora_B = nn.Embedding(item_num, lora_rank)
        self.gmf_lora_A = nn.Parameter(torch.randn(lora_rank, 2*predictive_factor) * 0.01)

        self.mlp = nn.Sequential(
            nn.Linear(4*predictive_factor, 2*predictive_factor), nn.ReLU(),
            nn.Linear(2*predictive_factor, predictive_factor),   nn.ReLU(),
            nn.Linear(predictive_factor, predictive_factor//2),  nn.ReLU()
        )
        self.gmf_out = nn.Linear(2*predictive_factor, 1)
        self.gmf_out.weight = nn.Parameter(torch.ones(1, 2*predictive_factor))
        self.mlp_out = nn.Linear(predictive_factor//2, 1)
        self.output_logits = nn.Linear(predictive_factor, 1)
        self.model_blending = 0.5

        self.initialize_weights()
        self.join_output_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_lora_B.weight, std=0.01)
        nn.init.normal_(self.gmf_lora_B.weight, std=0.01)
        # A initialized small so initial item embeddings ≈ 0
        nn.init.normal_(self.mlp_lora_A, std=0.01)
        nn.init.normal_(self.gmf_lora_A, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def forward(self, x):
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output  = self.mlp_forward(user_id, item_id)
        return self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)

    def gmf_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        # item_emb = B[item_id] @ A  →  (batch, r) @ (r, emb_dim) = (batch, emb_dim)
        item_emb = self.gmf_lora_B(item_id) @ self.gmf_lora_A
        return torch.mul(user_emb, item_emb)

    def mlp_forward(self, user_id, item_id):
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_lora_B(item_id) @ self.mlp_lora_A
        return self.mlp(torch.cat([user_emb, item_emb], dim=1))

    def join_output_weights(self):
        W = nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    def get_lora_update(self):
        """Return LoRA matrices + shared layers for upload."""
        state = self.state_dict()
        return {
            # Use state_dict keys directly — consistent with load_state_dict
            'mlp_lora_A':             self.mlp_lora_A.data.cpu(),
            'mlp_lora_B.weight':      self.mlp_lora_B.weight.data.cpu(),
            'gmf_lora_A':             self.gmf_lora_A.data.cpu(),
            'gmf_lora_B.weight':      self.gmf_lora_B.weight.data.cpu(),
            'mlp.0.weight':           state['mlp.0.weight'].cpu(),
            'mlp.0.bias':             state['mlp.0.bias'].cpu(),
            'mlp.2.weight':           state['mlp.2.weight'].cpu(),
            'mlp.2.bias':             state['mlp.2.bias'].cpu(),
            'mlp.4.weight':           state['mlp.4.weight'].cpu(),
            'mlp.4.bias':             state['mlp.4.bias'].cpu(),
            'gmf_out.weight':         state['gmf_out.weight'].cpu(),
            'gmf_out.bias':           state['gmf_out.bias'].cpu(),
            'mlp_out.weight':         state['mlp_out.weight'].cpu(),
            'mlp_out.bias':           state['mlp_out.bias'].cpu(),
            'output_logits.weight':   state['output_logits.weight'].cpu(),
            'output_logits.bias':     state['output_logits.bias'].cpu(),
        }

    def load_server_weights(self, server_weights: dict):
        """Load A, B and shared layers from server using state_dict keys."""
        device = self.mlp_lora_A.device
        self.mlp_lora_A.data.copy_(server_weights['mlp_lora_A'].to(device))
        self.mlp_lora_B.weight.data.copy_(server_weights['mlp_lora_B.weight'].to(device))
        self.gmf_lora_A.data.copy_(server_weights['gmf_lora_A'].to(device))
        self.gmf_lora_B.weight.data.copy_(server_weights['gmf_lora_B.weight'].to(device))
        sd = self.state_dict()
        shared_keys = [k for k in server_weights
                       if k not in ('mlp_lora_A', 'mlp_lora_B.weight',
                                    'gmf_lora_A', 'gmf_lora_B.weight')]
        for k in shared_keys:
            sd[k] = server_weights[k].to(device)
        self.load_state_dict(sd, strict=False)

if __name__ == '__main__':
    ncf = NeuralCollaborativeFiltering(100, 100, 64, lora_rank=8)
    print(ncf)