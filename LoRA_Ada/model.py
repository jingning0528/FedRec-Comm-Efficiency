import torch
import torch.nn as nn

# Adaptive rank per bandwidth tier
LORA_RANK_BY_BANDWIDTH = {
    "slow":   4,
    "medium": 8,
    "fast":   16
}
MAX_LORA_RANK = max(LORA_RANK_BY_BANDWIDTH.values())

class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=32, lora_rank=8):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.lora_rank = lora_rank          # this client's rank (2, 8, or 32)
        self.item_num = item_num
        self.predictive_factor = predictive_factor

        self.mlp_user_embeddings = nn.Embedding(user_num, 2*predictive_factor)
        self.gmf_user_embeddings = nn.Embedding(user_num, 2*predictive_factor)

        # Item embeddings as LoRA: emb(i) = B[i] @ A
        # B: (item_num, r_client)  A: (r_client, emb_dim)
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
        nn.init.normal_(self.mlp_lora_A, std=0.01)
        nn.init.normal_(self.gmf_lora_A, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def forward(self, x):
        user_id, item_id = x[:, 0], x[:, 1]
        return self.output_logits(
            torch.cat([self.gmf_forward(user_id, item_id),
                       self.mlp_forward(user_id, item_id)], dim=1)).view(-1)

    def gmf_forward(self, user_id, item_id):
        return torch.mul(self.gmf_user_embeddings(user_id),
                         self.gmf_lora_B(item_id) @ self.gmf_lora_A)

    def mlp_forward(self, user_id, item_id):
        return self.mlp(torch.cat([self.mlp_user_embeddings(user_id),
                                   self.mlp_lora_B(item_id) @ self.mlp_lora_A], dim=1))

    def join_output_weights(self):
        W = nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    def get_lora_update(self):
        """
        Upload client's A, B padded to MAX_LORA_RANK with zeros.
        Slow clients (r=2)  upload small matrices, padded to shape of max_rank.
        Padding zeros → those dimensions contribute nothing to FedAvg.
        Shared layers always sent in full.
        """
        r   = self.lora_rank
        mr  = MAX_LORA_RANK
        emb = 2 * self.predictive_factor

        # Pad A: (r, emb) → (mr, emb)
        mlp_A_pad = torch.zeros(mr, emb)
        mlp_A_pad[:r, :] = self.mlp_lora_A.data.cpu()
        gmf_A_pad = torch.zeros(mr, emb)
        gmf_A_pad[:r, :] = self.gmf_lora_A.data.cpu()

        # Pad B: (item_num, r) → (item_num, mr)
        mlp_B_pad = torch.zeros(self.item_num, mr)
        mlp_B_pad[:, :r] = self.mlp_lora_B.weight.data.cpu()
        gmf_B_pad = torch.zeros(self.item_num, mr)
        gmf_B_pad[:, :r] = self.gmf_lora_B.weight.data.cpu()

        state = self.state_dict()
        return {
            'mlp_lora_A':           mlp_A_pad,
            'mlp_lora_B.weight':    mlp_B_pad,
            'gmf_lora_A':           gmf_A_pad,
            'gmf_lora_B.weight':    gmf_B_pad,
            'lora_rank':            torch.tensor(r),   # tell server this client's rank
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

    def load_server_weights(self, server_weights: dict):
        """
        Download: server sends max_rank A, B.
        Client truncates to its own rank r — just slicing, zero computation.
        """
        r      = self.lora_rank
        device = self.mlp_lora_A.device

        # Truncate A: (mr, emb) → (r, emb)
        self.mlp_lora_A.data.copy_(server_weights['mlp_lora_A'][:r, :].to(device))
        self.gmf_lora_A.data.copy_(server_weights['gmf_lora_A'][:r, :].to(device))

        # Truncate B: (item_num, mr) → (item_num, r)
        self.mlp_lora_B.weight.data.copy_(server_weights['mlp_lora_B.weight'][:, :r].to(device))
        self.gmf_lora_B.weight.data.copy_(server_weights['gmf_lora_B.weight'][:, :r].to(device))

        # Shared layers
        sd = self.state_dict()
        skip = {'mlp_lora_A', 'mlp_lora_B.weight', 'gmf_lora_A', 'gmf_lora_B.weight', 'lora_rank'}
        for k in server_weights:
            if k not in skip and k in sd:
                sd[k] = server_weights[k].to(device)
        self.load_state_dict(sd, strict=False)

if __name__ == '__main__':
    ncf = NeuralCollaborativeFiltering(100, 100, 64, lora_rank=8)
    print(ncf)