import torch
import torch.nn as nn
from .model import MAX_LORA_RANK

class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, item_num, predictive_factor=32, lora_rank=MAX_LORA_RANK):
        super(ServerNeuralCollaborativeFiltering, self).__init__()
        self.lora_rank = lora_rank          # server always stores MAX_LORA_RANK
        self.item_num = item_num
        self.predictive_factor = predictive_factor

        # Server always stores max_rank A, B
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
        nn.init.normal_(self.mlp_lora_B.weight, std=0.01)
        nn.init.normal_(self.gmf_lora_B.weight, std=0.01)
        nn.init.normal_(self.mlp_lora_A, std=0.01)
        nn.init.normal_(self.gmf_lora_A, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def join_output_weights(self):
        W = nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    def get_download_payload(self):
        """Full max_rank A, B + shared layers. Clients truncate to their own rank."""
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def forward(self):
        return torch.tensor(0.0)

if __name__ == '__main__':
    ncf = ServerNeuralCollaborativeFiltering(100, 64)
    print(ncf)