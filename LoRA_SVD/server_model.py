import torch

class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, item_num, predictive_factor=32):
        super(ServerNeuralCollaborativeFiltering, self).__init__()
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)
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

    def layer_setter(self, model, model_copy):
        for m, mc in zip(model.parameters(), model_copy.parameters()):
            mc.data[:] = m.data[:]

    def set_weights(self, model):
        self.layer_setter(model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(model.mlp,           self.mlp)
        self.layer_setter(model.gmf_out,       self.gmf_out)
        self.layer_setter(model.mlp_out,       self.mlp_out)
        self.layer_setter(model.output_logits, self.output_logits)

    def apply_lora_updates(self, lora_updates: list):
        """
        Merge averaged LoRA updates into base item embeddings.
        lora_updates: list of dicts with keys lora_A_mlp_item, lora_B_mlp_item, etc.
        Update rule: W_new = W_old + mean_i(B_i @ A_i)
        """
        n = len(lora_updates)
        avg_delta_mlp = sum(u['lora_B_mlp_item'] @ u['lora_A_mlp_item'] for u in lora_updates) / n
        avg_delta_gmf = sum(u['lora_B_gmf_item'] @ u['lora_A_gmf_item'] for u in lora_updates) / n
        self.mlp_item_embeddings.weight.data += avg_delta_mlp
        self.gmf_item_embeddings.weight.data += avg_delta_gmf

    def forward(self):
        return torch.tensor(0.0)

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat(
            (self.model_blending * self.gmf_out.weight,
             (1 - self.model_blending) * self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

    def get_compressed_download(self, rank: int):
        """
        SVD-compress item embeddings for download.
        Shared layers (MLP, output) sent in full — they are small.
        Returns a dict the client can reconstruct from.
        """
        def svd_compress(W, r):
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            return U[:, :r], S[:r], Vh[:r, :]

        U_mlp, S_mlp, Vh_mlp = svd_compress(self.mlp_item_embeddings.weight.data.cpu(), rank)
        U_gmf, S_gmf, Vh_gmf = svd_compress(self.gmf_item_embeddings.weight.data.cpu(), rank)

        state = self.state_dict()
        return {
            # Compressed item embeddings
            'U_mlp': U_mlp, 'S_mlp': S_mlp, 'Vh_mlp': Vh_mlp,
            'U_gmf': U_gmf, 'S_gmf': S_gmf, 'Vh_gmf': Vh_gmf,
            # Shared layers in full (small — Linear weights)
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

if __name__ == '__main__':
    ncf = ServerNeuralCollaborativeFiltering(100, 64)
    print(ncf)