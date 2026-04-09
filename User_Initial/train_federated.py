import torch
from .train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from .server_model import ServerNeuralCollaborativeFiltering
import copy
from .llm_item_embeddings import generate_item_embeddings, load_ml1m_item_texts, generate_item_embeddings_for_matrix

class Utils:
    def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
        self.epoch = 0
        self.num_clients = num_clients
        self.local_path = local_path
        self.server_path = server_path

    def load_pytorch_client_model(self, path):
        return torch.jit.load(path)

    def get_user_models(self, loader):
        models = []
        for client_id in range(self.num_clients):
            models.append({'model':loader(self.local_path+"dp"+str(client_id)+".pt")})
        return models

    def get_previous_federated_model(self):
        self.epoch += 1
        return torch.jit.load(self.server_path+"server"+str(self.epoch-1)+".pt")

    def save_federated_model(self, model):
        torch.jit.save(model, self.server_path+"server"+str(self.epoch)+".pt")

def federate(utils):
    client_models = utils.get_user_models(utils.load_pytorch_client_model)
    server_model = utils.get_previous_federated_model()
    if len(client_models) == 0:
        utils.save_federated_model(server_model)
        return
    n = len(client_models)
    server_new_dict = copy.deepcopy(client_models[0]['model'].state_dict())
    for i in range(1, len(client_models)):
        client_dict = client_models[i]['model'].state_dict()
        for k in client_dict.keys():
            server_new_dict[k] += client_dict[k] 
    for k in server_new_dict.keys():
        server_new_dict[k] = server_new_dict[k] / n
    server_model.load_state_dict(server_new_dict)
    utils.save_federated_model(server_model)

class FederatedNCF:
    def __init__(self, ui_matrix, num_clients=50, user_per_client_range=[1, 5], mode="ncf", aggregation_epochs=50, local_epochs=10, batch_size=128, latent_dim=32, seed=0, use_llm_init=False, item_ids=None):
        random.seed(seed)
        self.ui_matrix = ui_matrix
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = num_clients
        self.latent_dim = latent_dim
        self.user_per_client_range = user_per_client_range
        self.mode = mode
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.pretrained_item_embeddings = None
        if use_llm_init:
            item_texts_dict = load_ml1m_item_texts()
            if item_ids is not None:
                self.pretrained_item_embeddings = generate_item_embeddings_for_matrix(
                    item_texts_dict, item_ids, target_dim=2*latent_dim
                )
            else:
                # Fallback: assume items are 1..N in order
                item_ids = list(range(1, ui_matrix.shape[1] + 1))
                self.pretrained_item_embeddings = generate_item_embeddings_for_matrix(
                    item_texts_dict, item_ids, target_dim=2*latent_dim
                )
        self.clients = self.generate_clients()
        # Track which user rows each client owns (for user init from items)
        self.client_user_slices = self._compute_client_user_slices()
        self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=5e-4) for client in self.clients]
        self.utils = Utils(self.num_clients)
        # Flag to track whether user embeddings have been initialized
        self.user_embeddings_initialized = False

    def _compute_client_user_slices(self):
        """Recompute user row slices for each client (must match generate_clients logic)."""
        random.seed(0)  # Use same seed as generate_clients to reproduce slices
        # We need to replay the same random choices. 
        # However, generate_clients already consumed random state.
        # Instead, store slices during generation. We'll refactor slightly.
        # For now, re-derive from client ui_matrix shapes.
        slices = []
        start = 0
        for client in self.clients:
            num_users = client.ui_matrix.shape[0]
            slices.append((start, start + num_users))
            start += num_users
        return slices

    def generate_clients(self):
        start_index = 0
        clients = []
        for i in range(self.num_clients):
            users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
            clients.append(NCFTrainer(self.ui_matrix[start_index:start_index+users], epochs=self.local_epochs, batch_size=self.batch_size))
            start_index += users
        return clients

    def initialize_user_embeddings_from_items(self):
        """
        For each client, initialize user embeddings as the average of
        the item embeddings for items that user has interacted with.
        Uses the current item embeddings from the client's model.
        """
        for client in self.clients:
            client.ncf.initialize_user_embeddings_from_items(client.ui_matrix)

    def single_round(self, epoch=0, first_time=False):
        single_round_results = {key:[] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
        bar = tqdm(enumerate(self.clients), total=self.num_clients)
        for client_id, client in bar:
            results = client.train(self.ncf_optimizers[client_id])
            for k,i in results.items():
                single_round_results[k].append(i)
            printing_single_round = {"epoch": epoch}
            printing_single_round.update({k:round(sum(i)/len(i), 4) for k,i in single_round_results.items()})
            model = torch.jit.script(client.ncf.to(torch.device("cpu")))
            torch.jit.save(model, "./models/local/dp"+str(client_id)+".pt")
            bar.set_description(str(printing_single_round))
        bar.close()

    def extract_item_models(self):
        for client_id in range(self.num_clients):
            model = torch.jit.load("./models/local/dp"+str(client_id)+".pt")
            item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
            item_model.set_weights(model)
            item_model = torch.jit.script(item_model.to(torch.device("cpu")))
            torch.jit.save(item_model, "./models/local_items/dp"+str(client_id)+".pt")

    def train(self):
        first_time = True
        server_model = ServerNeuralCollaborativeFiltering(
            item_num=self.ui_matrix.shape[1], 
            predictive_factor=self.latent_dim,
            pretrained_item_embeddings=self.pretrained_item_embeddings
        )
        server_model = torch.jit.script(server_model.to(torch.device("cpu")))
        torch.jit.save(server_model, "./models/central/server"+str(0)+".pt")
        for epoch in range(self.aggregation_epochs):
            # Load server model and distribute item embeddings to clients
            server_model = torch.jit.load("./models/central/server"+str(epoch)+".pt", map_location=self.device)
            _ = [client.ncf.to(self.device) for client in self.clients]
            _ = [client.ncf.load_server_weights(server_model) for client in self.clients]
            
            # Only initialize user embeddings from item averages on the FIRST round
            if not self.user_embeddings_initialized:
                self.initialize_user_embeddings_from_items()
                self.user_embeddings_initialized = True
            
            self.single_round(epoch=epoch, first_time=first_time)
            first_time = False
            self.extract_item_models()
            federate(self.utils)

if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    fncf = FederatedNCF(dataloader.ratings, num_clients=100, user_per_client_range=[1, 1], 
        mode="ncf", aggregation_epochs=50, local_epochs=5, batch_size=128, use_llm_init=True)
    fncf.train()
