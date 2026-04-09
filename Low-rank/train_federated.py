import torch
from .train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from .server_model import ServerNeuralCollaborativeFiltering
import copy

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
    def __init__(self, ui_matrix, clients_per_round=500, eval_clients=200, mode="ncf", aggregation_epochs=100, local_epochs=3, batch_size=128, latent_dim=32, seed=0, lr=1e-3):
        random.seed(seed)
        self.ui_matrix = ui_matrix
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.total_users = ui_matrix.shape[0]
        self.clients_per_round = clients_per_round
        self.latent_dim = latent_dim
        self.mode = mode
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.utils = Utils(self.clients_per_round)

        # Fixed evaluation set — same users every round for stable metrics
        all_indices = list(range(self.total_users))
        random.shuffle(all_indices)
        self.eval_user_indices = sorted(all_indices[:eval_clients])
        self.train_user_indices = all_indices[eval_clients:]  # remaining users for training

    def sample_clients(self):
        """Randomly sample users from training set and create fresh clients each round."""
        selected_user_indices = random.sample(self.train_user_indices, self.clients_per_round)
        clients = []
        for user_idx in selected_user_indices:
            clients.append(NCFTrainer(self.ui_matrix[user_idx:user_idx+1], epochs=self.local_epochs, batch_size=self.batch_size))
        optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=self.lr) for client in clients]
        return clients, optimizers

    def single_round(self, clients, optimizers, epoch=0):
        """Train only — no evaluation here."""
        bar = tqdm(enumerate(clients), total=self.clients_per_round)
        for client_id, client in bar:
            client.train(optimizers[client_id])
            model = torch.jit.script(client.ncf.to(torch.device("cpu")))
            torch.jit.save(model, "./models/local/dp"+str(client_id)+".pt")
            bar.set_description(f"Training round {epoch}")
        bar.close()

    def evaluate(self, server_model, epoch):
        """Evaluate on fixed set of users for stable metrics."""
        eval_results = {key:[] for key in ["hit_ratio@10", "ndcg@10"]}
        for user_idx in self.eval_user_indices:
            eval_client = NCFTrainer(self.ui_matrix[user_idx:user_idx+1], epochs=0, batch_size=self.batch_size)
            eval_client.ncf.to(self.device)
            eval_client.ncf.load_server_weights(server_model)
            metrics = eval_client.evaluate()
            for k, v in metrics.items():
                if k in eval_results:
                    eval_results[k].append(v)
        avg_results = {k: round(sum(v)/len(v), 4) for k, v in eval_results.items()}
        print(f"Epoch {epoch} | Eval HR@10: {avg_results['hit_ratio@10']} | Eval NDCG@10: {avg_results['ndcg@10']}")
        return avg_results

    def extract_item_models(self):
        for client_id in range(self.clients_per_round):
            model = torch.jit.load("./models/local/dp"+str(client_id)+".pt")
            item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
            item_model.set_weights(model)
            item_model = torch.jit.script(item_model.to(torch.device("cpu")))
            torch.jit.save(item_model, "./models/local_items/dp"+str(client_id)+".pt")

    def train(self):
        server_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
        server_model = torch.jit.script(server_model.to(torch.device("cpu")))
        torch.jit.save(server_model, "./models/central/server"+str(0)+".pt")

        for epoch in range(self.aggregation_epochs):
            # Sample random training clients
            clients, optimizers = self.sample_clients()

            server_model = torch.jit.load("./models/central/server"+str(epoch)+".pt", map_location=self.device)
            _ = [client.ncf.to(self.device) for client in clients]
            _ = [client.ncf.load_server_weights(server_model) for client in clients]
            self.single_round(clients, optimizers, epoch=epoch)
            self.extract_item_models()
            federate(self.utils)

            # Evaluate on fixed users using the new server model
            new_server = torch.jit.load("./models/central/server"+str(epoch+1)+".pt", map_location=self.device)
            self.evaluate(new_server, epoch)

if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    fncf = FederatedNCF(
        dataloader.ratings,
        clients_per_round=200,   # more clients per round for stability
        eval_clients=200,
        aggregation_epochs=100,
        local_epochs=3,          # fewer local epochs to reduce client drift
        batch_size=128,
        lr=1e-3                  # higher lr since user embeddings start fresh
    )
    fncf.train()