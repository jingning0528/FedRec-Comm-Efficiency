import torch
from .train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from .server_model import ServerNeuralCollaborativeFiltering
import copy
import time
import numpy as np
import os
import sys
from datetime import datetime

# Bandwidth settings (in Mbps)
BANDWIDTH_PROFILES = {
    "slow":   {"upload": 1,   "download": 2},    # 30% of users
    "medium": {"upload": 10,  "download": 20},   # 40% of users
    "fast":   {"upload": 50,  "download": 100},  # 30% of users
}

def assign_bandwidth(num_clients, seed=0):
    """Assign bandwidth profile to each client: 30% slow, 40% medium, 30% fast."""
    rng = random.Random(seed)
    profiles = []
    for i in range(num_clients):
        r = rng.random()
        if r < 0.3:
            profiles.append("slow")
        elif r < 0.7:
            profiles.append("medium")
        else:
            profiles.append("fast")
    return profiles

def get_model_size_bits(model):
    """Calculate model size in bits (float32 = 32 bits per param)."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 32  # bits

def calc_comm_time(size_bits, bandwidth_mbps):
    """Calculate communication time in seconds."""
    bandwidth_bps = bandwidth_mbps * 1e6
    return size_bits / bandwidth_bps

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
        return 0.0
    
    agg_start = time.time()
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
    agg_time = time.time() - agg_start
    return agg_time

class TeeLogger:
    """Writes output to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal

class FederatedNCF:
    def __init__(self, ui_matrix, num_clients=50, user_per_client_range=[1, 5], mode="ncf",
                 aggregation_epochs=50, local_epochs=10, batch_size=128, latent_dim=32, seed=0):
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
        self.clients = self.generate_clients()
        self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=5e-4) for client in self.clients]
        self.utils = Utils(self.num_clients)
        self.bandwidth_profiles = assign_bandwidth(num_clients, seed=seed)
        self.timing_log = []  # stores per-epoch timing summary
        self.metrics_log = []

        # Root of project = parent of this file's folder
        # __file__ = .../FedRec_Comm_Efficiency/FedNCF/train_federated.py
        # root     = .../FedRec_Comm_Efficiency/
        root         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_name  = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        timestamp    = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir      = os.path.join(root, "result_figure")
        self.log_path = os.path.join(log_dir, f"{folder_name}-{timestamp}.txt")

        self.logger  = TeeLogger(self.log_path)
        sys.stdout   = self.logger
        print(f"Log file : {self.log_path}")
        print(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

    def generate_clients(self):
        start_index = 0
        clients = []
        for i in range(self.num_clients):
            users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
            clients.append(NCFTrainer(self.ui_matrix[start_index:start_index+users], epochs=self.local_epochs, batch_size=self.batch_size))
            start_index += users
        return clients

    def single_round(self, epoch=0, first_time=False, server_model_size_bits=0, item_model_size_bits=0):
        single_round_results = {key:[] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
        client_timings = []

        bar = tqdm(enumerate(self.clients), total=self.num_clients)
        for client_id, client in bar:
            profile_name = self.bandwidth_profiles[client_id]
            bw = BANDWIDTH_PROFILES[profile_name]

            # --- Download time: server model sent to client ---
            download_time = calc_comm_time(server_model_size_bits, bw["download"])

            # --- Local training time ---
            train_start = time.time()
            results = client.train(self.ncf_optimizers[client_id])
            train_time = time.time() - train_start

            # --- Upload time: item model sent to server ---
            upload_time = calc_comm_time(item_model_size_bits, bw["upload"])

            client_timings.append({
                "client_id": client_id,
                "bandwidth": profile_name,
                "download_time_s": round(download_time, 4),
                "train_time_s": round(train_time, 4),
                "upload_time_s": round(upload_time, 4),
                "total_time_s": round(download_time + train_time + upload_time, 4),
            })

            for k,i in results.items():
                single_round_results[k].append(i)
            printing_single_round = {"epoch": epoch}
            printing_single_round.update({k:round(sum(i)/len(i), 4) for k,i in single_round_results.items()})
            model = torch.jit.script(client.ncf.to(torch.device("cpu")))
            torch.jit.save(model, "./models/local/dp"+str(client_id)+".pt")
            bar.set_description(str(printing_single_round))
        bar.close()

        # ── Save metrics for this round ──
        self.metrics_log.append({
            "epoch":        epoch,
            "loss":         round(float(np.mean(single_round_results["loss"])),         6),
            "hit_ratio@10": round(float(np.mean(single_round_results["hit_ratio@10"])), 6),
            "ndcg@10":      round(float(np.mean(single_round_results["ndcg@10"])),      6),
        })

        return client_timings

    def extract_item_models(self):
        for client_id in range(self.num_clients):
            model = torch.jit.load("./models/local/dp"+str(client_id)+".pt")
            item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
            item_model.set_weights(model)
            item_model = torch.jit.script(item_model.to(torch.device("cpu")))
            torch.jit.save(item_model, "./models/local_items/dp"+str(client_id)+".pt")

    def print_timing_summary(self, epoch, client_timings, agg_time):
        by_profile = {"slow": [], "medium": [], "fast": []}
        for ct in client_timings:
            by_profile[ct["bandwidth"]].append(ct)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Timing Summary")
        print(f"{'='*60}")
        for profile, timings in by_profile.items():
            if not timings:
                continue
            avg_dl  = np.mean([t["download_time_s"] for t in timings])
            avg_tr  = np.mean([t["train_time_s"] for t in timings])
            avg_ul  = np.mean([t["upload_time_s"] for t in timings])
            avg_tot = np.mean([t["total_time_s"] for t in timings])
            print(f"  [{profile.upper():6s}] n={len(timings):2d} | "
                  f"download={avg_dl:.4f}s | train={avg_tr:.4f}s | "
                  f"upload={avg_ul:.4f}s | total={avg_tot:.4f}s")

        all_totals = [t["total_time_s"] for t in client_timings]
        print(f"  [ROUND ] bottleneck (max client time) = {max(all_totals):.4f}s")
        print(f"  [AGG   ] aggregation time             = {agg_time:.4f}s")
        print(f"  [TOTAL ] round time (bottleneck+agg)  = {max(all_totals)+agg_time:.4f}s")
        print(f"{'='*60}\n")

    def train(self):
        first_time = True
        server_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
        server_model = torch.jit.script(server_model.to(torch.device("cpu")))
        torch.jit.save(server_model, "./models/central/server"+str(0)+".pt")

        # Pre-calculate model sizes
        server_model_size_bits = get_model_size_bits(server_model)
        item_model_size_bits = server_model_size_bits  # same architecture

        print(f"Server model size: {server_model_size_bits/8/1024:.2f} KB")
        print(f"Item model size:   {item_model_size_bits/8/1024:.2f} KB")
        print(f"Bandwidth distribution: "
              f"{self.bandwidth_profiles.count('slow')} slow, "
              f"{self.bandwidth_profiles.count('medium')} medium, "
              f"{self.bandwidth_profiles.count('fast')} fast")

        for epoch in range(self.aggregation_epochs):
            server_model = torch.jit.load("./models/central/server"+str(epoch)+".pt", map_location=self.device)
            _ = [client.ncf.to(self.device) for client in self.clients]
            _ = [client.ncf.load_server_weights(server_model) for client in self.clients]

            client_timings = self.single_round(
                epoch=epoch, first_time=first_time,
                server_model_size_bits=server_model_size_bits,
                item_model_size_bits=item_model_size_bits
            )
            first_time = False
            self.extract_item_models()
            agg_time = federate(self.utils)
            self.print_timing_summary(epoch, client_timings, agg_time)
            self.timing_log.append({"epoch": epoch, "client_timings": client_timings, "agg_time": agg_time})

        self._save_final_summary()
        self.logger.close()

    def _save_final_summary(self):
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"{'Epoch':>6} | {'Loss':>8} | {'HR@10':>8} | {'NDCG@10':>8}")
        print(f"{'-'*42}")
        for m in self.metrics_log:
            print(f"{m['epoch']:>6} | {m['loss']:>8.4f} | {m['hit_ratio@10']:>8.4f} | {m['ndcg@10']:>8.4f}")
        print(f"{'='*60}")
        if self.timing_log:
            all_totals  = [max(t["total_time_s"] for t in tl["client_timings"])
                           for tl in self.timing_log]
            total_round = sum(all_totals)
            total_agg   = sum(t["agg_time"] for t in self.timing_log)
            print(f"\nTotal rounds     : {len(self.timing_log)}")
            print(f"Total round time : {total_round:.2f}s ({total_round/60:.2f} min)")
            print(f"Total agg time   : {total_agg:.2f}s")
            print(f"Avg time / round : {total_round/len(self.timing_log):.4f}s")
        if self.metrics_log:
            last = self.metrics_log[-1]
            print(f"\nFinal metrics:")
            print(f"  Loss    : {last['loss']}")
            print(f"  HR@10   : {last['hit_ratio@10']}")
            print(f"  NDCG@10 : {last['ndcg@10']}")
        print(f"\nLog saved → {self.log_path}")

if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    fncf = FederatedNCF(dataloader.ratings, num_clients=60, user_per_client_range=[1, 1], mode="ncf", aggregation_epochs=50, local_epochs=5, batch_size=128)
    fncf.train()