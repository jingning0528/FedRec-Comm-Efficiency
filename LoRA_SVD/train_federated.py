import torch
from .train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from .server_model import ServerNeuralCollaborativeFiltering
import copy
import time
import numpy as np

# Bandwidth settings (in Mbps)
BANDWIDTH_PROFILES = {
    "slow":   {"upload": 2,   "download": 4},
    "medium": {"upload": 10,  "download": 20},
    "fast":   {"upload": 50,  "download": 100},
}

def assign_bandwidth(num_clients, seed=0):
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
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 32

def get_lora_size_bits(lora_update: dict):
    """Calculate bits for LoRA matrices only (upload payload)."""
    total = sum(v.numel() for v in lora_update.values())
    return total * 32

def get_compressed_update_size_bits(lora_rank, item_num, emb_dim):
    """
    SVD upload: U(item_num×r) + S(r) + Vh(r×emb_dim) for mlp and gmf.
    """
    params = 2 * (item_num * lora_rank + lora_rank + lora_rank * emb_dim)
    return params * 32

def get_compressed_download_size_bits(lora_rank, item_num, emb_dim, model):
    """
    Download = SVD of item embeddings + shared layers in full.
    """
    svd_params   = 2 * (item_num * lora_rank + lora_rank + lora_rank * emb_dim)
    shared_params = sum(p.numel() for name, p in model.named_parameters()
                        if 'item_embeddings' not in name)
    return (svd_params + shared_params) * 32

def calc_comm_time(size_bits, bandwidth_mbps):
    return size_bits / (bandwidth_mbps * 1e6)

class Utils:
    def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
        self.epoch = 0
        self.num_clients = num_clients
        self.local_path = local_path
        self.server_path = server_path

    def get_lora_updates(self):
        updates = []
        for client_id in range(self.num_clients):
            updates.append(torch.load(self.local_path + f"lora_dp{client_id}.pt"))
        return updates

    def get_previous_federated_model(self):
        self.epoch += 1
        return torch.jit.load(self.server_path + f"server{self.epoch-1}.pt")

    def save_federated_model(self, model):
        torch.jit.save(model, self.server_path + f"server{self.epoch}.pt")

def federate(utils):
    """Aggregate SVD-compressed upload deltas + FedAvg shared layers, save as state_dict."""
    lora_updates = utils.get_lora_updates()

    if len(lora_updates) == 0:
        utils.epoch += 1
        return 0.0

    # Load current server state, then increment epoch
    prev_state = torch.load(f"./models/central/server{utils.epoch}.pt")
    utils.epoch += 1  # increment AFTER loading, BEFORE saving

    agg_start = time.time()
    updates_cpu = [{k: v.cpu() for k, v in u.items()} for u in lora_updates]
    n = len(updates_cpu)

    avg_delta_mlp = sum(
        u['U_mlp'] @ torch.diag(u['S_mlp']) @ u['Vh_mlp'] for u in updates_cpu
    ) / n
    avg_delta_gmf = sum(
        u['U_gmf'] @ torch.diag(u['S_gmf']) @ u['Vh_gmf'] for u in updates_cpu
    ) / n

    prev_state['mlp_item_embeddings.weight'] += avg_delta_mlp
    prev_state['gmf_item_embeddings.weight'] += avg_delta_gmf

    shared_keys = [k for k in updates_cpu[0].keys()
                   if k not in ('U_mlp','S_mlp','Vh_mlp','U_gmf','S_gmf','Vh_gmf')]
    for k in shared_keys:
        prev_state[k] = sum(u[k] for u in updates_cpu) / n

    torch.save(prev_state, f"./models/central/server{utils.epoch}.pt")
    return time.time() - agg_start

class FederatedNCF:
    def __init__(self, ui_matrix, num_clients=50, user_per_client_range=[1, 5], mode="ncf",
                 aggregation_epochs=50, local_epochs=10, batch_size=128, latent_dim=32,
                 lora_rank=8, seed=0):
        random.seed(seed)
        self.ui_matrix = ui_matrix
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = num_clients
        self.latent_dim = latent_dim
        self.lora_rank = lora_rank
        self.user_per_client_range = user_per_client_range
        self.mode = mode
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.clients = self.generate_clients()
        self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=5e-4) for client in self.clients]
        self.utils = Utils(self.num_clients)
        self.bandwidth_profiles = assign_bandwidth(num_clients, seed=seed)
        self.timing_log = []
        self.cumulative_time = 0.0
        self.cumulative_download_time = 0.0
        self.cumulative_train_time    = 0.0
        self.cumulative_upload_time   = 0.0
        self.cumulative_agg_time      = 0.0

    def generate_clients(self):
        start_index = 0
        clients = []
        for i in range(self.num_clients):
            users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
            clients.append(NCFTrainer(self.ui_matrix[start_index:start_index+users],
                                      epochs=self.local_epochs, batch_size=self.batch_size,
                                      lora_rank=self.lora_rank))
            start_index += users
        return clients

    def single_round(self, epoch=0, server_model_size_bits=0, lora_size_bits=0):
        single_round_results = {key: [] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
        client_timings = []

        bar = tqdm(enumerate(self.clients), total=self.num_clients)
        for client_id, client in bar:
            profile_name = self.bandwidth_profiles[client_id]
            bw = BANDWIDTH_PROFILES[profile_name]

            download_time = calc_comm_time(server_model_size_bits, bw["download"])

            train_start = time.time()
            results = client.train(self.ncf_optimizers[client_id])
            train_time = time.time() - train_start

            upload_time = calc_comm_time(lora_size_bits, bw["upload"])

            client_timings.append({
                "client_id":       client_id,
                "bandwidth":       profile_name,
                "download_time_s": round(download_time, 4),
                "train_time_s":    round(train_time, 4),
                "upload_time_s":   round(upload_time, 4),
                "total_time_s":    round(download_time + train_time + upload_time, 4),
            })

            # Compute and save compressed update BEFORE jit scripting
            update = client.ncf.get_compressed_update()
            torch.save(update, f"./models/local_items/lora_dp{client_id}.pt")

            # Save jit model for other purposes (optional)
            model = torch.jit.script(client.ncf.to(torch.device("cpu")))
            torch.jit.save(model, f"./models/local/dp{client_id}.pt")

            for k, i in results.items():
                single_round_results[k].append(i)
            printing = {"epoch": epoch}
            printing.update({k: round(sum(i)/len(i), 4) for k, i in single_round_results.items()})
            bar.set_description(str(printing))
        bar.close()
        return client_timings

    def extract_lora_updates(self):
        pass  # updates already saved in single_round

    def print_timing_summary(self, epoch, client_timings, agg_time,
                              server_model_size_bits, lora_size_bits, full_size_bits):
        by_profile = {"slow": [], "medium": [], "fast": []}
        for ct in client_timings:
            by_profile[ct["bandwidth"]].append(ct)

        all_totals    = [t["total_time_s"]    for t in client_timings]
        all_downloads = [t["download_time_s"] for t in client_timings]
        all_trains    = [t["train_time_s"]    for t in client_timings]
        all_uploads   = [t["upload_time_s"]   for t in client_timings]

        # Bottleneck = slowest client per phase (they run in parallel)
        round_download = max(all_downloads)
        round_train    = max(all_trains)
        round_upload   = max(all_uploads)
        round_time     = max(all_totals) + agg_time

        self.cumulative_download_time += round_download
        self.cumulative_train_time    += round_train
        self.cumulative_upload_time   += round_upload
        self.cumulative_agg_time      += agg_time
        self.cumulative_time          += round_time

        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Timing Summary  [LoRA rank={self.lora_rank}]")
        print(f"{'='*70}")

        # Model size comparison
        shared_bits = full_size_bits - 2 * (self.ui_matrix.shape[1] * 2 * self.latent_dim * 32)
        item_full_bits = full_size_bits - shared_bits
        item_svd_bits  = server_model_size_bits - shared_bits  # SVD item part of download

        print(f"  [MODEL SIZE]")
        print(f"    Full model:             {full_size_bits/8/1024:.2f} KB  (100%)")
        print(f"    ├─ item embeddings:     {item_full_bits/8/1024:.2f} KB  "
              f"({100*item_full_bits/full_size_bits:.1f}%)")
        print(f"    └─ shared layers:       {shared_bits/8/1024:.2f} KB  "
              f"({100*shared_bits/full_size_bits:.1f}%)")
        print(f"  Download (SVD item + shared layers):")
        print(f"    Total download:         {server_model_size_bits/8/1024:.2f} KB  "
              f"({100*server_model_size_bits/full_size_bits:.1f}%)")
        print(f"    ├─ SVD item (rank={self.lora_rank}):  {item_svd_bits/8/1024:.2f} KB  "
              f"({100*item_svd_bits/full_size_bits:.1f}%)")
        print(f"    └─ shared layers:       {shared_bits/8/1024:.2f} KB  "
              f"({100*shared_bits/full_size_bits:.1f}%)")
        print(f"  Upload (SVD delta + shared layers):")
        print(f"    Total upload:           {lora_size_bits/8/1024:.2f} KB  "
              f"({100*lora_size_bits/full_size_bits:.1f}%)")

        print(f"{'='*70}")

        # Per-profile timing
        for profile, timings in by_profile.items():
            if not timings:
                continue
            avg_dl  = np.mean([t["download_time_s"] for t in timings])
            avg_tr  = np.mean([t["train_time_s"]    for t in timings])
            avg_ul  = np.mean([t["upload_time_s"]   for t in timings])
            avg_tot = np.mean([t["total_time_s"]    for t in timings])
            print(f"  [{profile.upper():6s}] n={len(timings):2d} | "
                  f"download={avg_dl:.4f}s | train={avg_tr:.4f}s | "
                  f"upload={avg_ul:.4f}s | total={avg_tot:.4f}s")

        print(f"{'='*70}")
        print(f"  [THIS ROUND]  download={round_download:.4f}s | "
              f"train={round_train:.4f}s | upload={round_upload:.4f}s | "
              f"agg={agg_time:.4f}s | total={round_time:.4f}s")
        print(f"  [CUMULATIVE]  download={self.cumulative_download_time:.4f}s | "
              f"train={self.cumulative_train_time:.4f}s | "
              f"upload={self.cumulative_upload_time:.4f}s | "
              f"agg={self.cumulative_agg_time:.4f}s | "
              f"total={self.cumulative_time:.4f}s ({self.cumulative_time/60:.2f} min) "
              f"over {epoch+1} rounds")
        print(f"  [AVG/ROUND ]  download={self.cumulative_download_time/(epoch+1):.4f}s | "
              f"train={self.cumulative_train_time/(epoch+1):.4f}s | "
              f"upload={self.cumulative_upload_time/(epoch+1):.4f}s | "
              f"agg={self.cumulative_agg_time/(epoch+1):.4f}s")
        print(f"{'='*70}\n")

    def train(self):
        server_model = ServerNeuralCollaborativeFiltering(
            item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
        torch.save(server_model.state_dict(), "./models/central/server0.pt")

        emb_dim  = 2 * self.latent_dim
        item_num = self.ui_matrix.shape[1]

        full_size_bits      = get_model_size_bits(server_model)
        download_size_bits  = get_compressed_download_size_bits(self.lora_rank, item_num, emb_dim, server_model)
        upload_size_bits    = get_compressed_update_size_bits(self.lora_rank, item_num, emb_dim)

        print(f"\n{'='*70}")
        print(f"Model Size Overview")
        print(f"{'='*70}")
        print(f"  Full model:          {full_size_bits/8/1024:.2f} KB")
        print(f"  Compressed download: {download_size_bits/8/1024:.2f} KB "
              f"({100*download_size_bits/full_size_bits:.1f}% of full)")
        print(f"  LoRA upload:         {upload_size_bits/8/1024:.2f} KB "
              f"({100*upload_size_bits/full_size_bits:.1f}% of full)")
        print(f"Bandwidth: {self.bandwidth_profiles.count('slow')} slow, "
              f"{self.bandwidth_profiles.count('medium')} medium, "
              f"{self.bandwidth_profiles.count('fast')} fast")
        print(f"{'='*70}\n")

        for epoch in range(self.aggregation_epochs):
            server_model = ServerNeuralCollaborativeFiltering(
                item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
            server_model.load_state_dict(torch.load(f"./models/central/server{epoch}.pt"))
            server_model.eval()

            compressed_download = server_model.get_compressed_download(self.lora_rank)
            _ = [client.ncf.to(self.device) for client in self.clients]
            _ = [client.ncf.load_server_weights(
                    {k: v.to(self.device) for k, v in compressed_download.items()}
                 ) for client in self.clients]

            client_timings = self.single_round(
                epoch=epoch,
                server_model_size_bits=download_size_bits,
                lora_size_bits=upload_size_bits,
            )
            agg_time = federate(self.utils)
            self.print_timing_summary(epoch, client_timings, agg_time,
                                      download_size_bits, upload_size_bits, full_size_bits)
            self.timing_log.append({
                "epoch": epoch, "client_timings": client_timings, "agg_time": agg_time
            })

if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    fncf = FederatedNCF(dataloader.ratings, num_clients=60, user_per_client_range=[1, 1],
                        aggregation_epochs=100, local_epochs=5, batch_size=128)
    fncf.train()