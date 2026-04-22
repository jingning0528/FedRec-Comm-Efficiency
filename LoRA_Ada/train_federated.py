import torch
from .train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from .server_model import ServerNeuralCollaborativeFiltering
from .model import LORA_RANK_BY_BANDWIDTH, MAX_LORA_RANK
import time
import numpy as np

BANDWIDTH_PROFILES = {
    "slow":   {"upload": 5,   "download": 10},
    "medium": {"upload": 10,  "download": 20},
    "fast":   {"upload": 20,  "download": 40},
}

def assign_bandwidth(num_clients, seed=0):
    rng = random.Random(seed)
    profiles = []
    for i in range(num_clients):
        r = rng.random()
        if r < 0.4:   profiles.append("slow")
        elif r < 0.8: profiles.append("medium")
        else:         profiles.append("fast")
    return profiles

def get_payload_size_bits(lora_rank, item_num, emb_dim, shared_params):
    """Upload payload size for a client with given rank (padded A,B to max_rank)."""
    # Client sends padded max_rank matrices — server needs consistent shapes for FedAvg
    # But actual data transferred = only the non-zero part = rank r
    # For honest comm simulation, use actual rank r (not padded)
    lora_params = 2 * (lora_rank * emb_dim + item_num * lora_rank)
    return (lora_params + shared_params) * 32

def get_download_size_bits(item_num, emb_dim, shared_params):
    """Download = max_rank A, B + shared layers (same for all clients)."""
    lora_params = 2 * (MAX_LORA_RANK * emb_dim + item_num * MAX_LORA_RANK)
    return (lora_params + shared_params) * 32

def get_full_model_size_bits(item_num, emb_dim, shared_params):
    full_item_params = 2 * (item_num * emb_dim)
    return (full_item_params + shared_params) * 32

def calc_comm_time(size_bits, bandwidth_mbps):
    return size_bits / (bandwidth_mbps * 1e6)

class Utils:
    def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
        self.epoch = 0
        self.num_clients = num_clients
        self.local_path = local_path
        self.server_path = server_path

    def get_lora_updates(self):
        return [torch.load(self.local_path + f"lora_dp{i}.pt") for i in range(self.num_clients)]

def federate(utils):
    """
    FedAvg on padded A, B (max_rank shape) and shared layers.
    Slow clients' padding zeros → their contribution to high-rank dims = 0 (correct).
    """
    updates = utils.get_lora_updates()
    if len(updates) == 0:
        utils.epoch += 1
        return 0.0

    prev_state = torch.load(f"./models/central/server{utils.epoch}.pt")
    utils.epoch += 1

    agg_start = time.time()
    updates_cpu = [{k: v.cpu() for k, v in u.items() if k != 'lora_rank'}
                   for u in updates]
    n = len(updates_cpu)

    new_state = {}
    for k in updates_cpu[0].keys():
        new_state[k] = sum(u[k] for u in updates_cpu) / n

    for k in prev_state:
        if k not in new_state:
            new_state[k] = prev_state[k]

    torch.save(new_state, f"./models/central/server{utils.epoch}.pt")
    return time.time() - agg_start

class FederatedNCF:
    def __init__(self, ui_matrix, num_clients=50, user_per_client_range=[1, 5],
                 aggregation_epochs=50, local_epochs=10, batch_size=128, latent_dim=32, seed=0):
        random.seed(seed)
        self.ui_matrix = ui_matrix
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = num_clients
        self.latent_dim = latent_dim
        self.user_per_client_range = user_per_client_range
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.bandwidth_profiles = assign_bandwidth(num_clients, seed=seed)
        self.clients = self.generate_clients()
        self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=5e-4)
                                for client in self.clients]
        self.utils = Utils(self.num_clients)
        self.timing_log = []
        self.cumulative_time          = 0.0
        self.cumulative_download_time = 0.0
        self.cumulative_train_time    = 0.0
        self.cumulative_upload_time   = 0.0
        self.cumulative_agg_time      = 0.0

    def generate_clients(self):
        start_index = 0
        clients = []
        for i in range(self.num_clients):
            users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
            rank  = LORA_RANK_BY_BANDWIDTH[self.bandwidth_profiles[i]]
            clients.append(NCFTrainer(
                self.ui_matrix[start_index:start_index+users],
                epochs=self.local_epochs, batch_size=self.batch_size,
                lora_rank=rank))
            start_index += users
        return clients

    def single_round(self, epoch, download_payload, shared_params, emb_dim, item_num):
        single_round_results = {k: [] for k in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
        client_timings = []

        bar = tqdm(enumerate(self.clients), total=self.num_clients)
        for client_id, client in bar:
            profile_name = self.bandwidth_profiles[client_id]
            bw   = BANDWIDTH_PROFILES[profile_name]
            rank = LORA_RANK_BY_BANDWIDTH[profile_name]

            # Download: truncate server A,B to this client's rank before sending
            client_payload = truncate_payload(download_payload, rank)
            client.ncf.to(self.device)
            client.ncf.load_server_weights(
                {k: v.to(self.device) for k, v in client_payload.items()})

            # Download size = client's own rank (not max_rank)
            download_bits = get_payload_size_bits(rank, item_num, emb_dim, shared_params)
            download_time = calc_comm_time(download_bits, bw["download"])

            train_start = time.time()
            results = client.train(self.ncf_optimizers[client_id])
            train_time = time.time() - train_start

            # Upload: actual rank r
            upload_bits = get_payload_size_bits(rank, item_num, emb_dim, shared_params)
            upload_time = calc_comm_time(upload_bits, bw["upload"])

            client_timings.append({
                "client_id":       client_id,
                "bandwidth":       profile_name,
                "lora_rank":       rank,
                "comm_kb":         round((download_bits + upload_bits) / 8 / 1024, 2),
                "download_time_s": round(download_time, 4),
                "train_time_s":    round(train_time, 4),
                "upload_time_s":   round(upload_time, 4),
                "total_time_s":    round(download_time + train_time + upload_time, 4),
            })

            update = client.ncf.get_lora_update()
            torch.save(update, f"./models/local_items/lora_dp{client_id}.pt")

            for k, i in results.items():
                single_round_results[k].append(i)
            printing = {"epoch": epoch}
            printing.update({k: round(sum(i)/len(i), 4) for k, i in single_round_results.items()})
            bar.set_description(str(printing))
        bar.close()
        return client_timings

    def print_timing_summary(self, epoch, client_timings, agg_time,
                              download_size_bits, full_size_bits, shared_params, emb_dim, item_num):
        by_profile = {"slow": [], "medium": [], "fast": []}
        for ct in client_timings:
            by_profile[ct["bandwidth"]].append(ct)

        all_totals    = [t["total_time_s"]    for t in client_timings]
        all_downloads = [t["download_time_s"] for t in client_timings]
        all_trains    = [t["train_time_s"]    for t in client_timings]
        all_uploads   = [t["upload_time_s"]   for t in client_timings]

        round_download = max(all_downloads)
        round_train    = max(all_trains)
        round_upload   = max(all_uploads)
        round_time     = max(all_totals) + agg_time

        self.cumulative_download_time += round_download
        self.cumulative_train_time    += round_train
        self.cumulative_upload_time   += round_upload
        self.cumulative_agg_time      += agg_time
        self.cumulative_time          += round_time

        print(f"\n{'='*72}")
        print(f"Epoch {epoch}  [Adaptive LoRA | slow=r{LORA_RANK_BY_BANDWIDTH['slow']} "
              f"medium=r{LORA_RANK_BY_BANDWIDTH['medium']} "
              f"fast=r{LORA_RANK_BY_BANDWIDTH['fast']}]")
        print(f"{'='*72}")
        print(f"  [MODEL SIZE]  Full model (ref): {full_size_bits/8/1024:.2f} KB")
        for profile, rank in LORA_RANK_BY_BANDWIDTH.items():
            bits = get_payload_size_bits(rank, item_num, emb_dim, shared_params)
            print(f"  [{profile:6s}] r={rank:2d}  "
                  f"download={bits/8/1024:.2f} KB ({100*bits/full_size_bits:.1f}%)  "
                  f"upload={bits/8/1024:.2f} KB ({100*bits/full_size_bits:.1f}%)  "
                  f"[symmetric ✓]")
        print(f"{'='*72}")
        for profile, timings in by_profile.items():
            if not timings:
                continue
            avg_dl  = np.mean([t["download_time_s"] for t in timings])
            avg_tr  = np.mean([t["train_time_s"]    for t in timings])
            avg_ul  = np.mean([t["upload_time_s"]   for t in timings])
            avg_tot = np.mean([t["total_time_s"]    for t in timings])
            rank    = timings[0]["lora_rank"]
            print(f"  [{profile.upper():6s}] r={rank:2d} n={len(timings):2d} | "
                  f"download={avg_dl:.4f}s | train={avg_tr:.4f}s | "
                  f"upload={avg_ul:.4f}s | total={avg_tot:.4f}s")
        print(f"{'='*72}")
        print(f"  [THIS ROUND]  download={round_download:.4f}s | train={round_train:.4f}s | "
              f"upload={round_upload:.4f}s | agg={agg_time:.4f}s | total={round_time:.4f}s")
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
        print(f"{'='*72}\n")

    def train(self):
        item_num = self.ui_matrix.shape[1]
        emb_dim  = 2 * self.latent_dim

        server_model = ServerNeuralCollaborativeFiltering(
            item_num=item_num, predictive_factor=self.latent_dim, lora_rank=MAX_LORA_RANK)
        torch.save(server_model.state_dict(), "./models/central/server0.pt")

        shared_params      = sum(p.numel() for name, p in server_model.named_parameters()
                                 if 'lora' not in name)
        download_size_bits = get_download_size_bits(item_num, emb_dim, shared_params)
        full_size_bits     = get_full_model_size_bits(item_num, emb_dim, shared_params)

        print(f"\n{'='*72}")
        print(f"Adaptive LoRA Overview  [max_rank={MAX_LORA_RANK}, no SVD]")
        print(f"{'='*72}")
        print(f"  Full model (reference):  {full_size_bits/8/1024:.2f} KB")
        print(f"  Download (all clients):  {download_size_bits/8/1024:.2f} KB "
              f"({100*download_size_bits/full_size_bits:.1f}%)")
        for profile, rank in LORA_RANK_BY_BANDWIDTH.items():
            bits = get_payload_size_bits(rank, item_num, emb_dim, shared_params)
            print(f"  Upload  [{profile:6s}] r={rank:2d}: {bits/8/1024:.2f} KB "
                  f"({100*bits/full_size_bits:.1f}%)")
        print(f"  Bandwidth: {self.bandwidth_profiles.count('slow')} slow, "
              f"{self.bandwidth_profiles.count('medium')} medium, "
              f"{self.bandwidth_profiles.count('fast')} fast")
        print(f"{'='*72}\n")

        for epoch in range(self.aggregation_epochs):
            server_model = ServerNeuralCollaborativeFiltering(
                item_num=item_num, predictive_factor=self.latent_dim, lora_rank=MAX_LORA_RANK)
            server_model.load_state_dict(torch.load(f"./models/central/server{epoch}.pt"))
            server_model.eval()

            # Pass full max_rank payload — single_round truncates per client
            download_payload = server_model.get_download_payload()

            client_timings = self.single_round(
                epoch, download_payload, shared_params, emb_dim, item_num)
            agg_time = federate(self.utils)
            self.print_timing_summary(epoch, client_timings, agg_time,
                                      download_size_bits, full_size_bits,
                                      shared_params, emb_dim, item_num)
            self.timing_log.append({"epoch": epoch, "client_timings": client_timings,
                                    "agg_time": agg_time})

def truncate_payload(payload: dict, rank: int) -> dict:
    """
    Truncate server's max_rank A, B to client's rank.
    A: (max_rank, emb_dim) → (rank, emb_dim)
    B: (item_num, max_rank) → (item_num, rank)
    Shared layers passed through unchanged.
    """
    out = {}
    for k, v in payload.items():
        if k == 'mlp_lora_A' or k == 'gmf_lora_A':   # (max_rank, emb_dim)
            out[k] = v[:rank, :]
        elif k == 'mlp_lora_B.weight' or k == 'gmf_lora_B.weight':  # (item_num, max_rank)
            out[k] = v[:, :rank]
        else:
            out[k] = v
    return out

if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    fncf = FederatedNCF(dataloader.ratings, num_clients=60, user_per_client_range=[1, 1],
                        aggregation_epochs=100, local_epochs=5, batch_size=128)
    fncf.train()