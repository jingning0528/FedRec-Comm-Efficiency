import torch
from .train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from .server_model import ServerNeuralCollaborativeFiltering
import time
import numpy as np

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
        if r < 0.3:   profiles.append("slow")
        elif r < 0.7: profiles.append("medium")
        else:         profiles.append("fast")
    return profiles

def get_lora_payload_size_bits(lora_rank, item_num, emb_dim, shared_params):
    """
    LoRA payload (both upload and download are identical structure):
      A: r × emb_dim
      B: item_num × r
      (×2 for mlp and gmf)
      + shared layers
    Upload == Download size — symmetric compression, no SVD.
    """
    lora_params   = 2 * (lora_rank * emb_dim + item_num * lora_rank)
    return (lora_params + shared_params) * 32

def get_full_model_size_bits(lora_rank, item_num, emb_dim, shared_params):
    """Full (uncompressed) item embedding size for comparison."""
    full_item_params = 2 * (item_num * emb_dim)  # mlp + gmf full embeddings
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
    FedAvg directly on A, B matrices and shared layers.
    No SVD — mean(A_i) and mean(B_i) are the new server A, B.
    """
    updates = utils.get_lora_updates()
    if len(updates) == 0:
        utils.epoch += 1
        return 0.0

    prev_state = torch.load(f"./models/central/server{utils.epoch}.pt")
    utils.epoch += 1

    agg_start = time.time()
    updates_cpu = [{k: v.cpu() for k, v in u.items()} for u in updates]
    n = len(updates_cpu)

    # FedAvg on every parameter (A, B, shared layers)
    new_state = {}
    for k in updates_cpu[0].keys():
        new_state[k] = sum(u[k] for u in updates_cpu) / n

    # Preserve keys not in update (e.g. buffers)
    for k in prev_state:
        if k not in new_state:
            new_state[k] = prev_state[k]

    torch.save(new_state, f"./models/central/server{utils.epoch}.pt")
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
            clients.append(NCFTrainer(
                self.ui_matrix[start_index:start_index+users],
                epochs=self.local_epochs, batch_size=self.batch_size,
                lora_rank=self.lora_rank))
            start_index += users
        return clients

    def single_round(self, epoch=0, payload_size_bits=0):
        single_round_results = {k: [] for k in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
        client_timings = []

        bar = tqdm(enumerate(self.clients), total=self.num_clients)
        for client_id, client in bar:
            profile_name = self.bandwidth_profiles[client_id]
            bw = BANDWIDTH_PROFILES[profile_name]

            # Upload == Download size (symmetric LoRA)
            download_time = calc_comm_time(payload_size_bits, bw["download"])
            upload_time   = calc_comm_time(payload_size_bits, bw["upload"])

            train_start = time.time()
            results = client.train(self.ncf_optimizers[client_id])
            train_time = time.time() - train_start

            client_timings.append({
                "client_id":       client_id,
                "bandwidth":       profile_name,
                "download_time_s": round(download_time, 4),
                "train_time_s":    round(train_time, 4),
                "upload_time_s":   round(upload_time, 4),
                "total_time_s":    round(download_time + train_time + upload_time, 4),
            })

            # Save LoRA update directly (no SVD)
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
                              payload_size_bits, full_size_bits):
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

        pct = 100 * payload_size_bits / full_size_bits
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}  [LoRA rank={self.lora_rank} | no SVD | symmetric compression]")
        print(f"{'='*70}")
        print(f"  [MODEL SIZE]")
        print(f"    Full model (reference): {full_size_bits/8/1024:.2f} KB  (100%)")
        print(f"    LoRA payload (up=down): {payload_size_bits/8/1024:.2f} KB  ({pct:.1f}%)")
        print(f"    Saved per round:        {(full_size_bits*2 - payload_size_bits*2)/8/1024:.2f} KB  "
              f"({100-pct:.1f}% reduction both ways)")
        print(f"{'='*70}")
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
        print(f"{'='*70}\n")

    def train(self):
        item_num = self.ui_matrix.shape[1]
        emb_dim  = 2 * self.latent_dim

        server_model = ServerNeuralCollaborativeFiltering(
            item_num=item_num, predictive_factor=self.latent_dim, lora_rank=self.lora_rank)
        torch.save(server_model.state_dict(), "./models/central/server0.pt")

        shared_params = sum(p.numel() for name, p in server_model.named_parameters()
                            if 'lora' not in name)
        payload_size_bits  = get_lora_payload_size_bits(self.lora_rank, item_num, emb_dim, shared_params)
        full_size_bits     = get_full_model_size_bits(self.lora_rank, item_num, emb_dim, shared_params)

        print(f"\n{'='*70}")
        print(f"Model Size Overview  [LoRA rank={self.lora_rank}, no SVD]")
        print(f"{'='*70}")
        print(f"  Full item emb (reference): {full_size_bits/8/1024:.2f} KB")
        print(f"  LoRA payload (up = down):  {payload_size_bits/8/1024:.2f} KB "
              f"({100*payload_size_bits/full_size_bits:.1f}% of full)")
        print(f"  Upload == Download:        symmetric ✓  |  SVD overhead: none ✓")
        print(f"  Bandwidth: {self.bandwidth_profiles.count('slow')} slow, "
              f"{self.bandwidth_profiles.count('medium')} medium, "
              f"{self.bandwidth_profiles.count('fast')} fast")
        print(f"{'='*70}\n")

        for epoch in range(self.aggregation_epochs):
            server_model = ServerNeuralCollaborativeFiltering(
                item_num=item_num, predictive_factor=self.latent_dim, lora_rank=self.lora_rank)
            server_model.load_state_dict(torch.load(f"./models/central/server{epoch}.pt"))
            server_model.eval()

            download_payload = server_model.get_download_payload()
            _ = [client.ncf.to(self.device) for client in self.clients]
            _ = [client.ncf.load_server_weights(
                    {k: v.to(self.device) for k, v in download_payload.items()}
                 ) for client in self.clients]

            client_timings = self.single_round(epoch=epoch, payload_size_bits=payload_size_bits)
            agg_time = federate(self.utils)
            self.print_timing_summary(epoch, client_timings, agg_time,
                                      payload_size_bits, full_size_bits)
            self.timing_log.append({"epoch": epoch, "client_timings": client_timings, "agg_time": agg_time})

if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    fncf = FederatedNCF(dataloader.ratings, num_clients=60, user_per_client_range=[1, 1],
                        aggregation_epochs=100, local_epochs=5, batch_size=128, lora_rank=8)
    fncf.train()