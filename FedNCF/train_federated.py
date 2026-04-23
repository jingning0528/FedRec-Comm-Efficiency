import torch
import numpy as np
import random
import copy
import time
import os
import sys
from datetime import datetime
from tqdm import tqdm

from .train_single import NCFTrainer
from .server_model import ServerNeuralCollaborativeFiltering
from dataloader import MovielensDatasetLoader


# ── Bandwidth simulation ──────────────────────────────────────────────────────

BANDWIDTH_PROFILES = {
    "slow":   {"upload": 100,  "download": 100},   # 30%
    "medium": {"upload": 100,  "download": 100},   # 40%
    "fast":   {"upload": 100,  "download": 100},  # 30%
}

# BANDWIDTH_PROFILES = {
#     "slow":   {"upload":  1,  "download":  2},   # 30%
#     "medium": {"upload": 10,  "download": 20},   # 40%
#     "fast":   {"upload": 50,  "download": 100},  # 30%
# }

def assign_bandwidth(num_clients: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(num_clients):
        r = rng.random()
        out.append("slow" if r < 0.3 else ("medium" if r < 0.7 else "fast"))
    return out

def model_size_bits(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters()) * 32

def comm_time(size_bits: int, bandwidth_mbps: float) -> float:
    return size_bits / (bandwidth_mbps * 1e6)


# ── FedAvg aggregation ────────────────────────────────────────────────────────

class ModelStore:
    """Thin wrapper around model save/load paths."""
    def __init__(self, local_items="./models/local_items/",
                 local="./models/local/",
                 central="./models/central/"):
        self.local_items = local_items
        self.local       = local
        self.central     = central
        for p in [local_items, local, central]:
            os.makedirs(p, exist_ok=True)

    def save_client(self, model, client_id):
        torch.jit.save(torch.jit.script(model.cpu()), f"{self.local}dp{client_id}.pt")

    def save_item_model(self, item_model, client_id):
        torch.jit.save(torch.jit.script(item_model.cpu()),
                       f"{self.local_items}dp{client_id}.pt")

    def load_item_model(self, client_id):
        return torch.jit.load(f"{self.local_items}dp{client_id}.pt")

    def save_server(self, model, epoch):
        torch.jit.save(torch.jit.script(model.cpu()), f"{self.central}server{epoch}.pt")

    def load_server(self, epoch, device):
        return torch.jit.load(f"{self.central}server{epoch}.pt", map_location=device)


def fedavg(store: ModelStore, num_clients: int, server_model, epoch: int) -> float:
    """FedAvg: average item-model weights from all clients, save as new server."""
    t0 = time.time()
    client_models = [store.load_item_model(i) for i in range(num_clients)]

    avg_dict = copy.deepcopy(client_models[0].state_dict())
    for cm in client_models[1:]:
        for k, v in cm.state_dict().items():
            avg_dict[k] += v
    for k in avg_dict:
        avg_dict[k] = avg_dict[k] / num_clients

    server_model.load_state_dict(avg_dict)
    store.save_server(server_model, epoch + 1)
    return time.time() - t0


# ── Logger ────────────────────────────────────────────────────────────────────

class TeeLogger:
    def __init__(self, log_path: str):
        self._terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, 'w', buffering=1)

    def write(self, msg):
        self._terminal.write(msg)
        self._file.write(msg)

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._terminal


# ── Federated NCF ─────────────────────────────────────────────────────────────

class FederatedNCF:
    def __init__(self,
                 train_matrix:       np.ndarray,
                 num_clients:        int   = 604,
                 aggregation_epochs: int   = 50,
                 local_epochs:       int   = 1,
                 batch_size:         int   = 256,
                 latent_dim:         int   = 64,
                 lr:                 float = 1e-4,
                 seed:               int   = 0,
                 device:             str   = None,
                 eval_fraction:      float = 1.0,   # ← 0.0–1.0: fraction of clients evaluated
                 eval_every:         int   = 1):     # ← evaluate every N epochs

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Use passed device, or auto-detect
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps"  if torch.backends.mps.is_available() else
                "cpu"
            )

        self.num_clients        = num_clients
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs       = local_epochs
        self.batch_size         = batch_size
        self.latent_dim         = latent_dim
        self.lr                 = lr
        self.item_num           = train_matrix.shape[1]

        self.store              = ModelStore()
        self.bandwidth_profiles = assign_bandwidth(num_clients, seed=seed)
        self.metrics_log        = []
        self.eval_log           = []
        self.timing_log         = []

        # ── One client per user — leave-one-out split happens inside NCFTrainer ──
        # train_matrix[i] → NCFTrainer splits it into:
        #   - train interactions (all except held-out item)  → used for local SGD
        #   - test item (held-out)                           → used for evaluate_standard()
        assert train_matrix.shape[0] == num_clients, \
            f"num_clients={num_clients} must equal train_matrix rows={train_matrix.shape[0]}"

        self.clients = [
            NCFTrainer(train_matrix[i:i+1], epochs=local_epochs,
                       batch_size=batch_size, latent_dim=latent_dim,
                       device=self.device, global_user_offset=i)
            for i in range(num_clients)
        ]
        self.optimizers = [
            torch.optim.Adam(c.ncf.parameters(), lr=lr) for c in self.clients
        ]

        # ── Logging ───────────────────────────────────────────────────────────
        root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir   = os.path.join(root, "result_figure")
        folder    = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(log_dir, f"{folder}-{timestamp}.txt")
        self.logger   = TeeLogger(self.log_path)
        sys.stdout    = self.logger

        print(f"Log file : {self.log_path}")
        print(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device   : {self.device}")
        print(f"{'='*60}")
        print(f"Dataset       : ML-1M  |  Items: {self.item_num}")
        print(f"Users         : {num_clients}  (same users train & test, leave-one-out split)")
        print(f"Local epochs  : {local_epochs}")
        print(f"Batch size    : {batch_size}")
        print(f"Latent dim    : {latent_dim}")
        print(f"Learning rate : {lr}")
        print(f"Bandwidth     : "
              f"{self.bandwidth_profiles.count('slow')} slow, "
              f"{self.bandwidth_profiles.count('medium')} medium, "
              f"{self.bandwidth_profiles.count('fast')} fast")
        print(f"Eval fraction : {eval_fraction:.0%}  ({n_eval} / {num_clients} users)")
        print(f"Eval every    : every {eval_every} epoch(s)")
        print(f"{'='*60}")

    # ── Single training round ─────────────────────────────────────────────────

    def _single_round(self, epoch: int, server_bits: int, item_bits: int) -> list:
        timings     = []
        agg_results = {"loss": [], "hit_ratio@10": [], "ndcg@10": []}

        bar = tqdm(enumerate(self.clients), total=self.num_clients,
                   desc=f"Epoch {epoch}")
        for cid, client in bar:
            bw         = BANDWIDTH_PROFILES[self.bandwidth_profiles[cid]]
            dl_time    = comm_time(server_bits, bw["download"])
            t0         = time.time()
            results    = client.train(self.optimizers[cid])
            train_time = time.time() - t0
            ul_time    = comm_time(item_bits, bw["upload"])

            timings.append({
                "client_id":       cid,
                "bandwidth":       self.bandwidth_profiles[cid],
                "download_time_s": round(dl_time,    4),
                "train_time_s":    round(train_time, 4),
                "upload_time_s":   round(ul_time,    4),
                "total_time_s":    round(dl_time + train_time + ul_time, 4),
            })
            for key in ["loss", "hit_ratio@10", "ndcg@10"]:
                agg_results[key].append(results[key])

            self.store.save_client(client.ncf, cid)
            client.ncf.to(self.device)          # ← restore to device after save moves it to CPU

            bar.set_postfix({"loss": f"{results['loss']:.4f}",
                             "HR@10": f"{results['hit_ratio@10']:.4f}"})
        bar.close()

        self.metrics_log.append({
            "epoch":        epoch,
            "loss":         round(float(np.mean(agg_results["loss"])),         6),
            "hit_ratio@10": round(float(np.mean(agg_results["hit_ratio@10"])), 6),
            "ndcg@10":      round(float(np.mean(agg_results["ndcg@10"])),      6),
        })
        return timings

    # ── Extract item-only models for aggregation ──────────────────────────────

    def _extract_item_models(self):
        for cid in range(self.num_clients):
            full_model = torch.jit.load(f"./models/local/dp{cid}.pt")
            item_model = ServerNeuralCollaborativeFiltering(
                item_num=self.item_num, predictive_factor=self.latent_dim)
            item_model.set_weights(full_model)
            self.store.save_item_model(item_model, cid)

    # ── Standard evaluation ───────────────────────────────────────────────────

    def _evaluate(self, epoch: int, k: int = 10, n_neg: int = 99):
        """
        Evaluate every training client using their held-out test item.
        Each client's model already has up-to-date weights from local training.
        No fine-tuning needed — user embeddings were trained during local SGD.
        """
        hrs, ndcgs, losses = [], [], []
        for cid in self.eval_client_ids:
            client = self.clients[cid]
            res    = client.evaluate_standard(n_neg=n_neg, k=k)
            n      = res["evaluated_users"]
            if n > 0:
                hrs.extend(   [res[f"hr@{k}"]]   * n)
                ndcgs.extend( [res[f"ndcg@{k}"]] * n)
                losses.extend([res["eval_loss"]]  * n)

        hr_mean   = float(np.mean(hrs))    if hrs    else 0.0
        ndcg_mean = float(np.mean(ndcgs))  if ndcgs  else 0.0
        loss_mean = float(np.mean(losses)) if losses else 0.0
        total_n   = len(hrs)

        record = {
            "epoch":           epoch,
            f"hr@{k}":         round(hr_mean,   6),
            f"ndcg@{k}":       round(ndcg_mean, 6),
            "eval_loss":       round(loss_mean, 6),
            "evaluated_users": total_n,
        }
        self.eval_log.append(record)

        print(f"\n[Standard Eval — Epoch {epoch:>3d}]  "
              f"HR@{k} = {hr_mean:.4f}  |  NDCG@{k} = {ndcg_mean:.4f}  |  "
              f"Eval Loss = {loss_mean:.4f}  ({total_n} / {self.num_clients} users)\n")
        return record

    # ── Timing summary ────────────────────────────────────────────────────────

    def _print_timing(self, epoch: int, timings: list, agg_time: float):
        by_bw = {"slow": [], "medium": [], "fast": []}
        for t in timings:
            by_bw[t["bandwidth"]].append(t)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Timing Summary")
        print(f"{'='*60}")
        for name, ts in by_bw.items():
            if not ts:
                continue
            print(f"  [{name.upper():6s}] n={len(ts):3d} | "
                  f"download={np.mean([t['download_time_s'] for t in ts]):.4f}s | "
                  f"train={np.mean([t['train_time_s'] for t in ts]):.4f}s | "
                  f"upload={np.mean([t['upload_time_s'] for t in ts]):.4f}s | "
                  f"total={np.mean([t['total_time_s'] for t in ts]):.4f}s")
        bottleneck = max(t["total_time_s"] for t in timings)
        print(f"  [ROUND ] bottleneck = {bottleneck:.4f}s")
        print(f"  [AGG   ] agg time   = {agg_time:.4f}s")
        print(f"  [TOTAL ] round time = {bottleneck + agg_time:.4f}s")
        print(f"{'='*60}")

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        server_model = ServerNeuralCollaborativeFiltering(
            item_num=self.item_num, predictive_factor=self.latent_dim)
        self.store.save_server(server_model, 0)

        server_bits = model_size_bits(server_model)
        item_bits   = server_bits
        print(f"Server model size : {server_bits / 8 / 1024:.2f} KB")

        for epoch in range(self.aggregation_epochs):
            # 1. Distribute server item weights to all clients
            sv = self.store.load_server(epoch, self.device)
            for client in self.clients:
                client.ncf.to(self.device)      # ← ensure on device before loading weights
                client.ncf.load_server_weights(sv)

            # 2. Local training (trains BOTH user embeddings AND item embeddings)
            timings  = self._single_round(epoch, server_bits, item_bits)

            # 3. Extract item-only models & FedAvg (user embeddings stay local)
            self._extract_item_models()
            agg_time = fedavg(self.store, self.num_clients, server_model, epoch)

            # 4. Print timing
            self._print_timing(epoch, timings, agg_time)
            self.timing_log.append({"epoch": epoch,
                                    "timings": timings, "agg_time": agg_time})

            # 5. Evaluate only on selected epochs and fraction of users
            if (epoch + 1) % self.eval_every == 0 or epoch == self.aggregation_epochs - 1:
                self._evaluate(epoch, k=10, n_neg=99)

        self._final_summary()
        self.logger.close()

    # ── Final summary ─────────────────────────────────────────────────────────

    def _final_summary(self):
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        print(f"\n-- Training metrics (batch-level, reference only) --")
        print(f"{'Epoch':>6} | {'Loss':>8} | {'HR@10(train)':>12} | {'NDCG@10(train)':>14}")
        print("-" * 48)
        for m in self.metrics_log:
            print(f"{m['epoch']:>6} | {m['loss']:>8.4f} | "
                  f"{m['hit_ratio@10']:>12.4f} | {m['ndcg@10']:>14.4f}")

        if self.eval_log:
            print(f"\n-- Standard leave-one-out evaluation (1 pos + 99 neg, paper-standard) --")
            print(f"{'Epoch':>6} | {'HR@10':>8} | {'NDCG@10':>10} | {'EvalLoss':>10} | {'Users':>7}")
            print("-" * 52)
            for e in self.eval_log:
                print(f"{e['epoch']:>6} | {e['hr@10']:>8.4f} | "
                      f"{e['ndcg@10']:>10.4f} | {e['eval_loss']:>10.4f} | "
                      f"{e['evaluated_users']:>7}")
            best_hr   = max(self.eval_log, key=lambda x: x["hr@10"])
            best_ndcg = max(self.eval_log, key=lambda x: x["ndcg@10"])
            last      = self.eval_log[-1]
            print(f"\nBest  HR@10   : {best_hr['hr@10']:.6f}  (epoch {best_hr['epoch']})")
            print(f"Best  NDCG@10 : {best_ndcg['ndcg@10']:.6f}  (epoch {best_ndcg['epoch']})")
            print(f"Final HR@10   : {last['hr@10']:.6f}")
            print(f"Final NDCG@10 : {last['ndcg@10']:.6f}")

        if self.timing_log:
            rounds      = [max(t["total_time_s"] for t in tl["timings"])
                           for tl in self.timing_log]
            total_round = sum(rounds)
            total_agg   = sum(t["agg_time"] for t in self.timing_log)
            print(f"\nTotal rounds     : {len(self.timing_log)}")
            print(f"Total round time : {total_round:.2f}s  ({total_round/60:.2f} min)")
            print(f"Avg time / round : {total_round/len(self.timing_log):.4f}s")
            print(f"Total agg time   : {total_agg:.2f}s")

        print(f"\nLog saved → {self.log_path}")
        print(f"{'='*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Device: auto-detects GPU, falls back to CPU ───────────────────────────
    # Colab (NVIDIA): "cuda"
    # MacBook (Apple Silicon): "mps"
    # MacBook (CPU):  "cpu"
    DEVICE = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {DEVICE}")

    dataloader   = MovielensDatasetLoader()
    all_ratings  = dataloader.ratings          # (6040, 3706)

    n_clients    = 604
	#n_clients    = 604
    train_matrix = all_ratings[:n_clients]

    fncf = FederatedNCF(
        train_matrix       = train_matrix,
        num_clients        = n_clients,
        aggregation_epochs = 50,
        local_epochs       = 1,
        batch_size         = 256,
        latent_dim         = 64,
        lr                 = 1e-4,
        seed               = 42,
        device             = DEVICE,
        eval_fraction      = 0.2,   # evaluate on 20% of users (~121) — fast
        eval_every         = 5,     # evaluate every 5 epochs
    )
    fncf.train()
    
	# fncf = FederatedNCF(
    #     train_matrix       = train_matrix,
    #     num_clients        = n_clients,
    #     aggregation_epochs = 50,
    #     local_epochs       = 2, # lower 
    #     batch_size         = 256, # lower
    #     latent_dim         = 64, 
    #     lr                 = 5e-4, # lower
    #     seed               = 42,
    #     device             = DEVICE,           # ← pass device explicitly
    # )
    # fncf.train()