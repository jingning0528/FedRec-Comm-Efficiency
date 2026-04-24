import torch
import numpy as np
import random
import time
import os
import sys
from datetime import datetime
from tqdm import tqdm

from .train_single import NCFTrainer
from .server_model import ServerNeuralCollaborativeFiltering
from dataloader import MovielensDatasetLoader

BANDWIDTH_PROFILES = {
    "slow":   {"upload": 100, "download": 100},
    "medium": {"upload": 100, "download": 100},
    "fast":   {"upload": 100, "download": 100},
}

# BANDWIDTH_PROFILES = {
#     "slow":   {"upload": 2,   "download": 4},
#     "medium": {"upload": 10,  "download": 20},
#     "fast":   {"upload": 50,  "download": 100},
# }


def assign_bandwidth(num_clients, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(num_clients):
        r = rng.random()
        out.append("slow" if r < 0.3 else ("medium" if r < 0.7 else "fast"))
    return out


def get_lora_payload_size_bits(lora_rank, item_num, emb_dim, shared_params):
    """Standard LoRA payload: lora_A (rank×d) + lora_B (N×rank) per embedding pair."""
    lora_params = 2 * (lora_rank * emb_dim + item_num * lora_rank)   # mlp + gmf
    return (lora_params + shared_params) * 32


def get_full_model_size_bits(item_num, emb_dim, shared_params):
    full_item_params = 2 * (item_num * emb_dim)                       # mlp + gmf
    return (full_item_params + shared_params) * 32


def calc_comm_time(size_bits, bandwidth_mbps):
    return size_bits / (bandwidth_mbps * 1e6)


class Utils:
    def __init__(self, num_clients, local_path="./models/local_items/",
                 server_path="./models/central/"):
        self.epoch       = 0
        self.num_clients = num_clients
        self.local_path  = local_path
        self.server_path = server_path

    def get_lora_updates(self):
        return [torch.load(self.local_path + f"lora_dp{i}.pt")
                for i in range(self.num_clients)]


def federate(utils):
    """Standard FedAvg on LoRA adapters (A, B) and shared MLP/output layers."""
    updates = utils.get_lora_updates()
    if not updates:
        utils.epoch += 1
        return 0.0

    prev_state = torch.load(f"./models/central/server{utils.epoch}.pt")
    utils.epoch += 1

    t0          = time.time()
    updates_cpu = [{k: v.cpu() for k, v in u.items()} for u in updates]
    n           = len(updates_cpu)

    new_state = {}
    for k in updates_cpu[0]:
        new_state[k] = sum(u[k] for u in updates_cpu) / n

    # Carry over buffers (E0) and any keys not in the upload payload
    for k in prev_state:
        if k not in new_state:
            new_state[k] = prev_state[k]

    torch.save(new_state, f"./models/central/server{utils.epoch}.pt")
    return time.time() - t0


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


class FederatedNCF:
    def __init__(self,
                 train_matrix:       np.ndarray,
                 num_clients:        int   = 604,
                 aggregation_epochs: int   = 50,
                 local_epochs:       int   = 2,
                 batch_size:         int   = 256,
                 latent_dim:         int   = 64,
                 lora_rank:          int   = 8,
                 lr:                 float = 5e-4,
                 seed:               int   = 0,
                 device:             str   = None,
                 eval_fraction:      float = 0.2,
                 eval_every:         int   = 5):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

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
        self.lora_rank          = lora_rank
        self.lr                 = lr
        self.item_num           = train_matrix.shape[1]

        for p in ["./models/local_items/", "./models/local/", "./models/central/"]:
            os.makedirs(p, exist_ok=True)

        self.bandwidth_profiles = assign_bandwidth(num_clients, seed=seed)
        self.metrics_log        = []
        self.eval_log           = []
        self.timing_log         = []
        self.utils              = Utils(num_clients)

        self.cumulative_time          = 0.0
        self.cumulative_download_time = 0.0
        self.cumulative_train_time    = 0.0
        self.cumulative_upload_time   = 0.0
        self.cumulative_agg_time      = 0.0

        self.eval_every      = eval_every
        rng                  = np.random.default_rng(seed)
        n_eval               = max(1, int(num_clients * eval_fraction))
        self.eval_client_ids = rng.choice(num_clients, size=n_eval, replace=False).tolist()

        assert train_matrix.shape[0] == num_clients

        self.clients = [
            NCFTrainer(train_matrix[i:i+1], epochs=local_epochs,
                       batch_size=batch_size, latent_dim=latent_dim,
                       lora_rank=lora_rank, device=self.device,
                       global_user_offset=i)
            for i in range(num_clients)
        ]
        # Optimizer only touches trainable params (LoRA + user embs)
        self.optimizers = [
            torch.optim.Adam(
                filter(lambda p: p.requires_grad, c.ncf.parameters()), lr=lr
            ) for c in self.clients
        ]

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
        print(f"Users         : {num_clients}")
        print(f"Method        : Standard LoRA FedAvg")
        print(f"Local epochs  : {local_epochs}")
        print(f"Batch size    : {batch_size}")
        print(f"Latent dim    : {latent_dim}")
        print(f"LoRA rank     : {lora_rank}")
        print(f"Learning rate : {lr}")
        print(f"Bandwidth     : "
              f"{self.bandwidth_profiles.count('slow')} slow, "
              f"{self.bandwidth_profiles.count('medium')} medium, "
              f"{self.bandwidth_profiles.count('fast')} fast")
        print(f"Eval fraction : {eval_fraction:.0%}  ({n_eval} / {num_clients} users)")
        print(f"Eval every    : every {eval_every} epoch(s)")
        print(f"{'='*60}")

    def _single_round(self, epoch: int, payload_size_bits: int) -> list:
        timings     = []
        agg_results = {"loss": [], "hit_ratio@10": [], "ndcg@10": []}

        bar = tqdm(enumerate(self.clients), total=self.num_clients,
                   desc=f"Epoch {epoch}")
        for cid, client in bar:
            bw         = BANDWIDTH_PROFILES[self.bandwidth_profiles[cid]]
            dl_time    = calc_comm_time(payload_size_bits, bw["download"])
            t0         = time.time()
            results    = client.train(self.optimizers[cid])
            train_time = time.time() - t0
            ul_time    = calc_comm_time(payload_size_bits, bw["upload"])

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

            # Save LoRA update (lora_A, lora_B + shared layers)
            update = client.ncf.get_lora_update()
            torch.save(update, f"./models/local_items/lora_dp{cid}.pt")
            client.ncf.to(self.device)

            bar.set_postfix({"loss":  f"{results['loss']:.4f}",
                             "HR@10": f"{results['hit_ratio@10']:.4f}"})
        bar.close()

        self.metrics_log.append({
            "epoch":        epoch,
            "loss":         round(float(np.mean(agg_results["loss"])),         6),
            "hit_ratio@10": round(float(np.mean(agg_results["hit_ratio@10"])), 6),
            "ndcg@10":      round(float(np.mean(agg_results["ndcg@10"])),      6),
        })
        return timings

    def _evaluate(self, epoch: int, k: int = 10, n_neg: int = 99):
        hrs, ndcgs, losses = [], [], []
        for cid in self.eval_client_ids:
            res = self.clients[cid].evaluate_standard(n_neg=n_neg, k=k)
            n   = res["evaluated_users"]
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
        print(f"\n[Eval — Epoch {epoch:>3d}]  "
              f"HR@{k} = {hr_mean:.4f}  |  NDCG@{k} = {ndcg_mean:.4f}  |  "
              f"Loss = {loss_mean:.4f}  ({total_n} / {self.num_clients} users)\n")
        return record

    def _print_timing(self, epoch, timings, agg_time, payload_size_bits, full_size_bits):
        by_bw = {"slow": [], "medium": [], "fast": []}
        for t in timings:
            by_bw[t["bandwidth"]].append(t)

        all_totals = [t["total_time_s"] for t in timings]
        round_time = max(all_totals) + agg_time

        self.cumulative_download_time += max(t["download_time_s"] for t in timings)
        self.cumulative_train_time    += max(t["train_time_s"]    for t in timings)
        self.cumulative_upload_time   += max(t["upload_time_s"]   for t in timings)
        self.cumulative_agg_time      += agg_time
        self.cumulative_time          += round_time

        pct = 100 * payload_size_bits / full_size_bits
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Timing  [Standard LoRA rank={self.lora_rank} | "
              f"payload={payload_size_bits/8/1024:.2f} KB ({pct:.1f}% of full)]")
        print(f"{'='*60}")
        for name, ts in by_bw.items():
            if not ts:
                continue
            print(f"  [{name.upper():6s}] n={len(ts):3d} | "
                  f"download={np.mean([t['download_time_s'] for t in ts]):.4f}s | "
                  f"train={np.mean([t['train_time_s'] for t in ts]):.4f}s | "
                  f"upload={np.mean([t['upload_time_s'] for t in ts]):.4f}s")
        bottleneck = max(all_totals)
        print(f"  [ROUND ] bottleneck={bottleneck:.4f}s | agg={agg_time:.4f}s | "
              f"total={round_time:.4f}s")
        print(f"  [CUMUL ] total={self.cumulative_time:.4f}s "
              f"({self.cumulative_time/60:.2f} min) over {epoch+1} rounds")
        print(f"{'='*60}")

    def train(self):
        item_num = self.item_num
        emb_dim  = 2 * self.latent_dim

        server_model = ServerNeuralCollaborativeFiltering(
            item_num=item_num, predictive_factor=self.latent_dim,
            lora_rank=self.lora_rank)
        torch.save(server_model.state_dict(), "./models/central/server0.pt")

        shared_params = sum(
            p.numel() for name, p in server_model.named_parameters()
            if "lora" not in name
        )
        payload_size_bits = get_lora_payload_size_bits(
            self.lora_rank, item_num, emb_dim, shared_params)
        full_size_bits    = get_full_model_size_bits(item_num, emb_dim, shared_params)

        print(f"\n{'='*60}")
        print(f"Standard LoRA FedAvg  [rank={self.lora_rank}]")
        print(f"  Full item emb : {full_size_bits/8/1024:.2f} KB")
        print(f"  LoRA payload  : {payload_size_bits/8/1024:.2f} KB "
              f"({100*payload_size_bits/full_size_bits:.1f}%)")
        print(f"{'='*60}\n")

        for epoch in range(self.aggregation_epochs):
            # 1. Distribute server state to all clients
            server_model = ServerNeuralCollaborativeFiltering(
                item_num=item_num, predictive_factor=self.latent_dim,
                lora_rank=self.lora_rank)
            server_model.load_state_dict(
                torch.load(f"./models/central/server{epoch}.pt"))
            server_model.eval()

            download_payload = server_model.get_download_payload()
            for client in self.clients:
                client.ncf.to(self.device)
                client.ncf.load_server_weights(
                    {k: v.to(self.device) for k, v in download_payload.items()})

            # 2. Local training (only LoRA + user embs updated)
            timings = self._single_round(epoch, payload_size_bits)

            # 3. FedAvg on LoRA adapters + shared layers
            agg_time = federate(self.utils)

            # 4. Timing summary
            self._print_timing(epoch, timings, agg_time,
                               payload_size_bits, full_size_bits)
            self.timing_log.append({"epoch": epoch, "timings": timings,
                                    "agg_time": agg_time})

            # 5. Evaluate
            if (epoch + 1) % self.eval_every == 0 or epoch == self.aggregation_epochs - 1:
                self._evaluate(epoch, k=10, n_neg=99)

        self._final_summary()
        self.logger.close()

    def _final_summary(self):
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        if self.eval_log:
            print(f"\n-- Standard leave-one-out evaluation --")
            print(f"{'Epoch':>6} | {'HR@10':>8} | {'NDCG@10':>10} | "
                  f"{'EvalLoss':>10} | {'Users':>7}")
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
            total_round = self.cumulative_time
            print(f"\nTotal round time : {total_round:.2f}s  ({total_round/60:.2f} min)")
            print(f"Avg time / round : {total_round/len(self.timing_log):.4f}s")
            print(f"Total agg time   : {self.cumulative_agg_time:.2f}s")

        print(f"\nLog saved → {self.log_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    DEVICE = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    dataloader   = MovielensDatasetLoader()
    all_ratings  = dataloader.ratings
    n_clients    = 604
    train_matrix = all_ratings[:n_clients]

    fncf = FederatedNCF(
        train_matrix       = train_matrix,
        num_clients        = n_clients,
        aggregation_epochs = 50,
        local_epochs       = 2,
        batch_size         = 256,
        latent_dim         = 64,
        lora_rank          = 8,
        lr                 = 5e-4,
        seed               = 42,
        device             = DEVICE,
        eval_fraction      = 0.2,
        eval_every         = 5,
    )
    fncf.train()