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
    "slow":   {"upload": 100, "download": 100},
    "medium": {"upload": 100, "download": 100},
    "fast":   {"upload": 100, "download": 100},
}


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


# ── Model store ───────────────────────────────────────────────────────────────

class ModelStore:
    def __init__(self, local_items="./models/local_items/",
                 local="./models/local/",
                 central="./models/central/"):
        self.local_items = local_items
        self.local       = local
        self.central     = central
        for p in [local_items, local, central]:
            os.makedirs(p, exist_ok=True)

    def save_client(self, model, client_id):
        torch.save(model.cpu().state_dict(), f"{self.local}dp{client_id}.pt")

    def load_client_state(self, client_id):
        return torch.load(f"{self.local}dp{client_id}.pt", map_location="cpu")

    # ── SCoLR: save/load partial payload dict (not a full scripted model) ─────

    def save_partial(self, payload: dict, client_id: int):
        """Save SCoLR partial update (subsampled A rows + full B + MLP)."""
        torch.save(payload, f"{self.local_items}scolr_{client_id}.pt")

    def load_partial(self, client_id: int) -> dict:
        return torch.load(f"{self.local_items}scolr_{client_id}.pt",
                          map_location="cpu")

    def save_server(self, model, epoch):
        torch.save(model.cpu().state_dict(), f"{self.central}server{epoch}.pt")

    def load_server(self, epoch, device):
        sv = ServerNeuralCollaborativeFiltering.__new__(
            ServerNeuralCollaborativeFiltering)
        # Use a temp instance just to load state dict; actual loading done in train()
        return torch.load(f"{self.central}server{epoch}.pt", map_location=device)


# ── SCoLR sparse FedAvg ───────────────────────────────────────────────────────

def fedavg_scolr(store: ModelStore,
                 num_clients: int,
                 server_model: ServerNeuralCollaborativeFiltering,
                 epoch: int,
                 item_num: int) -> float:
    """
    SCoLR FedAvg (Algorithm A.2 from the paper, without secure aggregation):

    1. Collect partial uploads from all clients.
    2. For A (item rows): accumulate sum and count per item index;
       divide by count → only updates items seen by ≥1 client this round.
       Items with zero coverage carry over the server's previous value.
    3. For B, MLP tower, output heads: standard full FedAvg (all clients report).
    """
    t0 = time.time()

    # Load first payload to determine rank
    first     = store.load_partial(0)
    rank      = first["mlp_A_rows"].shape[1]

    # ── Per-item accumulators for A rows ──────────────────────────────────────
    mlp_A_sum   = torch.zeros(item_num, rank)
    gmf_A_sum   = torch.zeros(item_num, rank)
    item_counts = torch.zeros(item_num, dtype=torch.long)   # coverage counter

    # ── Global accumulators for B, MLP, output heads ─────────────────────────
    mlp_B_sum       = torch.zeros_like(first["mlp_B"])
    gmf_B_sum       = torch.zeros_like(first["gmf_B"])
    mlp_state_sum   = {k: torch.zeros_like(v) for k, v in first["mlp_state"].items()}
    gmf_out_w_sum   = torch.zeros_like(first["gmf_out_weight"])
    mlp_out_w_sum   = torch.zeros_like(first["mlp_out_weight"])
    out_logits_w_sum = torch.zeros_like(first["output_logits_w"])
    out_logits_b_sum = torch.zeros_like(first["output_logits_b"])

    for cid in range(num_clients):
        p = store.load_partial(cid)
        idx = p["item_indices"]                    # (s,)

        # ── Sparse A accumulation ─────────────────────────────────────────────
        mlp_A_sum.index_add_(0, idx, p["mlp_A_rows"])
        gmf_A_sum.index_add_(0, idx, p["gmf_A_rows"])
        item_counts[idx] += 1

        # ── Full accumulation ─────────────────────────────────────────────────
        mlp_B_sum        += p["mlp_B"]
        gmf_B_sum        += p["gmf_B"]
        for k in mlp_state_sum:
            mlp_state_sum[k] += p["mlp_state"][k]
        gmf_out_w_sum    += p["gmf_out_weight"]
        mlp_out_w_sum    += p["mlp_out_weight"]
        out_logits_w_sum += p["output_logits_w"]
        out_logits_b_sum += p["output_logits_b"]

    # ── Average A rows only for items with ≥1 report ─────────────────────────
    covered       = item_counts > 0                        # (item_num,) bool
    covered_idx   = covered.nonzero(as_tuple=True)[0]
    counts_f      = item_counts[covered_idx].float().unsqueeze(1)  # (s, 1)
    avg_mlp_A_covered = mlp_A_sum[covered_idx] / counts_f
    avg_gmf_A_covered = gmf_A_sum[covered_idx] / counts_f

    # ── Average global tensors ────────────────────────────────────────────────
    avg_mlp_B        = mlp_B_sum        / num_clients
    avg_gmf_B        = gmf_B_sum        / num_clients
    avg_mlp_state    = {k: v / num_clients for k, v in mlp_state_sum.items()}
    avg_gmf_out_w    = gmf_out_w_sum    / num_clients
    avg_mlp_out_w    = mlp_out_w_sum    / num_clients
    avg_out_logits_w = out_logits_w_sum / num_clients
    avg_out_logits_b = out_logits_b_sum / num_clients

    # ── Apply sparse update to server model ───────────────────────────────────
    server_model.apply_scolr_aggregate(
        avg_mlp_A         = avg_mlp_A_covered,
        avg_gmf_A         = avg_gmf_A_covered,
        item_indices      = covered_idx,
        avg_mlp_B         = avg_mlp_B,
        avg_gmf_B         = avg_gmf_B,
        avg_mlp_state     = avg_mlp_state,
        avg_gmf_out_w     = avg_gmf_out_w,
        avg_mlp_out_w     = avg_mlp_out_w,
        avg_output_logits_w = avg_out_logits_w,
        avg_output_logits_b = avg_out_logits_b,
    )

    store.save_server(server_model, epoch + 1)

    coverage_pct = covered.float().mean().item() * 100
    return time.time() - t0, coverage_pct


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


# ── Federated NCF (SCoLR) ─────────────────────────────────────────────────────

class FederatedNCF:
    def __init__(self,
                 train_matrix:       np.ndarray,
                 num_clients:        int   = 604,
                 aggregation_epochs: int   = 50,
                 local_epochs:       int   = 1,
                 batch_size:         int   = 256,
                 latent_dim:         int   = 64,
                 rank:               int   = 16,
                 subsample_rate:     float = 0.1,   # ← SCoLR: fraction of items uploaded
                 lr:                 float = 1e-4,
                 seed:               int   = 0,
                 device:             str   = None,
                 eval_fraction:      float = 1.0,
                 eval_every:         int   = 1):

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
        self.rank               = rank
        self.subsample_rate     = subsample_rate   # ← p in the paper
        self.lr                 = lr
        self.item_num           = train_matrix.shape[1]

        self.store              = ModelStore()
        self.bandwidth_profiles = assign_bandwidth(num_clients, seed=seed)
        self.metrics_log        = []
        self.eval_log           = []
        self.timing_log         = []

        self.eval_every      = eval_every
        rng_eval             = np.random.default_rng(seed)
        n_eval               = max(1, int(num_clients * eval_fraction))
        self.eval_client_ids = rng_eval.choice(num_clients, size=n_eval,
                                               replace=False).tolist()

        # Per-client RNG for reproducible subsampling
        self._client_rngs = [np.random.default_rng(seed + cid)
                             for cid in range(num_clients)]

        assert train_matrix.shape[0] == num_clients

        self.clients = [
            NCFTrainer(train_matrix[i:i+1], epochs=local_epochs,
                       batch_size=batch_size, latent_dim=latent_dim,
                       rank=rank, device=self.device, global_user_offset=i)
            for i in range(num_clients)
        ]
        self.optimizers = [
            torch.optim.Adam(c.ncf.parameters(), lr=lr) for c in self.clients
        ]

        # ── SCoLR communication cost ──────────────────────────────────────────
        # Upload per client: s*rank (subsampled A rows) + d*rank (B) + MLP factors
        s = max(1, int(self.item_num * subsample_rate))
        colr_A_params   = self.item_num * rank * 2   # full CoLR (mlp + gmf)
        scolr_A_params  = s * rank * 2               # SCoLR subsampled
        saving_vs_colr  = 100.0 * (1 - scolr_A_params / colr_A_params)
        full_item_params = self.item_num * 2 * latent_dim * 2
        scolr_total_item = scolr_A_params + latent_dim * rank * 2 * 2  # A + B parts
        saving_vs_full   = 100.0 * (1 - scolr_total_item / full_item_params)

        # ── Logging ───────────────────────────────────────────────────────────
        root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir   = os.path.join(root, "result_figure")
        folder    = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(log_dir, f"{folder}-{timestamp}.txt")
        self.logger   = TeeLogger(self.log_path)
        sys.stdout    = self.logger

        print(f"Log file  : {self.log_path}")
        print(f"Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device    : {self.device}")
        print(f"{'='*60}")
        print(f"Method        : SCoLR (Subsampling CoLR, no secure agg)")
        print(f"Dataset       : ML-1M  |  Items: {self.item_num}")
        print(f"Users         : {num_clients}")
        print(f"Local epochs  : {local_epochs}")
        print(f"Batch size    : {batch_size}")
        print(f"Latent dim    : {latent_dim}")
        print(f"CoLR rank     : {rank}")
        print(f"Subsample p   : {subsample_rate:.0%}  ({s} / {self.item_num} items per client)")
        print(f"Comm saving   : {saving_vs_colr:.1f}% vs CoLR  |  "
              f"{saving_vs_full:.1f}% vs full-rank")
        print(f"Learning rate : {lr}")
        print(f"Bandwidth     : "
              f"{self.bandwidth_profiles.count('slow')} slow, "
              f"{self.bandwidth_profiles.count('medium')} medium, "
              f"{self.bandwidth_profiles.count('fast')} fast")
        print(f"Eval fraction : {eval_fraction:.0%}  ({n_eval} / {num_clients} users)")
        print(f"Eval every    : every {eval_every} epoch(s)")
        print(f"{'='*60}")

    # ── Compute per-client upload size for SCoLR ──────────────────────────────

    def _scolr_upload_bits(self, num_sampled_items: int) -> int:
        """
        Upload payload per client:
          - A rows (subsampled):  num_sampled_items * rank * 2  (mlp + gmf)
          - B matrix (full):      2 * latent_dim * rank * 2
          - MLP LowRankLinear:    sum of A,B params across 3 layers
          - output heads:         small, fixed size
        """
        r, d = self.rank, self.latent_dim
        # item A rows
        bits  = num_sampled_items * r * 2 * 32
        # item B
        bits += (2 * d * r) * 2 * 32
        # MLP: 3 LowRankLinear layers (A+B per layer) + biases
        mlp_dims = [(4*d, 2*d), (2*d, d), (d, d//2)]
        for (in_f, out_f) in mlp_dims:
            bits += (out_f * r + in_f * r + out_f) * 32
        # output heads (small)
        bits += (2*d + d//2 + d + 1) * 32
        return bits

    # ── Single training round ─────────────────────────────────────────────────

    def _single_round(self, epoch: int, server_bits: int) -> list:
        timings     = []
        agg_results = {"loss": [], "hit_ratio@10": [], "ndcg@10": []}
        s           = max(1, int(self.item_num * self.subsample_rate))

        bar = tqdm(enumerate(self.clients), total=self.num_clients,
                   desc=f"Epoch {epoch}")
        for cid, client in bar:
            bw      = BANDWIDTH_PROFILES[self.bandwidth_profiles[cid]]
            dl_time = comm_time(server_bits, bw["download"])

            t0         = time.time()
            results    = client.train(self.optimizers[cid])
            train_time = time.time() - t0

            # ── SCoLR: randomly subsample item indices ────────────────────────
            item_indices = torch.from_numpy(
                self._client_rngs[cid].choice(self.item_num, size=s, replace=False)
            ).long()

            # ── Extract partial payload (subsampled A + full B + MLP) ─────────
            payload  = client.ncf.get_scolr_partial_update(item_indices)
            self.store.save_partial(payload, cid)

            ul_bits  = self._scolr_upload_bits(s)
            ul_time  = comm_time(ul_bits, bw["upload"])

            timings.append({
                "client_id":       cid,
                "bandwidth":       self.bandwidth_profiles[cid],
                "download_time_s": round(dl_time,    4),
                "train_time_s":    round(train_time, 4),
                "upload_time_s":   round(ul_time,    4),
                "total_time_s":    round(dl_time + train_time + ul_time, 4),
                "items_uploaded":  s,
            })
            for key in ["loss", "hit_ratio@10", "ndcg@10"]:
                agg_results[key].append(results[key])

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

    # ── Standard evaluation ───────────────────────────────────────────────────

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

        record = {"epoch": epoch, f"hr@{k}": round(hr_mean, 6),
                  f"ndcg@{k}": round(ndcg_mean, 6),
                  "eval_loss": round(loss_mean, 6), "evaluated_users": total_n}
        self.eval_log.append(record)
        print(f"\n[SCoLR Eval — Epoch {epoch:>3d}]  "
              f"HR@{k} = {hr_mean:.4f}  |  NDCG@{k} = {ndcg_mean:.4f}  |  "
              f"Eval Loss = {loss_mean:.4f}  ({total_n} / {self.num_clients} users)\n")
        return record

    # ── Timing summary ────────────────────────────────────────────────────────

    def _print_timing(self, epoch, timings, agg_time, coverage_pct):
        by_bw = {"slow": [], "medium": [], "fast": []}
        for t in timings:
            by_bw[t["bandwidth"]].append(t)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Timing Summary  "
              f"[SCoLR rank={self.rank} p={self.subsample_rate:.0%}]")
        print(f"{'='*60}")
        for name, ts in by_bw.items():
            if not ts:
                continue
            print(f"  [{name.upper():6s}] n={len(ts):3d} | "
                  f"dl={np.mean([t['download_time_s'] for t in ts]):.4f}s | "
                  f"train={np.mean([t['train_time_s'] for t in ts]):.4f}s | "
                  f"ul={np.mean([t['upload_time_s'] for t in ts]):.4f}s | "
                  f"total={np.mean([t['total_time_s'] for t in ts]):.4f}s")
        bottleneck = max(t["total_time_s"] for t in timings)
        print(f"  [ROUND ] bottleneck  = {bottleneck:.4f}s")
        print(f"  [AGG   ] agg time    = {agg_time:.4f}s")
        print(f"  [TOTAL ] round time  = {bottleneck + agg_time:.4f}s")
        print(f"  [COV   ] item coverage this round = {coverage_pct:.1f}%")
        print(f"{'='*60}")

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        server_model = ServerNeuralCollaborativeFiltering(
            item_num=self.item_num, predictive_factor=self.latent_dim,
            rank=self.rank)
        self.store.save_server(server_model, 0)

        # Download = full server model (all item factors)
        server_bits = model_size_bits(server_model)
        s           = max(1, int(self.item_num * self.subsample_rate))
        print(f"SCoLR server download  : {server_bits / 8 / 1024:.2f} KB")
        print(f"SCoLR upload per client: "
              f"{self._scolr_upload_bits(s) / 8 / 1024:.2f} KB  "
              f"(p={self.subsample_rate:.0%}, {s} items)")

        for epoch in range(self.aggregation_epochs):
            # 1. Broadcast full server item weights to all clients
            sv_state = self.store.load_server(epoch, self.device)
            for client in self.clients:
                client.ncf.to(self.device)
                # Manually load state dict into the client NCF
                # (server state dict matches item embedding + MLP keys)
                client_sd = client.ncf.state_dict()
                for k, v in sv_state.items():
                    # Only copy item-side keys (skip user embeddings)
                    if k in client_sd and "user_embedding" not in k:
                        client_sd[k] = v.to(self.device)
                client.ncf.load_state_dict(client_sd, strict=False)

            # 2. Local training + extract SCoLR partial payloads
            timings = self._single_round(epoch, server_bits)

            # 3. SCoLR sparse FedAvg
            agg_time, coverage_pct = fedavg_scolr(
                self.store, self.num_clients, server_model,
                epoch, self.item_num)

            # 4. Push updated server weights back to clients' NCF models
            sv_state_new = self.store.load_server(epoch + 1, self.device)
            for client in self.clients:
                client_sd = client.ncf.state_dict()
                for k, v in sv_state_new.items():
                    if k in client_sd and "user_embedding" not in k:
                        client_sd[k] = v.to(self.device)
                client.ncf.load_state_dict(client_sd, strict=False)

            # 5. Timing + logging
            self._print_timing(epoch, timings, agg_time, coverage_pct)
            self.timing_log.append({"epoch": epoch, "timings": timings,
                                    "agg_time": agg_time,
                                    "coverage_pct": coverage_pct})

            # 6. Evaluate
            if (epoch + 1) % self.eval_every == 0 or \
               epoch == self.aggregation_epochs - 1:
                self._evaluate(epoch, k=10, n_neg=99)

        self._final_summary()
        self.logger.close()

    # ── Final summary ─────────────────────────────────────────────────────────

    def _final_summary(self):
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"\n-- Training metrics (batch-level) --")
        print(f"{'Epoch':>6} | {'Loss':>8} | {'HR@10(train)':>12} | {'NDCG@10(train)':>14}")
        print("-" * 48)
        for m in self.metrics_log:
            print(f"{m['epoch']:>6} | {m['loss']:>8.4f} | "
                  f"{m['hit_ratio@10']:>12.4f} | {m['ndcg@10']:>14.4f}")

        if self.eval_log:
            print(f"\n-- SCoLR leave-one-out evaluation (1 pos + 99 neg) --")
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
            rounds = [max(t["total_time_s"] for t in tl["timings"])
                      for tl in self.timing_log]
            coverages = [tl["coverage_pct"] for tl in self.timing_log]
            print(f"\nTotal rounds      : {len(self.timing_log)}")
            print(f"Total round time  : {sum(rounds):.2f}s  ({sum(rounds)/60:.2f} min)")
            print(f"Avg time / round  : {sum(rounds)/len(self.timing_log):.4f}s")
            print(f"Avg item coverage : {np.mean(coverages):.1f}%  "
                  f"(p={self.subsample_rate:.0%} × {self.num_clients} clients)")

        print(f"\nLog saved → {self.log_path}")
        print(f"{'='*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {DEVICE}")

    dataloader   = MovielensDatasetLoader()
    all_ratings  = dataloader.ratings          # (6040, 3706)
    n_clients    = 604
    train_matrix = all_ratings[:n_clients]

    fncf = FederatedNCF(
        train_matrix       = train_matrix,
        num_clients        = n_clients,
        aggregation_epochs = 50,
        local_epochs       = 2,
        batch_size         = 256,
        latent_dim         = 64,
        rank               = 16,
        subsample_rate     = 0.1,   # ← SCoLR: 10% of items per client upload
        lr                 = 5e-4,
        seed               = 42,
        device             = DEVICE,
        eval_fraction      = 0.2,
        eval_every         = 5,
    )
    fncf.train()