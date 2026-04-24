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


# ── Communication cost helpers ────────────────────────────────────────────────

def _upload_bits(item_num: int, latent_dim: int, rank: int) -> int:
    """
    CoLR upload per client (Algorithm 1, line 16):
      - Full A_u (mlp + gmf):  item_num × rank × 2
      - MLP LowRankLinear factors + output heads
    B is NOT uploaded (server keeps its own B).
    """
    pf   = latent_dim
    bits = 0
    # Full item A (mlp + gmf) — NOT sparse (that is SCoLR's optimisation)
    bits += item_num * rank * 2 * 32
    # MLP 3 layers
    mlp_dims = [(4*pf, 2*pf), (2*pf, pf), (pf, pf//2)]
    for (in_f, out_f) in mlp_dims:
        bits += (out_f * rank + in_f * rank + out_f) * 32
    # output heads
    bits += (2*pf + pf//2 + pf + 1) * 32
    return bits


def _download_bits(item_num: int, latent_dim: int, rank: int) -> int:
    """
    CoLR download per client (Algorithm 1, lines 9-10):
      - Q dense (mlp + gmf):  item_num × 2*pf × 2
      - B (mlp + gmf):        2*pf × rank × 2
      - MLP + output heads
    """
    pf   = latent_dim
    bits = 0
    # Q dense (mlp + gmf)
    bits += item_num * 2 * pf * 2 * 32
    # B (mlp + gmf)
    bits += 2 * pf * rank * 2 * 32
    # MLP 3 layers
    mlp_dims = [(4*pf, 2*pf), (2*pf, pf), (pf, pf//2)]
    for (in_f, out_f) in mlp_dims:
        bits += (out_f * rank + in_f * rank + out_f) * 32
    # output heads
    bits += (2*pf + pf//2 + pf + 1) * 32
    return bits


# ── Federated NCF (Algorithm 1 faithful CoLR) ────────────────────────────────

class FederatedNCF:
    def __init__(self,
                 train_matrix:       np.ndarray,
                 num_clients:        int   = 604,
                 aggregation_epochs: int   = 50,
                 local_epochs:       int   = 1,
                 batch_size:         int   = 256,
                 latent_dim:         int   = 64,
                 rank:               int   = 16,
                 lr:                 float = 1e-4,
                 seed:               int   = 0,
                 device:             str   = None,
                 eval_fraction:      float = 1.0,
                 eval_every:         int   = 1):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu")

        self.num_clients        = num_clients
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs       = local_epochs
        self.batch_size         = batch_size
        self.latent_dim         = latent_dim
        self.rank               = rank
        self.lr                 = lr
        self.item_num           = train_matrix.shape[1]
        self.metrics_log        = []
        self.eval_log           = []
        self.timing_log         = []
        self.eval_every         = eval_every

        rng                  = np.random.default_rng(seed)
        n_eval               = max(1, int(num_clients * eval_fraction))
        self.eval_client_ids = rng.choice(num_clients, size=n_eval, replace=False).tolist()

        assert train_matrix.shape[0] == num_clients

        self.client_n_items = [int((train_matrix[i] > 0).sum())
                               for i in range(num_clients)]

        self.clients = [
            NCFTrainer(train_matrix[i:i+1], epochs=local_epochs,
                       batch_size=batch_size, latent_dim=latent_dim,
                       rank=rank, device=self.device, global_user_offset=i)
            for i in range(num_clients)
        ]
        self.optimizers = [
            torch.optim.Adam(c.ncf.parameters(), lr=lr) for c in self.clients
        ]
        self.bandwidth_profiles = assign_bandwidth(num_clients, seed=seed)

        # ── Comm cost report ──────────────────────────────────────────────────
        ul_bits        = _upload_bits(self.item_num, latent_dim, rank)
        dl_bits        = _download_bits(self.item_num, latent_dim, rank)
        # Full-rank baseline: item_num × embedding_dim × 2 (mlp + gmf)
        full_rank_bits = self.item_num * 2 * latent_dim * 2 * 32
        saving_ul      = 100.0 * (1 - ul_bits / full_rank_bits)

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
        print(f"Method        : CoLR (Algorithm 1 — B re-sampled every round)")
        print(f"Dataset       : ML-1M  |  Items: {self.item_num}")
        print(f"Users         : {num_clients}")
        print(f"Local epochs  : {local_epochs}")
        print(f"Batch size    : {batch_size}")
        print(f"Latent dim    : {latent_dim}")
        print(f"CoLR rank     : {rank}")
        print(f"Upload/client : {ul_bits/8/1024:.2f} KB  "
              f"(full A, saving {saving_ul:.1f}% vs full-rank)")
        print(f"Download/cl.  : {dl_bits/8/1024:.2f} KB  (dense Q + B + MLP)")
        print(f"Aggregation   : weighted FedAvg on A; merge Q = Q + B @ A.T")
        print(f"B sampling    : re-sampled from N(0, 1/sqrt(r)) EVERY round")
        print(f"Learning rate : {lr}")
        print(f"Eval fraction : {eval_fraction:.0%}  ({n_eval} / {num_clients} users)")
        print(f"Eval every    : every {eval_every} epoch(s)")
        print(f"{'='*60}")

    # ── Single training round ─────────────────────────────────────────────────

    def _single_round(self, epoch: int, server_model,
                      dl_bits: int, ul_bits: int) -> tuple:
        """
        CoLR per-round (Algorithm 1):
          For each client u:
            1. Receive Q(t) [dense] + B(t) + MLP  (Alg 1, lines 9-10)
            2. Reset A=0, freeze Q_base + B        (Alg 1, line 10-11)
            3. Train {A_u, p_u} locally            (Alg 1, lines 12-14)
            4. Upload full A_u                     (Alg 1, line 16)
        """
        timings       = []
        agg_results   = {"loss": [], "hit_ratio@10": [], "ndcg@10": []}
        client_mlp_As = []
        client_gmf_As = []
        client_shared = []

        bar = tqdm(enumerate(self.clients), total=self.num_clients,
                   desc=f"Epoch {epoch}")
        for cid, client in bar:
            bw      = BANDWIDTH_PROFILES[self.bandwidth_profiles[cid]]
            dl_time = comm_time(dl_bits, bw["download"])

            # ── Alg 1, lines 9-10: receive Q(t) + B(t), copy to client ───────
            client.ncf.to(self.device)
            client.ncf.load_server_weights(server_model)

            # ── Alg 1, lines 10-11: reset A=0, freeze Q_base + B ─────────────
            client.ncf.prepare_for_local_train()

            # ── Alg 1, lines 12-14: local training of {A_u, p_u} ─────────────
            t0         = time.time()
            results    = client.train(self.optimizers[cid])
            train_time = time.time() - t0

            # ── Collect full A_u + shared weights for upload ──────────────────
            a_dict = client.ncf.get_item_A()
            client_mlp_As.append(a_dict["mlp_A"])
            client_gmf_As.append(a_dict["gmf_A"])
            client_shared.append(client.ncf.get_shared_weights())

            # ── Unfreeze for next round's weight loading ──────────────────────
            client.ncf.unfreeze_all()

            ul_time = comm_time(ul_bits, bw["upload"])
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

            bar.set_postfix({"loss": f"{results['loss']:.4f}",
                             "HR@10": f"{results['hit_ratio@10']:.4f}"})
        bar.close()

        self.metrics_log.append({
            "epoch":        epoch,
            "loss":         round(float(np.mean(agg_results["loss"])),         6),
            "hit_ratio@10": round(float(np.mean(agg_results["hit_ratio@10"])), 6),
            "ndcg@10":      round(float(np.mean(agg_results["ndcg@10"])),      6),
        })
        return timings, client_mlp_As, client_gmf_As, client_shared

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
        print(f"\n[CoLR Eval — Epoch {epoch:>3d}]  "
              f"HR@{k} = {hr_mean:.4f}  |  NDCG@{k} = {ndcg_mean:.4f}  |  "
              f"Loss = {loss_mean:.4f}  ({total_n} / {self.num_clients} users)\n")
        return record

    # ── Timing summary ────────────────────────────────────────────────────────

    def _print_timing(self, epoch, timings, agg_time):
        by_bw = {"slow": [], "medium": [], "fast": []}
        for t in timings:
            by_bw[t["bandwidth"]].append(t)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Timing  [CoLR rank={self.rank}, B re-sampled]")
        print(f"{'='*60}")
        for name, ts in by_bw.items():
            if not ts:
                continue
            print(f"  [{name.upper():6s}] n={len(ts):3d} | "
                  f"dl={np.mean([t['download_time_s'] for t in ts]):.4f}s | "
                  f"train={np.mean([t['train_time_s'] for t in ts]):.4f}s | "
                  f"ul={np.mean([t['upload_time_s'] for t in ts]):.4f}s")
        bottleneck = max(t["total_time_s"] for t in timings)
        print(f"  [BOTTLENECK] {bottleneck:.4f}s  |  [AGG] {agg_time:.4f}s  |  "
              f"[TOTAL] {bottleneck + agg_time:.4f}s")
        print(f"{'='*60}")

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        server_model = ServerNeuralCollaborativeFiltering(
            item_num=self.item_num, predictive_factor=self.latent_dim,
            rank=self.rank).to(self.device)

        dl_bits = _download_bits(self.item_num, self.latent_dim, self.rank)
        ul_bits = _upload_bits(  self.item_num, self.latent_dim, self.rank)
        print(f"Download : {dl_bits/8/1024:.2f} KB  (dense Q + B + MLP)  |  "
              f"Upload : {ul_bits/8/1024:.2f} KB  (full A + MLP)\n")

        for epoch in range(self.aggregation_epochs):
            # ── 1. Sample B(t) ~ D_B  (Algorithm 1, line 3) ──────────────────
            #       Must happen BEFORE broadcasting to clients
            server_model.sample_new_B()

            # ── 2. Broadcast Q(t) + B(t) + MLP; local training; collect A ────
            timings, client_mlp_As, client_gmf_As, client_shared = \
                self._single_round(epoch, server_model, dl_bits, ul_bits)

            # ── 3. Aggregate A + merge Q (Algorithm 1, lines 18 + 7) ──────────
            #       Q(t+1) = Q(t) + B(t) @ A(t+1).T
            t0 = time.time()
            server_model.aggregate_and_merge(
                client_mlp_As  = client_mlp_As,
                client_gmf_As  = client_gmf_As,
                client_n_items = self.client_n_items,
                client_shared  = client_shared,
            )
            agg_time = time.time() - t0

            # ── 4. Logging ────────────────────────────────────────────────────
            self._print_timing(epoch, timings, agg_time)
            self.timing_log.append({"epoch": epoch,
                                    "timings": timings, "agg_time": agg_time})

            # ── 5. Evaluate ───────────────────────────────────────────────────
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
        print(f"\n-- Training metrics --")
        print(f"{'Epoch':>6} | {'Loss':>8} | {'HR@10':>8} | {'NDCG@10':>8}")
        print("-" * 40)
        for m in self.metrics_log:
            print(f"{m['epoch']:>6} | {m['loss']:>8.4f} | "
                  f"{m['hit_ratio@10']:>8.4f} | {m['ndcg@10']:>8.4f}")

        if self.eval_log:
            print(f"\n-- Leave-one-out evaluation --")
            print(f"{'Epoch':>6} | {'HR@10':>8} | {'NDCG@10':>8} | {'Loss':>8} | {'Users':>6}")
            print("-" * 48)
            for e in self.eval_log:
                print(f"{e['epoch']:>6} | {e['hr@10']:>8.4f} | "
                      f"{e['ndcg@10']:>8.4f} | {e['eval_loss']:>8.4f} | "
                      f"{e['evaluated_users']:>6}")
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
            print(f"\nAvg time/round : {np.mean(rounds):.4f}s")

        print(f"\nLog saved → {self.log_path}")
        print(f"{'='*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {DEVICE}")

    dataloader   = MovielensDatasetLoader()
    n_clients    = 604
    train_matrix = dataloader.ratings[:n_clients]

    fncf = FederatedNCF(
        train_matrix       = train_matrix,
        num_clients        = n_clients,
        aggregation_epochs = 50,
        local_epochs       = 2,
        batch_size         = 256,
        latent_dim         = 64,
        rank               = 16,
        lr                 = 5e-4,
        seed               = 42,
        device             = DEVICE,
        eval_fraction      = 0.2,
        eval_every         = 5,
    )
    fncf.train()