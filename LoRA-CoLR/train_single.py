import torch
import torch.nn.functional as F
import numpy as np
from .model import NeuralCollaborativeFiltering
from metrics import compute_metrics, evaluate_user_standard


def leave_one_out_split(ui_matrix: np.ndarray, seed: int = 42):
    rng          = np.random.default_rng(seed)
    train_matrix = ui_matrix.copy().astype(np.float32)
    test_items   = {}
    for u in range(ui_matrix.shape[0]):
        positives = np.where(ui_matrix[u] > 0)[0]
        if len(positives) < 2:
            continue
        test_item                  = int(rng.choice(positives))
        train_matrix[u, test_item] = 0.0
        test_items[u]              = test_item
    return train_matrix, test_items


class MatrixLoader:
    def __init__(self, ui_matrix: np.ndarray, seed: int = 0):
        np.random.seed(seed)
        self.ui_matrix = ui_matrix
        self._reset()

    def _reset(self):
        self.positives = np.argwhere(self.ui_matrix != 0)
        self.negatives = np.argwhere(self.ui_matrix == 0)

    def get_batch(self, batch_size: int):
        n_pos = batch_size // 4
        n_neg = batch_size - n_pos
        if self.positives.shape[0] < n_pos or self.negatives.shape[0] < n_neg:
            return None, None

        pos_idx = np.random.choice(self.positives.shape[0], n_pos, replace=False)
        neg_idx = np.random.choice(self.negatives.shape[0], n_neg, replace=False)
        pos     = self.positives[pos_idx]
        neg     = self.negatives[neg_idx]

        self.positives = np.delete(self.positives, pos_idx, axis=0)
        self.negatives = np.delete(self.negatives, neg_idx, axis=0)

        batch = np.concatenate([pos, neg], axis=0)
        np.random.shuffle(batch)
        y = np.array([self.ui_matrix[i, j] for i, j in batch], dtype=np.float32)
        return torch.tensor(batch, dtype=torch.int), torch.tensor(y)


class NCFTrainer:
    def __init__(self, ui_matrix: np.ndarray, epochs: int, batch_size: int,
                 latent_dim: int = 32, lora_rank: int = 8, device=None,
                 global_user_offset: int = 0, eval_seed: int = 42):
        self.full_ui_matrix     = ui_matrix
        self.train_matrix, self.test_items = leave_one_out_split(ui_matrix, seed=eval_seed)
        self.ui_matrix          = self.train_matrix
        self.epochs             = epochs
        self.batch_size         = batch_size
        self.latent_dim         = latent_dim
        self.lora_rank          = lora_rank
        self.global_user_offset = global_user_offset
        self.all_item_ids       = np.arange(ui_matrix.shape[1])
        self.device             = device or torch.device(
                                      "cuda" if torch.cuda.is_available() else "cpu")
        self.ncf = NeuralCollaborativeFiltering(
            ui_matrix.shape[0], ui_matrix.shape[1], latent_dim, lora_rank=lora_rank
        ).to(self.device)
        self.loader = MatrixLoader(self.ui_matrix)

    def _reset_loader(self):
        self.loader = MatrixLoader(self.ui_matrix)

    def train_batch(self, x: torch.Tensor, y: torch.Tensor,
                    optimizer: torch.optim.Optimizer):
        pos_mask = (y > 0)
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return 0.0, torch.zeros(y.shape[0])

        x_all  = torch.cat([x[pos_mask], x[neg_mask]], dim=0).to(self.device)
        scores = self.ncf(x_all).squeeze(-1)

        n_pos      = pos_mask.sum().item()
        pos_scores = scores[:n_pos]
        neg_scores = scores[n_pos:]

        diff     = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        bpr_loss = -F.logsigmoid(diff).mean()

        bpr_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        all_scores                           = torch.zeros(y.shape[0], device=self.device)
        all_scores[pos_mask.to(self.device)] = pos_scores.detach()
        all_scores[neg_mask.to(self.device)] = neg_scores.detach()
        return bpr_loss.item(), torch.sigmoid(all_scores).cpu()

    def train_model(self, optimizer: torch.optim.Optimizer, epochs: int = None):
        epochs   = epochs or self.epochs
        progress = {"epoch": [], "loss": [], "hit_ratio@10": [], "ndcg@10": []}

        for epoch in range(epochs):
            self._reset_loader()
            epoch_loss, epoch_hr, epoch_ndcg, steps = 0.0, 0.0, 0.0, 0

            while True:
                x, y = self.loader.get_batch(self.batch_size)
                if x is None:
                    break
                x, y     = x.to(self.device), y.to(self.device)
                loss, y_ = self.train_batch(x, y, optimizer)
                hr, ng   = compute_metrics(y.cpu().numpy(), y_.numpy())
                epoch_loss += loss; epoch_hr += hr; epoch_ndcg += ng; steps += 1

            if steps > 0:
                progress["epoch"].append(epoch)
                progress["loss"].append(epoch_loss / steps)
                progress["hit_ratio@10"].append(epoch_hr / steps)
                progress["ndcg@10"].append(epoch_ndcg / steps)

        last    = {k: v[-1] if v else 0.0 for k, v in progress.items() if k != "epoch"}
        results = {"num_users": self.ui_matrix.shape[0], **last}
        return results, progress

    def evaluate_standard(self, n_neg: int = 99, k: int = 10) -> dict:
        if not self.test_items:
            return {f"hr@{k}": 0.0, f"ndcg@{k}": 0.0,
                    "eval_loss": 0.0, "evaluated_users": 0}

        hrs, ndcgs, losses = [], [], []
        for local_u, test_item in self.test_items.items():
            interacted = set(np.where(self.full_ui_matrix[local_u] > 0)[0])
            hr, ng, loss = evaluate_user_standard(
                model        = self.ncf,
                user_id      = local_u,
                test_item    = test_item,
                all_item_ids = self.all_item_ids,
                interacted   = interacted,
                device       = self.device,
                n_neg        = n_neg,
                k            = k,
            )
            hrs.append(hr); ndcgs.append(ng); losses.append(loss)

        return {
            f"hr@{k}":         float(np.mean(hrs)),
            f"ndcg@{k}":       float(np.mean(ndcgs)),
            "eval_loss":       float(np.mean(losses)),
            "evaluated_users": len(hrs),
        }

    def train(self, optimizer: torch.optim.Optimizer, return_progress: bool = False):
        self.ncf.join_output_weights()
        results, progress = self.train_model(optimizer)
        return (results, progress) if return_progress else results