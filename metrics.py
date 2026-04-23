import numpy as np
import torch
import torch.nn.functional as F


# ── Batch-level metrics (training monitoring only) ────────────────────────────

def hit_ratio(y: np.ndarray, pred: np.ndarray, N: int = 10) -> float:
    """Batch HR@N: checks if best-rated item appears in top-N predicted."""
    pos_mask          = (y > 0)
    pred_pos          = pred * pos_mask
    best_idx          = int(np.argmax(y))
    top_n_idx         = np.argsort(pred_pos)[::-1][:N]
    return 1.0 if best_idx in top_n_idx else 0.0

def ndcg(y: np.ndarray, pred: np.ndarray, N: int = 10) -> float:
    """Batch NDCG@N."""
    top_idx   = np.argsort(y)[::-1][:N]
    actual    = y[top_idx]
    predicted = np.clip(np.round(pred[top_idx]), 0, None)
    denom     = np.log2(np.arange(2, N + 2))
    dcg       = np.sum((2 ** predicted - 1) / denom)
    idcg      = np.sum((2 ** actual    - 1) / denom)
    return float(dcg / idcg) if idcg > 0 else 0.0

def compute_metrics(y: np.ndarray, pred: np.ndarray) -> list:
    return [hit_ratio(y, pred), ndcg(y, pred)]


# ── Standard leave-one-out evaluation (paper protocol) ───────────────────────

def rank_of_test_item(test_score: float, neg_scores: np.ndarray) -> int:
    """0-based rank of test item; rank=0 means ranked 1st."""
    return int(np.sum(neg_scores >= test_score))

def standard_hr_at_k(rank: int, k: int = 10) -> float:
    return 1.0 if rank < k else 0.0

def standard_ndcg_at_k(rank: int, k: int = 10) -> float:
    return (1.0 / np.log2(rank + 2)) if rank < k else 0.0

def evaluate_user_standard(model: torch.nn.Module,
                            user_id: int,
                            test_item: int,
                            all_item_ids: np.ndarray,
                            interacted: set,
                            device: torch.device,
                            n_neg: int = 99,
                            k: int = 10) -> tuple:
    """
    Standard leave-one-out evaluation (He et al. NeurIPS 2017).

    Samples exactly n_neg negatives deterministically from (user_id, test_item)
    so results are identical across every call — epoch-to-epoch comparison valid.

    Returns
    -------
    hr, ndcg, bce_loss  (float, float, float)
    """
    # Deterministic negative sampling — same 99 negatives every epoch
    rng        = np.random.default_rng(seed=hash((int(user_id), int(test_item))) & 0xFFFFFFFF)
    candidates = all_item_ids[~np.isin(all_item_ids, list(interacted))]
    neg_items  = rng.choice(candidates, size=min(n_neg, len(candidates)), replace=False)

    eval_items = np.concatenate([[test_item], neg_items])           # (100,)
    users      = np.full(len(eval_items), user_id, dtype=np.int32)
    labels     = np.zeros(len(eval_items), dtype=np.float32)
    labels[0]  = 1.0

    x = torch.tensor(np.stack([users, eval_items], axis=1), dtype=torch.int).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(-1)                               # (100,)
        loss   = F.binary_cross_entropy_with_logits(logits, y).item()
        scores = torch.sigmoid(logits).cpu().numpy().flatten()
    model.train()

    rank = rank_of_test_item(scores[0], scores[1:])
    return standard_hr_at_k(rank, k), standard_ndcg_at_k(rank, k), loss