import os
import re
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_figure")

# ──────────────────────────────────────────────
# Parse log file
# ──────────────────────────────────────────────

def parse_log(log_path: str):
    """
    Parses the summary table from the log file:

     Epoch |     Loss |    HR@10 |  NDCG@10
    ------------------------------------------
         0 |   2.0617 |   0.2576 |   0.0021
         1 |   1.7391 |   0.3047 |   0.0243
    """
    metrics = []

    # Matches:  "     0 |   2.0617 |   0.2576 |   0.0021"
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*$"
    )

    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                metrics.append({
                    "epoch":        int(m.group(1)),
                    "loss":         float(m.group(2)),
                    "hit_ratio@10": float(m.group(3)),
                    "ndcg@10":      float(m.group(4)),
                })

    return sorted(metrics, key=lambda x: x["epoch"])


# ──────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────

def plot_single_log(log_name: str):
    if not log_name.endswith(".txt"):
        log_name += ".txt"

    log_path = os.path.join(RESULT_DIR, log_name)
    if not os.path.exists(log_path):
        print(f"  [ERROR] File not found: {log_path}")
        return

    base_name    = os.path.splitext(log_name)[0]
    metrics_list = parse_log(log_path)

    if not metrics_list:
        print(f"  [WARN] No metrics found in {log_name}")
        return

    print(f"  Parsed {len(metrics_list)} epochs from {log_name}")
    epochs = [r["epoch"] for r in metrics_list]

    # ── Figure 1: Loss ──
    _save_figure(
        x      = epochs,
        y      = [r["loss"] for r in metrics_list],
        xlabel = "Number of Training Rounds",
        ylabel = "Loss",
        title  = "Training Loss",
        color  = "#e74c3c",
        path   = os.path.join(RESULT_DIR, f"{base_name}_loss.png"),
    )

    # ── Figure 2: HR@10 ──
    _save_figure(
        x      = epochs,
        y      = [r["hit_ratio@10"] for r in metrics_list],
        xlabel = "Number of Training Rounds",
        ylabel = "Hit Ratio@10",
        title  = "Hit Ratio@10",
        color  = "#2ecc71",
        path   = os.path.join(RESULT_DIR, f"{base_name}_hr10.png"),
    )

    # ── Figure 3: NDCG@10 ──
    _save_figure(
        x      = epochs,
        y      = [r["ndcg@10"] for r in metrics_list],
        xlabel = "Number of Training Rounds",
        ylabel = "NDCG@10",
        title  = "NDCG@10",
        color  = "#3498db",
        path   = os.path.join(RESULT_DIR, f"{base_name}_ndcg10.png"),
    )


def _save_figure(x, y, xlabel, ylabel, title, color, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, color=color, linewidth=1.8)
    ax.fill_between(x, y, alpha=0.08, color=color)
    ax.annotate(f"final: {y[-1]:.4f}",
                xy=(x[-1], y[-1]),
                xytext=(-60, 12), textcoords="offset points",
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────
# Entry point — change log_name here
# ──────────────────────────────────────────────

if __name__ == "__main__":
    log_name = "FedNCF-20260422-145144"   # ← change this
    plot_single_log(log_name)