import os
import re
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_figure")

# ──────────────────────────────────────────────
# Parse log file — standard eval table
# ──────────────────────────────────────────────

def parse_standard_eval(log_path: str):
    """
    Parses:
     Epoch |    HR@10 |    NDCG@10 |   EvalLoss |   Users
    --------------------------------------------------------
         0 |   0.2615 |     0.1325 |     0.4821 |     260
    """
    metrics  = []
    in_table = False

    # matches rows with 5 pipe-separated numeric columns
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*$"
    )

    with open(log_path, 'r') as f:
        for line in f:
            if "Standard leave-one-out evaluation" in line:
                in_table = True
                continue
            if in_table:
                m = pattern.match(line)
                if m:
                    metrics.append({
                        "epoch":     int(m.group(1)),
                        "hr@10":     float(m.group(2)),
                        "ndcg@10":   float(m.group(3)),
                        "eval_loss": float(m.group(4)),
                        "users":     int(m.group(5)),
                    })
                elif metrics and line.strip() == "":
                    break

    return sorted(metrics, key=lambda x: x["epoch"])


def parse_training_metrics(log_path: str):
    """
    Parses the training batch metrics table:

     Epoch |     Loss | HR@10(train) | NDCG@10(train)
    """
    metrics = []
    in_table = False

    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*$"
    )

    with open(log_path, 'r') as f:
        for line in f:
            if "Training metrics (batch-level" in line:
                in_table = True
                continue
            if in_table:
                m = pattern.match(line)
                if m:
                    metrics.append({
                        "epoch":  int(m.group(1)),
                        "loss":   float(m.group(2)),
                    })
                elif metrics and line.strip() == "":
                    break

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
    std_metrics  = parse_standard_eval(log_path)

    if not std_metrics:
        print(f"  [WARN] No standard eval metrics found in {log_name}")
        return

    n_users   = std_metrics[0]["users"]
    epochs    = [r["epoch"]     for r in std_metrics]
    hr_vals   = [r["hr@10"]     for r in std_metrics]
    ndcg_vals = [r["ndcg@10"]   for r in std_metrics]
    loss_vals = [r["eval_loss"] for r in std_metrics]

    print(f"  Parsed {len(std_metrics)} epochs ({n_users} users) from {log_name}")

    # ── Figure 1: Eval Loss ──────────────────────────────────────────────────
    _save_figure(
        x      = epochs,
        y      = loss_vals,
        xlabel = "Communication Round",
        ylabel = "BCE Loss (eval set)",
        title  = f"Standard Eval Loss  (1 pos + 99 neg, {n_users} users)",
        color  = "#e74c3c",
        path   = os.path.join(RESULT_DIR, f"{base_name}_eval_loss.png"),
        lower_is_better = True,
    )

    # ── Figure 2: HR@10 ──────────────────────────────────────────────────────
    _save_figure(
        x      = epochs,
        y      = hr_vals,
        xlabel = "Communication Round",
        ylabel = "HR@10",
        title  = f"Hit Ratio@10  (1 pos + 99 neg, {n_users} users)",
        color  = "#2ecc71",
        path   = os.path.join(RESULT_DIR, f"{base_name}_standard_hr10.png"),
    )

    # ── Figure 3: NDCG@10 ────────────────────────────────────────────────────
    _save_figure(
        x      = epochs,
        y      = ndcg_vals,
        xlabel = "Communication Round",
        ylabel = "NDCG@10",
        title  = f"NDCG@10  (1 pos + 99 neg, {n_users} users)",
        color  = "#3498db",
        path   = os.path.join(RESULT_DIR, f"{base_name}_standard_ndcg10.png"),
    )

    # ── Figure 4: HR@10 + NDCG@10 combined ──────────────────────────────────
    _save_combined_figure(
        x       = epochs,
        y1      = hr_vals,
        y2      = ndcg_vals,
        label1  = "HR@10",
        label2  = "NDCG@10",
        color1  = "#2ecc71",
        color2  = "#3498db",
        xlabel  = "Communication Round",
        ylabel  = "Metric Value",
        title   = f"Standard Evaluation  (1 pos + 99 neg, {n_users} users)",
        path    = os.path.join(RESULT_DIR, f"{base_name}_standard_combined.png"),
    )

    # ── Figure 5: Loss + HR@10 dual-axis ────────────────────────────────────
    _save_dual_axis_figure(
        x       = epochs,
        y_loss  = loss_vals,
        y_hr    = hr_vals,
        n_users = n_users,
        path    = os.path.join(RESULT_DIR, f"{base_name}_loss_vs_hr.png"),
    )


def _save_figure(x, y, xlabel, ylabel, title, color, path, lower_is_better=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, color=color, linewidth=2.0, marker='o', markersize=3)
    ax.fill_between(x, y, alpha=0.08, color=color)

    best_val = min(y) if lower_is_better else max(y)
    best_idx = y.index(best_val)
    ax.annotate(f"best: {best_val:.4f}",
                xy=(x[best_idx], best_val),
                xytext=(10, 10), textcoords="offset points",
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.annotate(f"final: {y[-1]:.4f}",
                xy=(x[-1], y[-1]),
                xytext=(-70, -18), textcoords="offset points",
                fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

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


def _save_combined_figure(x, y1, y2, label1, label2, color1, color2,
                           xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y1, color=color1, linewidth=2.0, marker='o', markersize=3, label=label1)
    ax.plot(x, y2, color=color2, linewidth=2.0, marker='s', markersize=3, label=label2)
    ax.fill_between(x, y1, alpha=0.06, color=color1)
    ax.fill_between(x, y2, alpha=0.06, color=color2)
    ax.annotate(f"{y1[-1]:.4f}", xy=(x[-1], y1[-1]),
                xytext=(6,  4), textcoords="offset points", fontsize=9, color=color1)
    ax.annotate(f"{y2[-1]:.4f}", xy=(x[-1], y2[-1]),
                xytext=(6, -12), textcoords="offset points", fontsize=9, color=color2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


def _save_dual_axis_figure(x, y_loss, y_hr, n_users, path):
    """Eval loss (left axis, red) vs HR@10 (right axis, green) on same plot."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(x, y_loss, color="#e74c3c", linewidth=2.0, marker='o', markersize=3, label="Eval Loss")
    ax1.set_xlabel("Communication Round", fontsize=12)
    ax1.set_ylabel("BCE Loss (eval set)", fontsize=12, color="#e74c3c")
    ax1.tick_params(axis='y', labelcolor="#e74c3c")
    ax1.set_xlim(left=0)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, y_hr, color="#2ecc71", linewidth=2.0, marker='s', markersize=3, label="HR@10")
    ax2.set_ylabel("HR@10", fontsize=12, color="#2ecc71")
    ax2.tick_params(axis='y', labelcolor="#2ecc71")
    ax2.set_ylim(bottom=0)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center right")

    fig.suptitle(f"Eval Loss vs HR@10  ({n_users} users, 1 pos + 99 neg)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    log_name = "FedNCF-20260422-221203"
    plot_single_log(log_name)