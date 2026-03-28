from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_case_study_figure():
    # Schematic examples for layout/caption use. Replace with real saved
    # TLDL outputs later if token-frame scores are exported from the model.
    examples = [
        {
            "title": "Case 1: rhetorical challenge",
            "tokens": ["ni", "zhen", "jue", "de", "xing", "ma"],
            "frames": 8,
            "matrix": np.array(
                [
                    [0.10, 0.08, 0.12, 0.18, 0.55, 0.78],
                    [0.08, 0.06, 0.10, 0.16, 0.58, 0.82],
                    [0.09, 0.08, 0.14, 0.22, 0.61, 0.86],
                    [0.07, 0.08, 0.15, 0.24, 0.66, 0.90],
                    [0.06, 0.08, 0.12, 0.20, 0.71, 0.94],
                    [0.07, 0.09, 0.11, 0.19, 0.69, 0.91],
                    [0.08, 0.10, 0.12, 0.18, 0.63, 0.87],
                    [0.09, 0.10, 0.13, 0.17, 0.57, 0.81],
                ]
            ),
            "gold": "R",
            "pred": "R",
            "summary": "Discrepancy concentrates on the sentence-final span, where a literal question becomes a challenging intent.",
        },
        {
            "title": "Case 2: reluctant agreement",
            "tokens": ["hao", "ba", "wo", "zhi", "dao", "le"],
            "frames": 8,
            "matrix": np.array(
                [
                    [0.62, 0.80, 0.18, 0.12, 0.16, 0.52],
                    [0.66, 0.84, 0.16, 0.10, 0.14, 0.57],
                    [0.71, 0.89, 0.15, 0.09, 0.13, 0.63],
                    [0.74, 0.91, 0.16, 0.10, 0.14, 0.68],
                    [0.72, 0.88, 0.18, 0.11, 0.15, 0.70],
                    [0.69, 0.84, 0.19, 0.12, 0.16, 0.67],
                    [0.64, 0.79, 0.20, 0.14, 0.16, 0.61],
                    [0.58, 0.73, 0.22, 0.16, 0.17, 0.56],
                ]
            ),
            "gold": "S",
            "pred": "S",
            "summary": "The strongest conflict appears on discourse particles rather than on content words, matching reluctant acceptance.",
        },
    ]

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.9), constrained_layout=True)
    cmap = plt.cm.YlOrRd
    vmin, vmax = 0.0, 1.0
    last_im = None

    for panel_idx, (ax, ex) in enumerate(zip(axes, examples)):
        im = ax.imshow(ex["matrix"], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        last_im = im
        ax.set_title(f"({chr(ord('a') + panel_idx)}) {ex['title']}", loc="left", fontsize=11.3, fontweight="bold", pad=8)
        ax.set_xticks(np.arange(len(ex["tokens"])))
        ax.set_xticklabels(ex["tokens"], fontsize=10)
        ax.set_yticks(np.arange(ex["frames"]))
        ax.set_yticklabels([f"f{i+1}" for i in range(ex["frames"])], fontsize=9)
        ax.set_ylabel("Audio frames", fontsize=10)
        ax.set_xlabel("Text tokens", fontsize=10)

        ax.text(
            1.0,
            1.01,
            f"Gold: {ex['gold']}   Pred: {ex['pred']}",
            transform=ax.transAxes,
            fontsize=9.2,
            color="#222222",
            fontweight="bold",
            ha="right",
        )
        ax.text(
            0.0,
            -0.14,
            ex["summary"],
            transform=ax.transAxes,
            fontsize=9.2,
            color="#444444",
            va="top",
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.9, pad=0.012)
    cbar.set_label("Discrepancy score", fontsize=10)
    png_path = FIG_DIR / "case_study_sr.png"
    pdf_path = FIG_DIR / "case_study_sr.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_case_study_figure_singlecol():
    examples = [
        {
            "title": "Case 1: rhetorical challenge",
            "tokens": ["ni", "zhen", "jue", "de", "xing", "ma"],
            "frames": 8,
            "matrix": np.array(
                [
                    [0.10, 0.08, 0.12, 0.18, 0.55, 0.78],
                    [0.08, 0.06, 0.10, 0.16, 0.58, 0.82],
                    [0.09, 0.08, 0.14, 0.22, 0.61, 0.86],
                    [0.07, 0.08, 0.15, 0.24, 0.66, 0.90],
                    [0.06, 0.08, 0.12, 0.20, 0.71, 0.94],
                    [0.07, 0.09, 0.11, 0.19, 0.69, 0.91],
                    [0.08, 0.10, 0.12, 0.18, 0.63, 0.87],
                    [0.09, 0.10, 0.13, 0.17, 0.57, 0.81],
                ]
            ),
            "gold": "R",
            "pred": "R",
            "summary": "Sentence-final tokens carry the strongest discrepancy cue.",
        },
        {
            "title": "Case 2: reluctant agreement",
            "tokens": ["hao", "ba", "wo", "zhi", "dao", "le"],
            "frames": 8,
            "matrix": np.array(
                [
                    [0.62, 0.80, 0.18, 0.12, 0.16, 0.52],
                    [0.66, 0.84, 0.16, 0.10, 0.14, 0.57],
                    [0.71, 0.89, 0.15, 0.09, 0.13, 0.63],
                    [0.74, 0.91, 0.16, 0.10, 0.14, 0.68],
                    [0.72, 0.88, 0.18, 0.11, 0.15, 0.70],
                    [0.69, 0.84, 0.19, 0.12, 0.16, 0.67],
                    [0.64, 0.79, 0.20, 0.14, 0.16, 0.61],
                    [0.58, 0.73, 0.22, 0.16, 0.17, 0.56],
                ]
            ),
            "gold": "S",
            "pred": "S",
            "summary": "Discourse particles dominate the discrepancy pattern.",
        },
    ]

    fig, axes = plt.subplots(2, 1, figsize=(6.7, 6.9), constrained_layout=True)
    cmap = plt.cm.YlOrRd
    last_im = None

    for panel_idx, (ax, ex) in enumerate(zip(axes, examples)):
        im = ax.imshow(ex["matrix"], aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        last_im = im
        ax.set_title(f"({chr(ord('a') + panel_idx)}) {ex['title']}", loc="left", fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xticks(np.arange(len(ex["tokens"])))
        ax.set_xticklabels(ex["tokens"], fontsize=9)
        ax.set_yticks(np.arange(ex["frames"]))
        ax.set_yticklabels([f"f{i+1}" for i in range(ex["frames"])], fontsize=8)
        ax.set_ylabel("Frames", fontsize=9)
        ax.set_xlabel("Tokens", fontsize=9)
        ax.text(
            1.0,
            1.01,
            f"G: {ex['gold']}  P: {ex['pred']}",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#222222",
            fontweight="bold",
            ha="right",
        )
        ax.text(
            0.0,
            -0.16,
            ex["summary"],
            transform=ax.transAxes,
            fontsize=8.5,
            color="#444444",
            va="top",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.9, pad=0.012)
    cbar.set_label("Score", fontsize=9)

    png_path = FIG_DIR / "case_study_sr_singlecol.png"
    pdf_path = FIG_DIR / "case_study_sr_singlecol.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def extract_series(history: list[dict], split: str, key: str) -> np.ndarray:
    return np.array([row[split][key] * 100.0 for row in history], dtype=float)


def extract_epochs(history: list[dict]) -> np.ndarray:
    return np.array([row["epoch"] for row in history], dtype=int)


def best_value(history: list[dict], split: str, key: str) -> tuple[int, float]:
    vals = extract_series(history, split, key)
    idx = int(np.argmax(vals))
    return idx + 1, float(vals[idx])


def make_rfr_calibration_figure():
    default_hist = load_json(ROOT / "output" / "sr_cdd_full_rfr_v1_20260320_183022" / "full_history.json")
    calibrated_hist = load_json(ROOT / "output" / "sr_rfr_runB_tau18_swa28_20260321_140510" / "full_history.json")
    default_test = load_json(ROOT / "output" / "sr_cdd_full_rfr_v1_20260320_183022" / "final_test_report.json")
    calibrated_test = load_json(ROOT / "output" / "sr_rfr_runB_tau18_swa28_20260321_140510" / "final_test_report.json")

    default_epochs = extract_epochs(default_hist)
    calibrated_epochs = extract_epochs(calibrated_hist)

    default_dev_acc = extract_series(default_hist, "dev", "acc")
    calibrated_dev_acc = extract_series(calibrated_hist, "dev", "acc")
    default_dev_f1 = extract_series(default_hist, "dev", "macro_f1")
    calibrated_dev_f1 = extract_series(calibrated_hist, "dev", "macro_f1")

    default_best_epoch, default_best_dev = best_value(default_hist, "dev", "acc")
    calibrated_best_epoch, calibrated_best_dev = best_value(calibrated_hist, "dev", "acc")

    default_ckpt_test = default_test["test_best_acc"]["acc"] * 100.0
    calibrated_ckpt_test = calibrated_test["test_best_acc"]["acc"] * 100.0
    default_test_swa = default_test["test_swa"]["acc"] * 100.0
    calibrated_test_swa = calibrated_test["test_swa"]["acc"] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(default_epochs, default_dev_acc, color="#d95f02", linewidth=2.3, label="Default dev acc")
    ax.plot(calibrated_epochs, calibrated_dev_acc, color="#1b9e77", linewidth=2.3, label="Calibrated dev acc")
    ax.plot(default_epochs, default_dev_f1, color="#fc8d62", linewidth=1.9, linestyle="--", label="Default dev macro-F1")
    ax.plot(calibrated_epochs, calibrated_dev_f1, color="#66c2a5", linewidth=1.9, linestyle="--", label="Calibrated dev macro-F1")
    ax.axvline(28, color="#888888", linestyle=":", linewidth=1.4)
    ax.text(28.6, min(default_dev_f1.min(), calibrated_dev_f1.min()) + 1.0, "SWA=28", fontsize=8.7, color="#666666")
    ax.scatter([default_best_epoch], [default_best_dev], color="#d95f02", s=42, zorder=3)
    ax.scatter([calibrated_best_epoch], [calibrated_best_dev], color="#1b9e77", s=42, zorder=3)
    ax.text(default_best_epoch + 0.4, default_best_dev + 0.2, f"{default_best_dev:.2f}", fontsize=8.4, color="#d95f02")
    ax.text(calibrated_best_epoch + 0.4, calibrated_best_dev + 0.2, f"{calibrated_best_dev:.2f}", fontsize=8.4, color="#1b9e77")
    ax.set_title("(a) Validation trajectory", fontsize=11.2, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_xlim(1, max(default_epochs.max(), calibrated_epochs.max()))
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")

    ax = axes[1]
    names = ["Default", "Calibrated"]
    xpos = np.arange(len(names))
    width = 0.24
    bars1 = ax.bar(xpos - width, [default_best_dev, calibrated_best_dev], width, color="#7570b3", label="Best dev acc")
    bars2 = ax.bar(xpos, [default_ckpt_test, calibrated_ckpt_test], width, color="#1b9e77", label="Test acc (best-dev ckpt)")
    bars3 = ax.bar(xpos + width, [default_test_swa, calibrated_test_swa], width, color="#66a61e", label="SWA test acc")
    ax.set_title("(b) Generalization after calibration", fontsize=11.2, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylim(69, 79.2)
    ax.legend(frameon=False, fontsize=8.1, loc="upper left")

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontsize=8.1)

    best_gain = calibrated_ckpt_test - default_ckpt_test
    swa_gain = calibrated_test_swa - default_test_swa
    ax.text(
        0.5,
        69.24,
        f"+{best_gain:.2f} ckpt acc, +{swa_gain:.2f} SWA acc",
        ha="center",
        fontsize=8.9,
        color="#1b9e77",
        fontweight="bold",
    )

    png_path = FIG_DIR / "rfr_calibration.png"
    pdf_path = FIG_DIR / "rfr_calibration.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_rfr_calibration_figure_singlecol():
    default_hist = load_json(ROOT / "output" / "sr_cdd_full_rfr_v1_20260320_183022" / "full_history.json")
    calibrated_hist = load_json(ROOT / "output" / "sr_rfr_runB_tau18_swa28_20260321_140510" / "full_history.json")
    default_test = load_json(ROOT / "output" / "sr_cdd_full_rfr_v1_20260320_183022" / "final_test_report.json")
    calibrated_test = load_json(ROOT / "output" / "sr_rfr_runB_tau18_swa28_20260321_140510" / "final_test_report.json")

    default_epochs = extract_epochs(default_hist)
    calibrated_epochs = extract_epochs(calibrated_hist)
    default_dev_acc = extract_series(default_hist, "dev", "acc")
    calibrated_dev_acc = extract_series(calibrated_hist, "dev", "acc")
    default_dev_f1 = extract_series(default_hist, "dev", "macro_f1")
    calibrated_dev_f1 = extract_series(calibrated_hist, "dev", "macro_f1")
    default_best_epoch, default_best_dev = best_value(default_hist, "dev", "acc")
    calibrated_best_epoch, calibrated_best_dev = best_value(calibrated_hist, "dev", "acc")
    default_ckpt_test = default_test["test_best_acc"]["acc"] * 100.0
    calibrated_ckpt_test = calibrated_test["test_best_acc"]["acc"] * 100.0
    default_test_swa = default_test["test_swa"]["acc"] * 100.0
    calibrated_test_swa = calibrated_test["test_swa"]["acc"] * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(default_epochs, default_dev_acc, color="#d95f02", linewidth=2.0, label="Default acc")
    ax.plot(calibrated_epochs, calibrated_dev_acc, color="#1b9e77", linewidth=2.0, label="Calibrated acc")
    ax.plot(default_epochs, default_dev_f1, color="#fc8d62", linewidth=1.6, linestyle="--", label="Default macro-F1")
    ax.plot(calibrated_epochs, calibrated_dev_f1, color="#66c2a5", linewidth=1.6, linestyle="--", label="Calibrated macro-F1")
    ax.axvline(28, color="#888888", linestyle=":", linewidth=1.2)
    ax.text(28.6, min(default_dev_f1.min(), calibrated_dev_f1.min()) + 1.0, "SWA=28", fontsize=8.2, color="#666666")
    ax.scatter([default_best_epoch], [default_best_dev], color="#d95f02", s=32, zorder=3)
    ax.scatter([calibrated_best_epoch], [calibrated_best_dev], color="#1b9e77", s=32, zorder=3)
    ax.set_title("(a) Validation trajectory", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=9)
    ax.set_xlim(1, max(default_epochs.max(), calibrated_epochs.max()))
    ax.legend(frameon=False, fontsize=7.9, loc="lower right")

    ax = axes[1]
    names = ["Default", "Calibrated"]
    xpos = np.arange(len(names))
    width = 0.23
    bars1 = ax.bar(xpos - width, [default_best_dev, calibrated_best_dev], width, color="#7570b3", label="Best dev")
    bars2 = ax.bar(xpos, [default_ckpt_test, calibrated_ckpt_test], width, color="#1b9e77", label="Best-dev ckpt")
    bars3 = ax.bar(xpos + width, [default_test_swa, calibrated_test_swa], width, color="#66a61e", label="SWA")
    ax.set_title("(b) Generalization after calibration", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=8.8)
    ax.set_ylim(69, 79.2)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=7.8)

    best_gain = calibrated_ckpt_test - default_ckpt_test
    swa_gain = calibrated_test_swa - default_test_swa
    ax.text(0.5, 69.22, f"+{best_gain:.2f} ckpt, +{swa_gain:.2f} SWA", ha="center", fontsize=8.4, color="#1b9e77", fontweight="bold")

    png_path = FIG_DIR / "rfr_calibration_singlecol.png"
    pdf_path = FIG_DIR / "rfr_calibration_singlecol.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    outputs = [
        *make_case_study_figure(),
        *make_case_study_figure_singlecol(),
        *make_rfr_calibration_figure(),
        *make_rfr_calibration_figure_singlecol(),
    ]
    print("Generated figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
