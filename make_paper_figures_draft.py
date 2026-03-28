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


def make_figure2_tldl_case():
    examples = [
        {
            "title": "Case 1: Sarcastic praise",
            "tokens": ["ni", "zhen", "bang", "a"],
            "scores": [0.08, 0.16, 0.78, 0.92],
            "note": "Literal-positive words align poorly with a sharp, marked ending.",
        },
        {
            "title": "Case 2: Rhetorical question",
            "tokens": ["zhe", "yang", "ye", "xing", "ma"],
            "scores": [0.10, 0.12, 0.18, 0.73, 0.95],
            "note": "The sentence-final question particle concentrates the contradiction cue.",
        },
        {
            "title": "Case 3: Reluctant acceptance",
            "tokens": ["hao", "ba", "wo", "zhi", "dao", "le"],
            "scores": [0.67, 0.86, 0.20, 0.14, 0.22, 0.70],
            "note": "Discrepancy mass shifts to discourse particles rather than content words.",
        },
    ]

    fig, axes = plt.subplots(len(examples), 1, figsize=(10.5, 5.8), constrained_layout=True)
    cmap = plt.cm.YlOrRd

    for ax, ex in zip(axes, examples):
        arr = np.array([ex["scores"]], dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(ex["title"], loc="left", fontsize=12, fontweight="bold")
        ax.set_yticks([])
        ax.set_xticks(np.arange(len(ex["tokens"])))
        ax.set_xticklabels(ex["tokens"], fontsize=11)
        for idx, score in enumerate(ex["scores"]):
            txt_color = "black" if score < 0.58 else "white"
            ax.text(idx, 0, f"{score:.2f}", ha="center", va="center", color=txt_color, fontsize=10)
        ax.text(
            0.0,
            -0.85,
            ex["note"],
            transform=ax.transAxes,
            fontsize=10,
            color="#444444",
            va="top",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("Token discrepancy score", fontsize=11)
    fig.suptitle(
        "Figure 3. Illustrative TLDL case visualization",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.01,
        -0.01,
        "Draft figure: token scores are schematic placeholders for layout and caption development.",
        fontsize=9,
        color="#666666",
    )

    png_path = FIG_DIR / "figure3_tldl_case_draft.png"
    pdf_path = FIG_DIR / "figure3_tldl_case_draft.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def gaussian_mix(x: np.ndarray, mus: list[float], sigmas: list[float], weights: list[float]) -> np.ndarray:
    y = np.zeros_like(x)
    for mu, sigma, weight in zip(mus, sigmas, weights):
        y += weight * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return y


def best_dev(history: list[dict]) -> float:
    return max(item["dev"]["acc"] for item in history)


def make_figure3_gate_behavior():
    default_hist = load_json(ROOT / "output" / "sr_cdd_full_rfr_v1_20260320_183022" / "full_history.json")
    calibrated_hist = load_json(ROOT / "output" / "sr_rfr_runB_tau18_swa28_20260321_140510" / "full_history.json")
    default_test = load_json(ROOT / "output" / "sr_cdd_full_rfr_v1_20260320_183022" / "final_test_report.json")
    calibrated_test = load_json(ROOT / "output" / "sr_rfr_runB_tau18_swa28_20260321_140510" / "final_test_report.json")

    default_dev = best_dev(default_hist) * 100.0
    calibrated_dev = best_dev(calibrated_hist) * 100.0
    default_test_best = default_test["test_best_acc"]["acc"] * 100.0
    calibrated_test_best = calibrated_test["test_best_acc"]["acc"] * 100.0
    default_gap = default_dev - default_test_best
    calibrated_gap = calibrated_dev - calibrated_test_best

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    x = np.linspace(0.0, 1.0, 400)
    default_y = gaussian_mix(x, [0.16, 0.88], [0.08, 0.06], [0.44, 0.56])
    calibrated_y = gaussian_mix(x, [0.42, 0.63], [0.12, 0.11], [0.42, 0.58])

    ax = axes[0]
    ax.fill_between(x, default_y, color="#d95f02", alpha=0.30, label="Default RFR")
    ax.plot(x, default_y, color="#d95f02", linewidth=2.2)
    ax.fill_between(x, calibrated_y, color="#1b9e77", alpha=0.30, label="Calibrated RFR")
    ax.plot(x, calibrated_y, color="#1b9e77", linewidth=2.2)
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.set_title("Illustrative gate distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gate weight g  (1.0 -> prefer CDD, 0.0 -> prefer RFR)")
    ax.set_ylabel("Relative density")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=10, loc="upper center")
    ax.text(0.04, max(default_y.max(), calibrated_y.max()) * 0.93, "more raw-fusion reliance", fontsize=9, color="#555555")
    ax.text(0.63, max(default_y.max(), calibrated_y.max()) * 0.93, "more CDD reliance", fontsize=9, color="#555555")
    ax.text(
        0.02,
        -0.24,
        "Left panel is schematic because gate statistics were not saved in the run logs.",
        transform=ax.transAxes,
        fontsize=9,
        color="#666666",
    )

    ax = axes[1]
    names = ["Default RFR", "Calibrated RFR"]
    dev_scores = [default_dev, calibrated_dev]
    test_scores = [default_test_best, calibrated_test_best]
    xpos = np.arange(len(names))
    width = 0.34
    bars1 = ax.bar(xpos - width / 2, dev_scores, width, color="#7570b3", label="Best dev acc")
    bars2 = ax.bar(xpos + width / 2, test_scores, width, color="#66a61e", label="Best test acc")
    ax.set_title("Observed dev-test gap", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(68, 79.5)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    for i, gap in enumerate([default_gap, calibrated_gap]):
        ax.plot([xpos[i] - width / 2, xpos[i] + width / 2], [dev_scores[i] + 0.75, dev_scores[i] + 0.75], color="#444444")
        ax.text(xpos[i], dev_scores[i] + 1.0, f"gap = {gap:.2f}", ha="center", fontsize=10, color="#444444")

    improvement = calibrated_test_best - default_test_best
    ax.text(
        0.5,
        68.45,
        f"Calibrated RFR improves test accuracy by {improvement:.2f} points over default.",
        ha="center",
        fontsize=10,
        color="#1b9e77",
        fontweight="bold",
    )

    fig.suptitle(
        "Figure 3. Gate behavior analysis for RFR calibration",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    png_path = FIG_DIR / "figure3_gate_behavior_draft.png"
    pdf_path = FIG_DIR / "figure3_gate_behavior_draft.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    fig2 = make_figure2_tldl_case()
    fig3 = make_figure3_gate_behavior()
    print("Generated:")
    for path in [*fig2, *fig3]:
        print(path)


if __name__ == "__main__":
    main()
