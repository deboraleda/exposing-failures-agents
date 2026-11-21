# rq3_heatmap.py
#
# Heatmap of agent behavior x judge rationale codes from the RQ3 manual
# analysis, one panel per agent. The panels share the x and y axes and are
# placed next to each other with no gaps.
#
# Usage (from the repository root):
#   uv run --with matplotlib --with numpy python rq3_analysis/rq3_heatmap.py

import csv
import os
import textwrap
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(HERE, "rq3-analysis.csv")
OUTPUT_BASE = os.path.join(HERE, "rq3_heatmap")

# failure codes first, then the codes where the tool calls are correct
BEHAVIOR_CODES = [
    "WRONG_TOOL_CALL",
    "EXTRA_TOOL_CALL",
    "WRONG_VALUE_PARAMETER",
    "PARAMETER_OVERLOAD",
    "CLARIFICATION_OR_DEFERRAL",
    "REPORTS_MISSING_INFORMATION",
    "CORRECT_TOOL_CALL",
]

# judge approves (FP) first, then judge rejects (FN)
RATIONALE_CODES = [
    "CORRECT_RESPONSE_HIDES_FAILURE",
    "ACCEPTS_FALSE_PREMISE",
    "ACCEPTS_CLARIFICATION_AS_ANSWER",
    "PENALIZES_ANSWER_WITHIN_AGENT_SCOPE",
    "PENALIZES_INCOMPLETE_ANSWER",
    "CONTRADICTS_TOOL_WITH_WORLD_KNOWLEDGE",
]
N_FP_RATIONALES = 3

GRID_COLS = 3


def main():
    rows = list(csv.DictReader(open(INPUT_FILE, encoding="utf-8-sig")))
    agents = sorted({r["agent"] for r in rows})

    counts = {
        agent: Counter(
            (r["agent_behavior_code"], r["judge_rationale_code"])
            for r in rows if r["agent"] == agent
        )
        for agent in agents
    }

    vmax = max(n for c in counts.values() for n in c.values())
    cmap = ListedColormap(["#FFFFFF"] + [plt.cm.Blues(x) for x in np.linspace(0.3, 0.95, vmax)])
    norm = BoundaryNorm(np.arange(-0.5, vmax + 1), cmap.N)

    n_rows = int(np.ceil(len(agents) / GRID_COLS))
    fig, axes = plt.subplots(
        n_rows, GRID_COLS, sharex=True, sharey=True,
        figsize=(11, 9.5), gridspec_kw={"wspace": 0, "hspace": 0},
    )

    for ax, agent in zip(axes.flat, agents):
        matrix = np.array([[counts[agent][(b, j)] for j in RATIONALE_CODES] for b in BEHAVIOR_CODES])
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        for y, x in zip(*np.nonzero(matrix)):
            ax.text(x, y, matrix[y, x], ha="center", va="center", fontsize=8,
                    color="white" if matrix[y, x] > vmax / 2 else "black")

        # FP | FN boundary and light cell grid
        ax.axvline(N_FP_RATIONALES - 0.5, color="#B45309", lw=1.2)
        ax.set_xticks(np.arange(len(RATIONALE_CODES) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(BEHAVIOR_CODES) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#E5E7EB", lw=0.5)
        ax.tick_params(which="minor", length=0)

        # agent name in the top-right block (a wrong tool call never pairs
        # with an FN rationale, so those cells are always empty)
        ax.text(N_FP_RATIONALES + (len(RATIONALE_CODES) - N_FP_RATIONALES - 1) / 2, -0.3,
                textwrap.fill(agent, 18), ha="center", va="top", fontsize=8.5,
                fontweight="bold", color="#111827", linespacing=1.1)
        ax.tick_params(labelsize=7)

        for spine in ax.spines.values():
            spine.set_color("#6B7280")

    for ax in axes.flat[len(agents):]:
        ax.axis("off")

    axes[0, 0].set_yticks(range(len(BEHAVIOR_CODES)))
    axes[0, 0].set_yticklabels(BEHAVIOR_CODES, fontsize=7)
    axes[0, 0].set_xticks(range(len(RATIONALE_CODES)))
    for ax in axes[-1]:
        ax.set_xticklabels(RATIONALE_CODES, rotation=60, ha="right", fontsize=7)

    # FP / FN headers above the top row
    for ax in axes[0]:
        for x, label in ((N_FP_RATIONALES - 1) / 2, "FP: judge approves"), \
                        ((N_FP_RATIONALES + len(RATIONALE_CODES) - 1) / 2, "FN: judge rejects"):
            ax.text(x, -0.75, label, ha="center", va="bottom", fontsize=7.5, color="#B45309", clip_on=False)

    cbar = fig.colorbar(im, ax=axes, ticks=range(vmax + 1), fraction=0.02, pad=0.01)
    cbar.set_label("Runs", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUTPUT_BASE}.{ext}", bbox_inches="tight", dpi=200)
        print(f"Saved → {OUTPUT_BASE}.{ext}")


if __name__ == "__main__":
    main()
