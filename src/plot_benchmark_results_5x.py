import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns

# ------------------------
# CONFIG
# ------------------------
DATA_DIR = "../results"        # folder with Excel files
BENCHMARK = "stats"     # matches *_plot_*.xlsx
RUNS = 5               # run_1 ... run_5
OUT_DIR = "../results/plots"  # where to save plot
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"accuracy_{BENCHMARK}.pdf")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "font.size": 9,
})

# ------------------------
# LOAD FILES
# ------------------------
files = sorted(glob.glob(os.path.join(DATA_DIR, f"*_{BENCHMARK}_*.xlsx")))

if not files:
    raise RuntimeError("No matching Excel files found")

all_scores = []   # list of lists: [[run1..run5], ...]
labels = []       # model names

def extract_model_name(filename):
    name = os.path.splitext(filename)[0]
    if BENCHMARK in ["query","plot","stats"]:
        return name.split("results_")[-1]
    else:
        return name.split("classifier_")[-1]

# ------------------------
# EXTRACT ACCURACIES
# ------------------------
for path in files:
    model_name = extract_model_name(os.path.basename(path))
    run_scores = []

    for i in range(1, RUNS + 1):
        sheet = f"run_{i}"
        df = pd.read_excel(path, sheet_name=sheet)

        # accuracy = mean of score column (0/1)
        acc = df["score"].mean()
        run_scores.append(acc)

    all_scores.append(run_scores)
    labels.append(model_name)

# ------------------------
# PLOT: BOX + DOTS
# ------------------------
plt.figure(figsize=(9, 6))

dark_grey = "#444444"
light_grey = "#9e9e9e"

# boxplot with styling
bp = plt.boxplot(
    all_scores,
    labels=labels,
    showfliers=False,
    patch_artist=False,
    boxprops=dict(color=dark_grey),
    whiskerprops=dict(color=dark_grey),
    capprops=dict(color=dark_grey),
    medianprops=dict(color=dark_grey)
)

# overlay replicate dots (all grey)
for i, scores in enumerate(all_scores, start=1):
    x = np.random.normal(i, 0.04, size=len(scores))  # jitter
    plt.scatter(x, scores, color=light_grey, alpha=0.9, zorder=3)

    # mean annotation
    mean_val = np.mean(scores)

    # place label above the box (Q3)
    q3 = np.percentile(scores, 99)

    plt.text(
        i,
        q3 + 0.03,
        f"{mean_val:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=dark_grey,
        fontweight="bold"
    )

plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title(f"Accuracy across replicates ({BENCHMARK})", pad=20)
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Box plot saved to: {out_path}")

