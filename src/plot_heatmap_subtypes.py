import glob
import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------------
# CONFIG
# ------------------------
DATA_DIR = "../results"        # folder with Excel files
USED_BENCHMARKS = ["plot", "stats", "query"]  # only these benchmarks
RUNS = 5                       # run_1 ... run_5
OUT_DIR = "../results/plots"
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "heatmap_subtypes.pdf")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "font.size": 9,
})

# canonical subtype names and desired column order
SUBTYPE_ORDER = ["invalid", "count", "filtering", "grouping/ranking", "stats", "plot"]

# ------------------------
# HELPERS
# ------------------------
def canonical_subtype(raw):
    """Map various eval_type strings into canonical subtypes."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    # map plot-like types to "plot" if they contain the word 'plot'
    if s in ("pie","bar","line","scatter","histogram","box","heatmap","violin","histogram","donut"):
        return "plot"
    # grouping/ranking combination
    if "grouping" in s or "ranking" in s:
        return "grouping/ranking"
    # exact matches for other canonical types
    if s in ("invalid", "count", "filtering", "stats"):
        return s


def extract_model_name(filename):
    """
    Given filenames like:
      benchmark_query_results_gpt-oss_20b-cloud.xlsx
      benchmark_plot_results_mistral-large-3_675b-cloud.xlsx
      benchmark_stats_results_deepseek-v3.2_cloud.xlsx

    returns:
      gpt-oss_20b-cloud
      mistral-large-3_675b-cloud
      deepseek-v3.2_cloud
    """
    # get basename without extension
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]

    # remove the fixed leading pattern:
    # - optional leading "benchmark_"
    # - then one of "query", "plot", "stats"
    # - then "_results_" (literal)
    cleaned = re.sub(r'^(?:benchmark_)?(?:query|plot|stats)_results_', '', base, flags=re.IGNORECASE)

    # final cleanup: strip any leading/trailing underscores or dots
    cleaned = re.sub(r'^[\._\-]+|[\._\-]+$', '', cleaned)

    return cleaned

# ------------------------
# COLLECT ACCURACIES BY SUBTYPE
# ------------------------
# We'll collect a list of per-run accuracies for each (model, subtype).
# structure: acc_accumulator[model][subtype] = [acc_run1, acc_run2, ...]
acc_accumulator = {}

# find files for the selected benchmarks
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))
if not all_files:
    raise RuntimeError(f"No .xlsx files found in {DATA_DIR}")

# restrict to files that contain any of the USED_BENCHMARKS tokens
def is_file_for_bench(fname, bench_tokens):
    bname = os.path.basename(fname).lower()
    for b in bench_tokens:
        if f"_{b}_" in bname or f"_{b}." in bname or f"{b}_results" in bname:
            return True
    return False

relevant_files = [f for f in all_files if is_file_for_bench(f, USED_BENCHMARKS)]

if not relevant_files:
    raise RuntimeError(f"No files found for benchmarks {USED_BENCHMARKS} in {DATA_DIR}. Files found: {all_files}")

print(f"Found {len(relevant_files)} files for benchmarks {USED_BENCHMARKS}:")
for f in relevant_files:
    print("  ", os.path.basename(f))

for path in relevant_files:
    fname = os.path.basename(path)
    model_name = extract_model_name(fname)
    if model_name == "":
        model_name = os.path.splitext(fname)[0]

    # ensure model entry exists
    acc_accumulator.setdefault(model_name, {s: [] for s in SUBTYPE_ORDER})

    # iterate over run sheets
    for i in range(1, RUNS + 1):
        sheet = f"run_{i}"
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception as e:
            print(f"Warning: could not read {sheet} in {fname}: {e}")
            continue

        if "score" not in df.columns or "eval_type" not in df.columns:
            print(f"Warning: sheet {sheet} in {fname} missing required columns. Found: {df.columns.tolist()}")
            continue

        # map eval_type to canonical subtype
        df["subtype"] = df["eval_type"].apply(canonical_subtype)

        # for each canonical subtype, compute mean accuracy for this run, append
        for subtype in SUBTYPE_ORDER:
            # select rows mapped to this subtype
            sel = df[df["subtype"] == subtype]
            if sel.shape[0] == 0:
                # nothing for this run-subtype: skip (do not append NaN)
                continue
            acc = sel["score"].mean()
            acc_accumulator[model_name].setdefault(subtype, []).append(float(acc))

# ------------------------
# BUILD FINAL DATAFRAME: mean across all collected run-level scores
# ------------------------
# Create DataFrame with index=models, columns=subtypes
models = sorted(acc_accumulator.keys())
results = []
for m in models:
    row = {}
    for subtype in SUBTYPE_ORDER:
        vals = acc_accumulator[m].get(subtype, [])
        if len(vals) == 0:
            row[subtype] = np.nan
        else:
            row[subtype] = float(np.mean(vals))
    results.append(row)

df_subtypes = pd.DataFrame(results, index=models, columns=SUBTYPE_ORDER)
print("\nMean accuracy by model × subtype:")
print(df_subtypes)

# ------------------------
# Prepare for clustering: fill NaNs per-column with column mean
# ------------------------
df_for_plot = df_subtypes.copy()

print("\nMean accuracy by model × subtype:")
print(df_for_plot)

# Compute model-wise mean 
# mean across columns, skipping NaNs (so models missing a subtype are averaged over available ones)
df_subtypes["mean_accuracy"] = df_subtypes.mean(axis=1, skipna=True)

# sort by descending mean accuracy
df_sorted = df_subtypes.sort_values("mean_accuracy", ascending=False)

# keep heatmap data (drop the mean column)
heat_df = df_sorted.drop(columns=["mean_accuracy"])

print("\nMean accuracy by model × subtype (sorted by overall mean desc):")
print(df_sorted)

# ------------------------
# PLOT: cluster only by model (rows)
# ------------------------
sns.set(context="notebook", style="white")

plt.figure(figsize=(7,7)) 

ax = sns.heatmap(
    heat_df,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    linecolor="white",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    cbar_kws={"label": "Mean accuracy"},
)

# formatting
ax.set_ylabel("")  # leave model labels as yticklabels
ax.set_xlabel("Subtype")
ax.set_title("Mean accuracy by subtype (models ordered by overall mean, desc)")

plt.tight_layout()
plt.savefig(out_path, format="pdf", bbox_inches="tight")
plt.show()
print(f"\nHeatmap saved to: {out_path}")

