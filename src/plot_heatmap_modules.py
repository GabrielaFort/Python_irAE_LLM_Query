import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl

# ------------------------
# CONFIG
# ------------------------
DATA_DIR = "../results"        # folder with Excel files
BENCHMARKS = ["classifier", "query", "stats", "plot"]  # which benchmarks to include
RUNS = 5                       # run_1 ... run_5
OUT_DIR = "../results/plots"   # where to save plot
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "heatmap_mean_accuracies.pdf")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "font.size": 9,
})

# ------------------------
# HELPERS
# ------------------------
def extract_model_name(filename, benchmark):
    """Return the model name portion of the filename (no extension). 
    Adjust to match how your files are named.
    Example input names:
      results_mistral-large-3_stats_675b-cloud.xlsx  -> mistral-large-3_stats_675b-cloud  -> we want model part
    This function tries to be flexible with the split logic you used earlier.
    """
    name = os.path.splitext(filename)[0]
    # filenames are like results_{model}_{benchmark}_... or classifier_{model}_...
    if benchmark in ["query", "stats", "plot","classifier"]:
        # remove prefix if present
        if "results_" in name:
            candidate = name.split("results_")[-1]
        elif "classifier_" in name:
            candidate = name.split("classifier_")[-1]
        else:
            candidate = name
        # candidate might be like "mistral-large-3_stats_675b-cloud"
        # remove the suffix that contains the benchmark (if present)
        parts = candidate.split(f"_{benchmark}")
        model = parts[0]
        # sometimes there is an extra underscore before run/variant; strip trailing underscores
        return model.strip("_")
    else:
        # fallback: return full base name
        return name

# ------------------------
# COLLECT MEAN ACCURACIES
# ------------------------
# dictionary: {model_name: {benchmark: mean_acc}}
acc_dict = {}

for benchmark in BENCHMARKS:
    pattern = os.path.join(DATA_DIR, f"*_{benchmark}_*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files:
        # if no files for this benchmark, warn and continue
        print(f"Warning: no files found for benchmark '{benchmark}' with pattern {pattern}")
        continue

    for path in files:
        fname = os.path.basename(path)
        model_name = extract_model_name(fname, benchmark)

        # read each run sheet, compute mean of "score" column
        run_scores = []
        for i in range(1, RUNS + 1):
            sheet = f"run_{i}"
            try:
                df = pd.read_excel(path, sheet_name=sheet)
            except Exception as e:
                # if sheet missing, skip and warn
                print(f"Warning: could not read {sheet} in {fname}: {e}")
                continue

            if "score" not in df.columns:
                print(f"Warning: 'score' column not in {sheet} of {fname}. Available columns: {df.columns.tolist()}")
                continue

            acc = df["score"].mean()
            run_scores.append(acc)

        if len(run_scores) == 0:
            print(f"Warning: no valid runs found for {fname}; skipping.")
            continue

        mean_acc = float(np.mean(run_scores))

        if model_name not in acc_dict:
            acc_dict[model_name] = {}
        acc_dict[model_name][benchmark] = mean_acc

# ------------------------
# BUILD DATAFRAME
# ------------------------
# rows = models, cols = benchmarks
df = pd.DataFrame.from_dict(acc_dict, orient="index")
# order columns as BENCHMARKS
df = df.reindex(columns=BENCHMARKS)
# compute mean accuracy across the selected benchmarks for each model
df["mean_accuracy"] = df.mean(axis=1)
# sort models by descending mean accuracy
df = df.sort_values("mean_accuracy", ascending=False)

# drop the mean column from the heatmap data and keep it for annotations
heat_df = df.drop(columns=["mean_accuracy"])

print("Mean accuracy table (models x benchmarks), sorted by overall mean desc:")
print(df)

# ------------------------
# PLOT HEATMAP
# ------------------------
sns.set(context="notebook", style="white")

plt.figure(figsize=(7,7))  


# Choose cmap and annot formatting
cmap = "viridis"   

ax = sns.heatmap(
    heat_df,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    linecolor="white",
    cmap=cmap,
    vmin=0.0,
    vmax=1.0,
    cbar_kws={"label": "Mean accuracy"},
)

# tighten up labels / title
ax.set_ylabel("")
ax.set_xlabel("Benchmark")
ax.set_title("Mean accuracy across runs (models ordered by overall mean, desc)")

plt.tight_layout()
plt.savefig(out_path, format="pdf", bbox_inches="tight")
plt.show()
print(f"Heatmap saved to: {out_path}")