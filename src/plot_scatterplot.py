# Create scatter plot of parameter size versus average accuracy across benchmarks, and compute Pearson correlation.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import matplotlib as mpl

OUT_DIR = "../results/plots"   # where to save plot
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "scatter_parameters_vs_accuracy.pdf")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "font.size": 9,
})

# Load data
df = pd.read_csv("../results/parameters_vs_accuracy.csv")

# Run statistical test: Pearson correlation 
x = df["Parameters (b)"]          # raw parameter count
y = df["Average Accuracy"]

r, p = pearsonr(x, y)

print(f"Pearson r (raw params vs accuracy): {r:.3f}")
print(f"p-value: {p:.4g}")

# Seaborn style
sns.set_theme(style="whitegrid", context="talk")

# Scatter plot
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df,
    x="Parameters (b)",
    y="Average Accuracy",
    s=80
)

# Labels and title
plt.xlabel("Parameter Count (Billions)")
plt.ylabel("Average Accuracy")
plt.title("Average Accuracy vs Parameter Count")

plt.tight_layout()
plt.savefig(out_path, format="pdf", bbox_inches="tight")
plt.show()
print(f"Scatter plot saved to: {out_path}")

