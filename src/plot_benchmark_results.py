import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting results from benchmarking various agents

def read_benchmark_results(file_path):
    """Read benchmark results from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def plot_benchmark_results(df):
    """Plot benchmark results using seaborn."""
    sns.set(style="whitegrid")

    # Sort by Accuracy (descending)
    df = df.sort_values(by="Accuracy", ascending=False)

    plt.figure(figsize=(8, 7))
    
    ax = sns.barplot(x="Model", y="Accuracy", data=df)
    ax.set_title("Benchmark Results of Different Models")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}%', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.ylim(0, 100)
    plt.tight_layout()

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")

    plt.savefig("../data/benchmark_results.pdf", format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    # Example usage
    benchmark_file = "../data/plot_benchmark_accuracy_combined.csv"
    benchmark_df = read_benchmark_results(benchmark_file)
    plot_benchmark_results(benchmark_df)