import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
import os

from src.question_classifier import QuestionClassifier
from src.llm_client import LLMClient

def run_single_benchmark(df, classifier):
    """
    Run a single classification benchmark and return results df
    """
    question_col = df['question'].astype(str)

    # Run Predictions
    preds = []
    for question in question_col:
        pred = classifier.classify(question)
        preds.append(pred)

    results_df = df.copy()
    results_df["predicted"] = preds

    # Compute score (1 if correct, 0 if incorrect - exact match)
    true_labels = results_df['answer'].str.lower().str.strip()
    pred_labels = results_df['predicted'].str.lower().str.strip()
    results_df['score'] = (true_labels == pred_labels).astype(int)

    return results_df

def create_confusion_matrix(true_labels, pred_labels):
    """
    Create a confusion matrix of classification results as a DF
    """
    labels = sorted(true_labels.unique())
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    
    return cm_df

def benchmark_question_classifier(file_path, model_name, n_runs=5):
    """
    Benchmark QuestionClassifier on a labeled set of questions.
    Runs n_runs times and saves result to excel file with multiple sheets.

    Expected columns:
    - 'Question': the user question to classify
    - 'Answer': the expected classification label ('tableqa', 'plot', 'stats', 'guideline')
    """

    # Load benchmark data (csv file)
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower() for col in df.columns]

    # Detect relevant columns
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Input file must contain 'Question' and 'Answer' columns.")
    
    # Instantiate question classifier
    classifier = QuestionClassifier(LLMClient(
        model=model_name,
        api_url="https://ollama.com",
        temperature = 0,
        api_key=os.getenv("OLLAMA_API_KEY")
    ))

    # Create output file path/name
    out_path = f"results/benchmark_classifier_{model_name.replace(':', '_')}.xlsx"

    # Run benchmark n times and save output to excel
    with pd.ExcelWriter(out_path) as writer:
        all_run_preds = []
        for run_num in range(1, n_runs + 1):
            print(f"\nStarting classification run {run_num}/{n_runs} for model {model_name}...")

            # Run single benchmark case
            results_df = run_single_benchmark(df, classifier)

            # Calculate metrics 
            true_labels = results_df['answer'].str.lower().str.strip()
            pred_labels = results_df['predicted'].str.lower().str.strip()

            # Print classification report to screen
            print(f"\nClassification Report (Run {run_num}):\n")
            print(classification_report(true_labels, pred_labels, digits=3))

            # Create and print confusion matrix to screen
            cm_df = create_confusion_matrix(true_labels, pred_labels)
            print(f"\nConfusion Matrix (Run {run_num}):\n")
            print(tabulate(cm_df, headers='keys', tablefmt='fancy_grid'))

            # Save results to Excel sheet
            results_df.to_excel(writer, sheet_name=f"run_{run_num}", index=False) 

            # Save confusion matrix to separate sheet also 
            cm_df.to_excel(writer, sheet_name=f"run_{run_num}_cm")

            # Collect predictions for summary majority-vote confusion matrix
            all_run_preds.append(pred_labels.reset_index(drop=True))

            print(f"Completed run {run_num}/{n_runs}.\n")

        # Summary confusion matrix from majority vote across runs
        if all_run_preds:
            preds_matrix = pd.concat(all_run_preds, axis=1)
            preds_matrix.columns = [f"run_{i}" for i in range(1, len(all_run_preds) + 1)]
            # pandas mode returns sorted modes; pick the first in ties
            majority_pred = preds_matrix.mode(axis=1).iloc[:, 0]

            true_labels = df['answer'].str.lower().str.strip()
            summary_cm_df = create_confusion_matrix(true_labels, majority_pred)

            print("\nSummary Confusion Matrix (Majority Vote):\n")
            print(tabulate(summary_cm_df, headers='keys', tablefmt='fancy_grid'))
            summary_cm_df.to_excel(writer, sheet_name="summary_majority_cm")

            # Save majority-vote results in the same format as run_* sheets
            majority_results = df.copy()
            majority_results["predicted"] = majority_pred.values
            majority_results["score"] = (true_labels == majority_pred).astype(int)
            majority_results.to_excel(writer, sheet_name="summary_majority", index=False)

        print(f"All results saved to {out_path}")
        

if __name__ == "__main__":
    #model_list = ["devstral-2:123b-cloud","gpt-oss:20b-cloud","gpt-oss:120b-cloud","qwen3-coder:480b-cloud",
    #                "gemma3:27b-cloud","deepseek-v3.1:671b-cloud","glm-4.6:cloud","cogito-2.1:671b-cloud",
    #               "minimax-m2:cloud","kimi-k2:1t-cloud","deepseek-v3.2:cloud","glm-4.7:cloud","mistral-large-3:675b-cloud",
    #                "minimax-m2.1:cloud","gemini-3-flash-preview:cloud"]
    model_list = ["deepseek-v3.2:cloud"]
    
    
    for model in model_list:
        print(f"\n{'='*60}\nBenchmarking model: {model}\n{'='*60}")

        benchmark_question_classifier("data/question_classifier_benchmark.csv",
                                      model,
                                      n_runs=5)
        print(f"\nCompleted benchmarking for model: {model}\n")

