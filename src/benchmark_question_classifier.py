import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
import os

from src.question_classifier import QuestionClassifier
from src.llm_client import LLMClient

def benchmark_question_classifier(file_path, model_name):
    '''
    Benchmark QuestionClassifier on a labeled set of questions in Excel/CSV.
    
    Expected columns:
    - 'Question': The user question to classify
    - 'Answer': The expected classification label ('tableqa', 'plot', 'stats')
    '''

    # Load benchmark data
    df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    df.columns = [col.strip().lower() for col in df.columns]

    # Detect relevant columns
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Input file must contain 'Question' and 'Answer' columns.")

    question_col = df['question']
    answer_col = df['answer'] 
    
    # Instantiate QuestionClassifier
    classifier = QuestionClassifier(LLMClient(model=model_name,api_url="https://ollama.com",temperature=0,api_key=os.getenv("OLLAMA_API_KEY")))

    # Run predictions
    preds = []
    for question in question_col:
        pred = classifier.classify(question)
        preds.append(pred)

    df["Predicted"] = preds

    # Evaluation 
    true_labels = answer_col.str.lower().str.strip()
    pred_labels = df["Predicted"].str.lower().str.strip()

    print("\nClassification Report:\n")
    print(classification_report(true_labels, pred_labels, digits=3))

    print("\nConfusion Matrix:\n")
    labels = sorted(true_labels.unique())
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(tabulate(cm_df, headers='keys', tablefmt='fancy_grid'))

    # Save results
    out_path = file_path.replace(".xlsx", f"_results_{model_name}.xlsx").replace(".csv", f"_results_{model_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    model_list = ["deepseek-v3.1:671b-cloud","gpt-oss:20b-cloud","gpt-oss:120b-cloud","kimi-k2:1t-cloud","qwen3-coder:480b-cloud","glm-4.6:cloud","minimax-m2:cloud"]
    # Example usage
    for model in model_list:
        print(f"\n\nBenchmarking model: {model}\n")
        benchmark_question_classifier("~/Documents/Tan_Lab/Projects/Python_irAE_LLM_Query/data/question_classifier_benchmark.csv", model)




