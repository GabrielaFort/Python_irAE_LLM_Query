# Helpful functions
from src.llm_client import LLMClient
import pandas as pd 
import numpy as np

# Instantiate LLM clients with preset configurations   
def question_classifier_llm():
    myllm = LLMClient(model="qwen/qwen3-coder-30b",
                api_url="http://10.104.134.74:443",
                temperature=0)
    return myllm

def query_llm():
    myllm = LLMClient(model="qwen/qwen3-coder-30b",
                api_url="http://10.104.134.74:443",
                temperature=0)
    return myllm

def plotter_llm():
    myllm = LLMClient(model="qwen/qwen3-coder-30b",
                api_url="http://10.104.134.74:443",
                temperature=0.6)
    return myllm

def stats_llm():
    myllm = LLMClient(model="qwen/qwen3-coder-30b",
                api_url="http://10.104.134.74:443",
                temperature=0.1)
    return myllm  

def error_checker_llm():
    myllm = LLMClient(model="qwen/qwen3-coder-30b",
                api_url="http://10.104.134.74:443",
                temperature=0)
    return myllm  


# This function will take a dataframe as input and return a summary of the dataframe
# Include data types, column names, and a few example values from each column
# This will be added to the LLM prompts to help the LLM understand the data structure
def summarize_dataframe(df, max_rows=10):
    preview = {}
    hints = []
    dtypes = df.dtypes.astype(str).to_dict()

    # Build a formatted string showing each column and its dtype
    columns_with_types = ", ".join([f"{col} ({dtypes[col]})" for col in df.columns])


    for col in df.columns:
        non_null_values = df[col].dropna().astype(str).unique()[:max_rows]
        preview[col] = list(non_null_values)

    # Identify whether any columns contain comma-separated entries suggesting multiple values per row
        if df[col].astype(str).str.contains(",").any():
            hints.append(f"- Column '{col}' may contain multiple values per record, separated by commas.")

    # Detect ID-like columns
    for col in df.columns:
        nunique = df[col].nunique(dropna = False)
        if nunique == len(df):
            hints.append(f"- Column '{col}' is likely ID column with unique values for each record.")

    # Detect numeric columns and ranges
    for col in df.select_dtypes(include=[np.number]).columns:
        min_val = df[col].min()
        max_val = df[col].max()
        hints.append(f"- Column '{col}' is numeric with range [{min_val}, {max_val}].")

    # Detect categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_ratio = df[col].nunique(dropna = True) / len(df)
        if unique_ratio < 0.05 and df[col].nunique() < 30: # small number of discrete categories
            hints.append(f"- Column '{col}' is likely categorical with {df[col].nunique()} unique values.")

    # Build hint text
    if hints:
        hint_text = "\n".join(hints)
    else:
        hint_text = "No special structural notes detected."
    
    info = {
        "columns": columns_with_types,
        "num_rows": len(df),
        "num_columns": len(df.columns),
    }

    # Build preview table
    df_preview = pd.DataFrame.from_dict(preview, orient='index').transpose()

    summary = f"""
DataFrame Summary:
Rows: {info['num_rows']}, Columns: {info['num_columns']}
Columns: {info['columns']}
Preview (first {max_rows} unique non-null values per column):
{df_preview.to_markdown(index=False)}

Schema notes:
{hint_text}
"""
    return summary


# Function to clean up code exported from LLM 
# Remove any leading/trailing whitespace and ensure proper indentation
# Remove any markdown formatting if present
def clean_code(code):
    # Remove markdown code block formatting if present
    if code.startswith("```") and code.endswith("```"):
        code = "\n".join(code.split("\n")[1:-1])
    
    # Strip leading/trailing whitespace
    code = code.strip()
    
    return code


if __name__ == "__main__":
    # Simple test of summarize_dataframe function
    df = pd.read_csv("../data/data_david_new.csv", sep="$")
    #df = pd.read_csv("../data/irae_data_cleaned.csv")
    print(summarize_dataframe(df, max_rows=5))
    
  
