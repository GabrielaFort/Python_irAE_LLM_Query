# Helpful functions
from llm_client import LLMClient
import pandas as pd 

# Instantiate LLM clients with preset configurations
def llama3_llm():
    myllm = LLMClient(model="meta-llama-3-8b-instruct",
                api_url="http://localhost:1234",
                temperature=0)
    return myllm

def qwen_query_llm():
    myllm = LLMClient(model="qwen/qwen3-coder-30b",
                api_url="http://localhost:1234",
                temperature=0)
    return myllm

def qwen_plotter_llm():
    myllm = LLMClient(model="qwen/qwen3-coder-30b",
                api_url="http://localhost:1234",
                temperature=0.4)
    return myllm


# This function will take a dataframe as input and return a summary of the dataframe
# Include data types, column names, and a few example values from each column
# This will be added to the LLM prompts to help the LLM understand the data structure
def summarize_dataframe(df, max_rows=10):
    preview = {}
    for col in df.columns:
        non_null_values = df[col].dropna().unique()[:max_rows]
        preview[col] = list(non_null_values)
    
    info = {
        "columns": list(df.columns),
        "num_rows": len(df),
        "num_columns": len(df.columns),
    }

    df_preview = pd.DataFrame.from_dict(preview, orient='index').transpose()
    summary = f"""
DataFrame Summary:
Rows: {info['num_rows']}, Columns: {info['num_columns']}
Columns: {', '.join(info['columns'])}
Preview (first {max_rows} unique non-null values per column):
{df_preview.to_markdown(index=False)}
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

