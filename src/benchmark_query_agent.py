import pandas as pd
import numpy as np
import traceback
from scipy import stats
from collections import Counter
import os

from src.agents.query_agent import QueryAgent
from src.llm_client import LLMClient
from src.utils import summarize_dataframe, load_data



# Functions to benchmark query agent
# These functions serve to evaluate LLM-generated responses for different query types

# Is this handling NA's correctly?????
def filtering_eval(gold_result, llm_result):
    """
    Evaluate filtering-style queries by comparing LLM result to gold standard.
    Returns 1 if they match, else 0.
    Handles DataFrames, Series, arrays, and lists.
    """

    # Handle missing or invalid inputs
    if gold_result is None or llm_result is None:
        return 0

    # --- Flatten gold_result ---
    if isinstance(gold_result, pd.DataFrame):
        if gold_result.shape[1] == 1:
            gold_flat = gold_result.iloc[:, 0].dropna().values
        else:
            gold_flat = gold_result.values.flatten()
    elif isinstance(gold_result, pd.Series):
        gold_flat = gold_result.dropna().values
    elif isinstance(gold_result, (list, np.ndarray)):
        gold_flat = np.array(gold_result).flatten()
    else:
        return 0

    # --- Flatten llm_result ---
    if isinstance(llm_result, pd.DataFrame):
        if llm_result.shape[1] == 1:
            llm_flat = llm_result.iloc[:, 0].dropna().values
        else:
            llm_flat = llm_result.values.flatten()
    elif isinstance(llm_result, pd.Series):
        llm_flat = llm_result.dropna().values
    elif isinstance(llm_result, (list, np.ndarray)):
        llm_flat = np.array(llm_result).flatten()
    else:
        return 0

    # --- Convert both to strings for fair comparison ---
    gold_flat = pd.Series(gold_flat).dropna().astype(str).str.strip().str.lower()
    llm_flat  = pd.Series(llm_flat).dropna().astype(str).str.strip().str.lower()

    # --- Try numeric comparison first ---
    gold_num = pd.to_numeric(gold_flat, errors="coerce").dropna()
    llm_num  = pd.to_numeric(llm_flat, errors="coerce").dropna()
    if len(gold_num) and len(llm_num):
        if set(np.round(gold_num, 3)) == set(np.round(llm_num, 3)):
            return 1

    # --- Fallback: string-based comparison (order-insensitive) ---
    if set(gold_flat) == set(llm_flat):
        return 1

    return 0



def invalid_eval(llm_result):
    '''
    Evaluate if the LLM correctly refused to answer an invalid query.
    Returns 1 if LLM output if a string with no coding keywords, else 0.
    '''

    if not isinstance(llm_result, str):
        return 0

    coding_keywords = ['def ', 'import', 'print', 'elif', 'return', 'pandas', 'numpy','pd.','np.','dropna','groupby','==','.explode','.str.contains','.isna']

    if any(keyword in llm_result for keyword in coding_keywords):
        return 0

    return 1


# Will this work if llm result is interpreted as a string? 
def count_eval(gold_result, llm_result):
    '''
    Evaluate count-style queries by comparing LLM result to gold standard.
    Returns 1 if they match (within tolerance), else 0.
    '''
    def extract_num(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            vals = np.array(x.values).flatten()
            if len(vals) == 1:
                x = vals[0]
            else:
                return None
        if isinstance(x, str):
            x = ''.join(ch for ch in x if ch.isdigit() or ch in ['.','-',','])
            x = x.replace(',','')
        try:
            return float(x)
        except:
            return None
        
    gold_num = extract_num(gold_result)
    llm_num = extract_num(llm_result)

    if gold_num is None or llm_num is None:
        return 0
    
    if np.isclose(gold_num, llm_num, atol=0.1):
        return 1
    
    return 0


# Now need to generate LLM code using query agent
# Instead of sending question through question classifier, just call it a 'query' type
def run_query_agent(question,model):
    '''
    Given a question string, use the QueryAgent to generate and execute code.
    Returns the result dictionary from execute_code.
    '''
    # First, read in and clean dataset using function from app.py
    df = load_data()

    # Instantiate query agent
    myllm = LLMClient(model=model,
                    api_url="https://ollama.com",
                    temperature=0,
                    api_key=os.getenv("OLLAMA_API_KEY"))

    # Generate summary for df for prompt
    summary = summarize_dataframe(df)

    # Instantiate query agent
    agent = QueryAgent(df, myllm)

    # Generate code for question and summary
    code = agent.handle(question, summary)
    # Execute code
    result = agent.execute_code(code)

    return result



# And execute gold standard code
def run_gold_code(gold_code):
    '''
    Given a gold standard code string, execute the code safely.
    Returns the result variable from the executed code.
    '''
    # First, read in and clean dataset using function from app.py
    df = load_data()

    # Restrict variables accessible during execution
    try:
        safe_globals = {"pd": pd, "np": np, "stats": stats, "Counter": Counter, "__builtins__": __builtins__} 
        safe_locals = {"df" : df.copy()}

        # execute the gold standard code (it should assign the output to variable `result`)
        exec(gold_code, {**safe_globals, **safe_locals}, safe_locals)

        result = safe_locals.get("result", None)

        return result

    except Exception as e:
        print("Error executing gold code:")
        traceback.print_exc()
        return None


# Write main function
def benchmark_query_agent(benchmark_cases, model):
    '''
    Given a list of benchmark cases and an LLM model name,
    runs the query agent on each case and evaluates the results.
    
    benchmark_cases: list of dicts with keys:
        - question: str
        - gold_code: str
        - eval_type: str ("filtering", "count", "invalid")
    
    Returns a list of results with evaluation scores.
    '''
    results = []

    for case in benchmark_cases:
        question = case['question']
        gold_code = case['gold_code']
        eval_type = case['eval_type']

        print(f"Processing question: {question}")

        # Run query agent
        llm_result_dict = run_query_agent(question, model)
        llm_result = llm_result_dict.get('data', None)
        llm_type = llm_result_dict.get('type', None)
        print("LLM code executed.")

        # Skip scoring on execution errors
        if llm_type == "error":
            score = 0
        
        # Run gold code
        gold_result = run_gold_code(gold_code)
        print("Gold code executed.")

        # Evaluate based on eval_type
        if eval_type == "filtering":
            score = filtering_eval(gold_result, llm_result)
        elif eval_type == "count":
            score = count_eval(gold_result, llm_result)
        elif eval_type == "invalid":
            score = invalid_eval(llm_result)
        else:
            score = 0  # Unknown eval type

        results.append({
            "question": question,
            "eval_type": eval_type,
            "score": score,
            "llm_code": llm_result_dict.get("code", ""),
            "gold_code": gold_code,
            "llm_result_preview": str(llm_result)[:200],
            "gold_result_preview": str(gold_result)[:200]
        })

    # Make final df with results
    df_results = pd.DataFrame(results)

    if not df_results.empty:
        print("Benchmark Results Summary:")
        print(df_results.groupby('eval_type')['score'].mean().round(3))
        print("===================================\n")

    return df_results
        

# Example usage
if __name__ == "__main__":
    benchmark_cases = [
        {
            "question": "Return the patient IDs of those treated with pembrolizumab.",
            "gold_code": "result=df[df['ici_drug_name'].str.contains(r'Pembrolizumab', na=False, case=False)]['patient_id'].unique()",
            "eval_type": "filtering"
        },
        {
            "question": "How many patients experienced colitis?",
            "gold_code": "result=len(df[df['irae'].str.contains(r'Colitis',na=False,case=False)])",
            "eval_type": "count"
        },
        {
            "question": "Only return age and irae type and immunotherapy treatment for patients treated with pembrolizumab.",
            "gold_code": "pembro_patients = df[ df['ici_drug_name'].str.contains('Pembrolizumab', na=False, case=False)]\nresult = pembro_patients[['age', 'irae_type', 'ici_drug_name']]",
            "eval_type": "filtering"
        },
    ]

    results = benchmark_query_agent(benchmark_cases, model="deepseek-v3.1:671b-cloud")
    results.to_csv("benchmark_query_agent_results.csv", index=False)
    # print(results)