import pandas as pd
import numpy as np
import traceback
from scipy import stats
from collections import Counter
import os
import time

from src.agents.query_agent import QueryAgent
from src.llm_client import LLMClient
from src.utils import summarize_dataframe, load_data

# Functions to benchmark query agent
# These functions serve to evaluate LLM-generated responses for different query types

def normalize_dataframe(df, context="filtering", atol=0.01):
    """
    Standardize dataframe content for fair comparison:
    - Lowercase, strip whitespace, remove % symbols
    - Convert numeric-looking strings to float
    - Quantize or round numerics for tolerance
    - Replace missing with __NA__
    """
    df = df.copy()

    for c in df.columns:
        s = df[c]
        s = s.astype(str).str.strip().str.lower().str.replace("%", "", regex=False)
        s = s.replace({"nan": np.nan, "none": np.nan})

        # Try to coerce to float where possible
        if pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce").astype(float)  # <-- ensures 20 == 20.0
            if context == "filtering":
                s = np.round(s / atol) * atol
            elif context == "grouping":
                s = s.round(6)
        df[c] = s

    return df.fillna("__NA__")


def filtering_eval(gold_result, llm_result, atol=0.01):
    """
    Evaluate 'filtering' outputs (subset-style DataFrames).
    Requires identical rows (ignoring order) and same set of columns.
    Numeric quantization absorbs float/int differences.
    1-column outputs are name-tolerant.
    """
    import pandas as pd, numpy as np

    def to_df(x):
        if isinstance(x, pd.DataFrame): return x.copy()
        if isinstance(x, pd.Series): return x.reset_index(drop=True).to_frame()
        if isinstance(x, (list, np.ndarray)): return pd.DataFrame({"value": np.array(x).flatten()})
        return pd.DataFrame({"value": [x]})

    gold, llm = to_df(gold_result), to_df(llm_result)
    gold.columns = [str(c).strip().lower() for c in gold.columns]
    llm.columns = [str(c).strip().lower() for c in llm.columns]

    # Align columns if both are single-column or same set
    if set(gold.columns) != set(llm.columns):
        if gold.shape[1] == 1 and llm.shape[1] == 1:
            llm.columns = gold.columns
        else:
            return 0
    llm = llm[gold.columns]

    gold_s = normalize_dataframe(gold, "filtering", atol)
    llm_s  = normalize_dataframe(llm,  "filtering", atol)

    if gold_s.shape != llm_s.shape:
        return 0

    # Row-agnostic comparison
    gold_sorted = gold_s.sort_values(by=list(gold_s.columns)).reset_index(drop=True)
    llm_sorted  = llm_s.sort_values(by=list(gold_s.columns)).reset_index(drop=True)

    return int(gold_sorted.equals(llm_sorted))


def grouping_eval(gold_result, llm_result, atol=0.1):
    """
    Evaluate 'grouping' outputs (aggregated summaries).
    Requires same columns & row count.
    Allows rounding/float precision differences.
    Compares after row sorting and numeric tolerance.
    """
    import pandas as pd, numpy as np

    def to_df(x):
        if isinstance(x, pd.DataFrame): return x.copy()
        if isinstance(x, pd.Series): return x.reset_index(drop=True).to_frame()
        if isinstance(x, (list, np.ndarray)): return pd.DataFrame({"value": np.array(x).flatten()})
        return pd.DataFrame({"value": [x]})

    gold, llm = to_df(gold_result), to_df(llm_result)
    gold.columns = [str(c).strip().lower() for c in gold.columns]
    llm.columns  = [str(c).strip().lower() for c in llm.columns]

    if set(gold.columns) != set(llm.columns):
        if gold.shape[1] == 1 and llm.shape[1] == 1:
            llm.columns = gold.columns
        else:
            return 0
    llm = llm[gold.columns]

    gold_s = normalize_dataframe(gold, "grouping", atol)
    llm_s  = normalize_dataframe(llm,  "grouping", atol)

    if gold_s.shape != llm_s.shape:
        return 0

    # Sort both for row-agnostic comparison
    sort_cols = list(gold_s.columns)
    gold_s, llm_s = gold_s.sort_values(by=sort_cols).reset_index(drop=True), llm_s.sort_values(by=sort_cols).reset_index(drop=True)

    num_cols = gold_s.select_dtypes(include=[np.number]).columns

    for c in num_cols:
        g = pd.to_numeric(gold_s[c], errors="coerce")
        l = pd.to_numeric(llm_s[c], errors="coerce")

        # Round both sides to a consistent precision (e.g., 2 decimals)
        g_rounded = g.round(2)
        l_rounded = l.round(2)

        # Compare with a generous tolerance
        if not np.allclose(g_rounded, l_rounded, atol=max(atol, 0.5), equal_nan=True):
            return 0

    text_cols = [c for c in gold_s.columns if c not in num_cols]
    for c in text_cols:
        if not gold_s[c].fillna("").equals(llm_s[c].fillna("")):
            return 0

    return 1


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


def invalid_eval(llm_result):
    '''
    Evaluate if the LLM correctly refused to answer an invalid query.
    Returns 1 if LLM output if a string with no coding keywords, else 0.
    '''

    if not isinstance(llm_result, str):
        return 0

    coding_keywords = ['def ', 'import', 'print', 'elif', 'pandas', 'numpy','pd.','np.','dropna','groupby','==','.explode','.str.contains','.isna']

    if any(keyword in llm_result for keyword in coding_keywords):
        return 0

    return 1



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

    # Decode any escaped characters in the code
    gold_code = gold_code.encode('utf-8').decode('unicode_escape')

    # Restrict variables accessible during execution
    try:
        context = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "df": df.copy()
        }
        # execute the gold standard code (it should assign the output to variable `result`)
        exec(gold_code, context)

        result = context.get("result", None)

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

        # Skip scoring on execution errors
        if llm_type == "error":
            results.append({"question": question,
            "eval_type": eval_type,
            "score": 0,
            "llm_code": llm_result_dict.get("code", ""),
            "gold_code": gold_code,
            "llm_result_preview": str(llm_result)[:200],
            "gold_result_preview":""})
            time.sleep(40)
            continue

        # Run gold code
        gold_result = run_gold_code(gold_code)

        # Evaluate based on eval_type
        if eval_type == "filtering":
            score = filtering_eval(gold_result, llm_result)
        elif eval_type == "grouping":
            score = grouping_eval(gold_result, llm_result)
        elif eval_type == "ranking":
            score = grouping_eval(gold_result, llm_result)
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

        time.sleep(40) # Pause between requests to avoid rate limits

    # Make final df with results
    df_results = pd.DataFrame(results)

    if not df_results.empty:
        print("Benchmark Results Summary:")
        print(df_results.groupby('eval_type')['score'].mean().round(3))
        print("===================================\n")

    return df_results
        

# Example usage
if __name__ == "__main__":
    file_path = "data/benchmark_questions_111025.xlsx"
    df = pd.read_excel(file_path)

    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

    benchmark_cases = []

    for idx, row in df.iterrows():
        case = {
            "question": row['question'],
            "gold_code": str(row['answer']).strip(),
            "eval_type": row['category'].strip().lower()
        }
        benchmark_cases.append(case)

    # Run benchmark query agent
    model = "kimi-k2:1t-cloud"
    results = benchmark_query_agent(benchmark_cases, model=model)
    results.to_csv(f"data/benchmark_query_agent_results_{model}.csv", index=False)
    



