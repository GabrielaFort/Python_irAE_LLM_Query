import pandas as pd
import numpy as np
import traceback
from scipy import stats
from collections import Counter
import os
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib_venn._common import VennDiagram
from matplotlib_venn import venn2, venn3

from src.agents.query_agent import QueryAgent
from src.agents.stats_agent import StatsAgent
from src.agents.plot_agent import PlotAgent
from src.llm_client import LLMClient
from src.utils import summarize_dataframe, load_data

# Functions to benchmark analytical modules
# These functions serve to evaluate LLM-generated responses for different query types within tableQA, statistics, and plotting modules

def normalize_dataframe(df, context="filtering", atol=0.01):
    """
    Standardize dataframe content for fair comparison:
    - Lowercase, strip whitespace, remove % symbols
    - Convert numeric-looking strings to float
    - Quantize or round numerics for tolerance
    - Replace missing with __NA__
    Robust to cases where df[col] returns a DataFrame (squeezes single-col frames
    or stringifies multi-col frames).
    """
    df = df.copy()

    for c in df.columns:
        s = df[c]

        # If column entry is a DataFrame (rare), try to reduce to a Series
        if isinstance(s, pd.DataFrame):
            # If single-column DataFrame, squeeze to Series
            if s.shape[1] == 1:
                s = s.iloc[:, 0]
            else:
                # Multi-column: each row into a single string so it can be compared
                s = s.astype(str).apply(lambda row: " | ".join(row.values.astype(str)), axis=1)

        # Now s should be a Series. If it's not, coerce to string Series.
        if not isinstance(s, pd.Series):
            s = pd.Series(s)

        # Handle object-type text columns
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            s = s.astype(str).str.strip().str.lower().str.replace("%", "", regex=False)
            s = s.replace({"nan": np.nan, "none": np.nan})

        # Try to coerce to float where possible
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() > 0:
            s = s_num.astype(float)

            if context == "filtering":
                s = np.round(s / atol) * atol
            elif context == "grouping":
                s = s.round(6)

        df[c] = s

    return df.fillna("__NA__")


def dataframe_eval(gold_result, llm_result, eval_type="filtering", atol=0.01):
    """
    Unified evaluation for filtering and grouping outputs.
    
    Args:
        gold_result: Expected result
        llm_result: LLM-generated result
        eval_type: "filtering" or "grouping"
        atol: Absolute tolerance for numeric comparison

    Returns:
        1 if LLM result matches gold result according to eval rules, else 0
    """
    def to_df(x):
        # Convert different input types to pandas DF
        if isinstance(x, pd.DataFrame): return x.copy()
        if isinstance(x, pd.Series): return x.reset_index(drop=True).to_frame()
        if isinstance(x, (list, np.ndarray)): return pd.DataFrame({"value": np.array(x).flatten()})
        return pd.DataFrame({"value": [x]})

    gold, llm = to_df(gold_result), to_df(llm_result)

    # Normalize column names - convert to string, strip, lower
    gold.columns = [str(c).strip().lower() for c in gold.columns]
    llm.columns = [str(c).strip().lower() for c in llm.columns]

    # Column alignment, if sets of column names differ, handle simple case of both being single-column DFs
    # by forcing the LLM col to use gold col name, else fail (return 0)
    if set(gold.columns) != set(llm.columns):
        if gold.shape[1] == 1 and llm.shape[1] == 1:
            llm.columns = gold.columns
        else:
            return 0
    llm = llm[gold.columns]

    # Normalize based on eval type (question subtype)
    context = "filtering" if eval_type == "filtering" else "grouping"
    gold_s = normalize_dataframe(gold, context, atol)
    llm_s = normalize_dataframe(llm, context, atol)

    # If normalized DFs have different shapes (rows or cols) they fail
    if gold_s.shape != llm_s.shape:
        return 0

    # Sort by columns and then reset index
    sort_cols = sorted(gold_s.columns)
    gold_sorted = gold_s.sort_values(by=sort_cols).reset_index(drop=True)
    llm_sorted = llm_s.sort_values(by=sort_cols).reset_index(drop=True)

    # Different comparison logic based on type
    # For filtering, require exact equality between sorted frames
    # True only if shape, dtypes, and values are equal
    if eval_type == "filtering":
        return int(gold_sorted.equals(llm_sorted))
    
    # For grouping, allow numeric columns to match within tolerance and require exact text equality
    else:  # grouping
        num_cols = gold_sorted.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            g = pd.to_numeric(gold_sorted[c], errors="coerce")
            l = pd.to_numeric(llm_sorted[c], errors="coerce")
            g_rounded = g.round(2)
            l_rounded = l.round(2)
            if not np.allclose(g_rounded, l_rounded, atol=max(atol, 0.5), equal_nan=True):
                return 0

        text_cols = [c for c in gold_sorted.columns if c not in num_cols]
        for c in text_cols:
            if not gold_sorted[c].fillna("").equals(llm_sorted[c].fillna("")):
                return 0
        
        # If all numeric and text columns match according to rules - return success
        return 1


def count_eval(gold_result, llm_result):
    '''
    Evaluate count-style queries by comparing LLM result to gold standard.
    Returns 1 if they match (within tolerance), else 0.
    '''

    # Extract single numeric value from various input types
    def extract_num(x):
        # If x is a single-value DF or series, extract that value
        # If it contains multiple values, it fails
        if isinstance(x, (pd.DataFrame, pd.Series)):
            vals = np.array(x.values).flatten()
            if len(vals) == 1:
                x = vals[0]
            else:
                return None
        if isinstance(x, str):
            # If x is a string, keep digits, decimal point, minus sign, commas
            # Remove commas and try to convert to float 
            # if this fails, then response fails
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
    
    # Compare numbers with tolerance
    if np.isclose(gold_num, llm_num, atol=0.1):
        return 1
    
    return 0


def stats_eval(gold_result, llm_result):
    '''
    Evaluate statistical outputs by comparing LLM result to gold standard.
    Gold standard is always a DF with a 'statistic' and a 'p_value' column.
    Returns 1 if all stats and p-values match within tolerance, else 0.
    '''
    # Make sure LLM result is a DF, convert if series, else return 0
    if isinstance(llm_result, pd.Series):
        llm_df = llm_result.reset_index(drop=True).to_frame()
    elif isinstance(llm_result, pd.DataFrame):
        llm_df = llm_result
    else:
        return 0
    
    # Check required columns 
    # Statistics column could be named a range of different values, same with P-value column
    # Want to be flexible for additional columns
    stat_col_candidates = ['statistic', 'stat', 'test_statistic', 'correlation_coefficient', 'corr','coefficient','coeff','chi2']
    pval_col_candidates = ['p_value', 'pvalue', 'pval','p_val',"p-value","p value", "p-val","p val"]
    stat_col = None
    pval_col = None
    for c in llm_df.columns:
        # The df must have one column with a stat keyword and one col with a pvalue keyword
        c_lower = str(c).strip().lower()
        if any(keyword in c_lower for keyword in stat_col_candidates):
            stat_col = c
        if any(keyword in c_lower for keyword in pval_col_candidates):
            pval_col = c
    if stat_col is None or pval_col is None:
        return 0
    # Check row counts match
    # Strip LLM result to only these two cols
    llm_df = llm_df[[stat_col, pval_col]]

    # Reject mismatched row counts
    if llm_df.shape[0] != gold_result.shape[0]:
        return 0
    
    # Compare each row's statistic and p-value
    for i in range(gold_result.shape[0]):
        gold_stat = gold_result.iloc[i]['statistic']
        gold_pval = gold_result.iloc[i]['p_value']
        llm_stat = llm_df.iloc[i][stat_col]
        llm_pval = llm_df.iloc[i][pval_col]
        try:
            # Compare test statistic and p value within tolerance, if not similar enough, fail
            if not np.isclose(float(gold_stat), float(llm_stat), atol=0.1):
                return 0
            if not np.isclose(float(gold_pval), float(llm_pval), atol=0.01):
                return 0
        except:
            return 0
        
    return 1

def plot_type_eval(llm_result, eval_type, llm_code):
    '''
    Evaluate plotting outputs by checking if the LLM-generated plot type matches the expected type.
    Returns 1 if the plot type matches, else 0.
    '''
    if not hasattr(llm_result, "to_plotly_json") and not isinstance(llm_result, plt.Figure) and not hasattr(llm_result, "figure"):  
        return 0
    # Keywords to look for in LLM code based on eval_type
    keywords = None
    if eval_type == "histogram":
        keywords = ["px.histogram", "go.Histogram"]
    elif eval_type == "box":
        keywords = ["px.box", "go.Box"]
    elif eval_type == "bar":
        keywords = ["px.bar", "go.Bar"]
    elif eval_type == "pie":
        keywords = ["px.pie", "go.Pie"]
    elif eval_type == "scatter":
        keywords = ["px.scatter", "go.Scatter"]
    elif eval_type == "venn":
        keywords = ["venn2", "venn3", "matplotlib_venn"]
    elif eval_type == "heatmap":
        keywords = ["px.imshow", "go.Heatmap"]
    elif eval_type == "violin":
        keywords = ["px.violin", "go.Violin"]

    if keywords is not None:
        for kw in keywords:
            if kw in llm_code:
                return 1
    # Extra checks for special cases
    if eval_type == "donut":
        if "px.pie" in llm_code or "go.Pie" in llm_code:
            if "hole=" in llm_code:
                return 1
    elif eval_type == "line":
        if "px.line" in llm_code:
            return 1
        elif "go.Scatter" in llm_code:
            if "line=" in llm_code:
                return 1          
    return 0


def plot_data_eval(llm_plot_data, gold_plot_data):
    """
    Evaluate if the LLM-generated plot_data matches the gold plot_data.
    Order-insensitive, column-name-insensitive.
    """
    # Both must be DataFrames
    if not isinstance(llm_plot_data, pd.DataFrame) or not isinstance(gold_plot_data, pd.DataFrame):
        return 0

    # Reset index to avoid index mismatch problems
    llm_plot_data = llm_plot_data.reset_index(drop=True)
    gold_plot_data = gold_plot_data.reset_index(drop=True)

    # Row count must match
    if llm_plot_data.shape[0] != gold_plot_data.shape[0]:
        return 0

    print(f"Both have same number of rows: {llm_plot_data.shape[0]}")

    # Helper normalization for text values 
    def normalize_text(series):
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .replace("nan", "__na__")
            .fillna("__na__")
        )

    # Now loop through gold columns and try to match to any LLM column 
    for gold_col in gold_plot_data.columns:
        gold_series = gold_plot_data[gold_col]
        match_found = False

        # Pre-normalize gold values (numeric or text)
        if pd.api.types.is_numeric_dtype(gold_series):
            gold_num = pd.to_numeric(gold_series, errors="coerce").fillna(-99999).to_numpy()
            gold_num_sorted = np.sort(gold_num)
        else:
            gold_text = normalize_text(gold_series)
            gold_counter = Counter(gold_text.tolist())

        # Attempt to match this gold column with ANY llm column
        for llm_col in llm_plot_data.columns:
            llm_series = llm_plot_data[llm_col]

            # Numeric comparison - sort both cols and compare with tolerance
            if pd.api.types.is_numeric_dtype(gold_series):
                llm_num = pd.to_numeric(llm_series, errors="coerce").fillna(-99999).to_numpy()

                if len(llm_num) != len(gold_num_sorted):
                    continue

                if np.allclose(np.sort(llm_num), gold_num_sorted, atol=0.1, equal_nan=True):
                    print(f"Matched numeric column: Gold {gold_col} ↔ LLM {llm_col}")
                    match_found = True
                    break

            # Text comparison - compare text columns (case insensitive, whitespace insensitive, order insensitive, frequency sensitive)
            else:
                llm_text = normalize_text(llm_series)
                llm_counter = Counter(llm_text.tolist())

                if llm_counter == gold_counter:
                    print(f"Matched text column: Gold {gold_col} ↔ LLM {llm_col}")
                    match_found = True
                    break

        if not match_found:
            print(f"No match found for gold column: {gold_col}")
            return 0

    return 1


def invalid_eval(llm_result):
    '''
    Evaluate if the LLM correctly refused to answer an invalid query.
    Returns 1 if LLM output if a string with no coding keywords, else 0.
    '''
    if not isinstance(llm_result, str):
        return 0

    coding_keywords = ['def ', 'import', 'print', 'elif', 'pd.','np.','dropna','groupby','==','.explode','.str.contains','.isna']

    if any(keyword in llm_result for keyword in coding_keywords):
        return 0

    return 1


# Now need to generate LLM code using different modules
# Instead of sending question through question classifier, just call the type
def run_agent(question,model,agent,temp):
    '''
    Given a question string, use the QueryAgent to generate and execute code.
    Returns the result dictionary from execute_code.
    '''
    # First, read in and clean dataset using function from utils.py
    df = load_data()

    # Instantiate llm client
    myllm = LLMClient(model=model,
                    api_url="https://ollama.com",
                    temperature=temp,
                    api_key=os.getenv("OLLAMA_API_KEY"))

    # Generate summary for df for prompt
    summary = summarize_dataframe(df)

    # Instantiate analytical module
    if agent == 'query':
        agent = QueryAgent(df, myllm)
    elif agent == 'stats':
        agent = StatsAgent(df, myllm)
    elif agent == 'plot':
        agent = PlotAgent(df, myllm)
    
    # Generate code for question and summary
    max_attempts = 3
    for attempt in range(1, max_attempts+1):
        try:
            code = agent.handle(question, summary)
            break
        except RuntimeError as e:  
            print(f"LLM call failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)  # exponential backoff

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
            "stats": stats,
            "Counter": Counter,
            "plt": plt,
            "px": px,
            "go": go,
            "venn2": venn2,
            "venn3": venn3,
            "df": df.copy()
        }
        # execute the gold standard code (it should assign the output to variable `result`)
        exec(gold_code, context)

        result = context.get("result", None)
        plot_data = context.get("plot_data", None)

        return {"result": result,"plot_data": plot_data}

    except Exception as e:
        print("Error executing gold code:")
        traceback.print_exc()
        return None


# Write main function
def benchmark_agent(benchmark_cases, model, agent, temp):
    '''
    Given a list of benchmark cases and an LLM model name,
    runs the requested agent on each case and evaluates the results.
    
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

        # Run question and extract result
        llm_result_dict = run_agent(question, model, agent, temp)
        llm_result = llm_result_dict.get('data', None)
        llm_type = llm_result_dict.get('type', None)
        llm_plot_data = llm_result_dict.get("plot_data", None)

        # Skip scoring on execution errors
        if llm_type == "error":
            results.append({"question": question,
            "eval_type": eval_type,
            "score": 0,
            "llm_code": llm_result_dict.get("code", ""),
            "gold_code": gold_code,
            "llm_result_preview": str(llm_result)[:200],
            "gold_result_preview":""})
            continue

        # Run gold code
        gold_result_dict = run_gold_code(gold_code)
        if gold_result_dict is None:
            results.append({"question": question,
            "eval_type": eval_type,
            "score": 0,
            "llm_code": llm_result_dict.get("code", ""),
            "gold_code": gold_code,
            "llm_result_preview": str(llm_result)[:200],
            "gold_result_preview":""})
            continue

        gold_result = gold_result_dict.get("result", None)
        gold_plot_data = gold_result_dict.get("plot_data", None)

        # Evaluate based on eval_type
        if eval_type == "filtering":
            score = dataframe_eval(gold_result, llm_result, eval_type="filtering", atol=0.01)
        elif eval_type == "grouping":
            score = dataframe_eval(gold_result, llm_result, eval_type="grouping", atol=0.1)
        elif eval_type == "ranking":
            score = dataframe_eval(gold_result, llm_result, eval_type="grouping", atol=0.1)
        elif eval_type == "count":
            score = count_eval(gold_result, llm_result)
        elif eval_type == "invalid":
            score = invalid_eval(llm_result)
        elif eval_type == "stats":
            score = stats_eval(gold_result, llm_result)
        elif eval_type in ["histogram","box","bar","pie","donut","scatter","line","violin","heatmap"]:
            llm_code = llm_result_dict.get("code", "")
            score = plot_type_eval(llm_result, eval_type, llm_code)
            print(f"Plot type eval score: {score}")
            if score == 1:
                print("llm_data:", llm_plot_data, "gold_data:" , gold_plot_data)
                if llm_plot_data is None or gold_plot_data is None:
                    score = 0
                else:
                    score = plot_data_eval(llm_plot_data, gold_plot_data)
                    print(f"Plot data eval score: {score}")
        else:
            score = 0  # Unknown eval type

        if agent == "plot":
            results.append({
                "question": question,
                "eval_type": eval_type,
                "score": score,
                "llm_code": llm_result_dict.get("code", ""),
                "gold_code": gold_code,
                "llm_result_preview": str(llm_plot_data)[:2000],
                "gold_result_preview": str(gold_plot_data)[:2000]
            })
        else:
            results.append({
                "question": question,
                "eval_type": eval_type,
                "score": score,
                "llm_code": llm_result_dict.get("code", ""),
                "gold_code": gold_code,
                "llm_result_preview": str(llm_result)[:200],
                "gold_result_preview": str(gold_result)[:200]
            })

        time.sleep(2) # Pause between requests to avoid rate limits

    # Make final df with results
    df_results = pd.DataFrame(results)

    if not df_results.empty:
        print("Benchmark Results Summary:")
        print(df_results.groupby('eval_type')['score'].mean().round(3))
        print("===================================\n")

    return df_results


def ensure_text_file(path="text.txt", default_content="test"):
    """Create text.txt with default_content if it doesn't exist.
    This is for the purpose of testing an adversarial benchmarking question
    that asks the agents to delete text.txt """
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(default_content)
        

def main(n=5, benchmark_path="data/benchmark_questions_111025.xlsx", benchmark="query", model_name="cogito-2.1:671b-cloud"):
    '''
    Main function to run benchmark n times on a set of cases.
    Save each run's results to a separate sheet of an Excel file.
    So, each model will have its own Excel file with multiple sheets.
    Args:
        n: number of benchmark runs
        benchmark_path: path to Excel file with benchmark cases
        benchmark: which benchmark sheet to use ("query", "stats", "plot")
        model_name: LLM model to test
    '''
    # Assign sheet name based on benchmark, and temperature for each module LLM
    if benchmark == "query":
        sname = "table_qa_benchmark"
        agent = "query"
        temp = 0.0
    elif benchmark == "stats":
        sname = "statistics_benchmark"
        agent = "stats"
        temp = 0.1
    elif benchmark == "plot":
        sname = "plotting_benchmark"
        agent = "plot"
        temp = 0.5

    # Read in set of benchmark cases
    df = pd.read_excel(benchmark_path, sheet_name=sname)

    # Normalize col names
    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

    # Build list of benchmark cases
    benchmark_cases = []

    for idx, row in df.iterrows():
        case = {
            "question": row['question'],
            "gold_code": str(row['answer']).strip(),
            "eval_type": row['category'].strip().lower()
        }
        benchmark_cases.append(case)

    # Run benchmark agent n times and save to Excel
    with pd.ExcelWriter(f"results/benchmark_{benchmark}_results_{model_name.replace(':','_')}.xlsx") as writer:
        for i in range(n):
            print(f"Starting benchmark run {i+1}/{n} for model {model_name}...")
            results = benchmark_agent(benchmark_cases, model_name, agent, temp)
            results.to_excel(writer, sheet_name=f"run_{i+1}", index=False)
            print(f"Completed run {i+1}/{n}.\n")
            ensure_text_file("text.txt", "test")
        

# Main flow to iterate through all models 
if __name__ == "__main__":
    models_to_test = ["devstral-2:123b-cloud","gpt-oss:20b-cloud","gpt-oss:120b-cloud","qwen3-coder:480b-cloud",
                      "gemma3:27b-cloud","deepseek-v3.1:671b-cloud","glm-4.6:cloud","cogito-2.1:671b-cloud",
                      "minimax-m2:cloud","kimi-k2:1t-cloud","deepseek-v3.2:cloud","glm-4.7:cloud","mistral-large-3:675b-cloud",
                       "minimax-m2.1:cloud","gemini-3-flash-preview:cloud"]

    benchmark_set = "stats"  # Choose from "query", "stats", "plot"

    for model in models_to_test:
        main(n=5, benchmark_path="data/benchmark_questions_111025.xlsx", benchmark=benchmark_set, model_name=model)
        print(f"Completed {benchmark_set} benchmarking for model: {model}")


