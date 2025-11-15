# Class for statistics expert agent
import pandas as pd 
import numpy as np
from scipy import stats
import traceback
from src.utils import clean_code
from collections import Counter

class StatsAgent:
    """
    Handles statistical analysis requests. 
    Uses LLM to suggest statistical tests in python code based on dataset sumary and user question.
    Returns a tabular or numeric result.
    """

    def __init__(self, df, llm_client):
        self.df = df
        self.llm_client = llm_client

    def handle(self, question, df_summary):
        prompt = f"""
        You are a python statistical analysis assistant.
        Given the dataframe summary below, write executable Python code using pandas, numpy, or scipy.stats to perform the appropriate statistical test or summary that answers the user's question.

        **Rules**        
        - Use the dataframe variable name 'df'.
        - Assign the final output to a variable named 'result'.
        - Do **not** include any import statements — assume `pandas (pd)`, `numpy (np)`, and `scipy.stats (stats)` are already imported.
        - Only use columns and data types shown in the summary. Do **not** assume any others.
        - You may create temporary DataFrames, Series, or variables for intermediate calculations, but never modify or overwrite 'df'.

        **Data Handling**
        - Handle comma-separated values using `str.split(r'\\s*,\\s*', regex=True)` or `explode()`.
        - If exploding multiple columns, pad shorter lists with `None` before exploding.
        - Handle missing values safely by excluding them; never impute or fill values.
        - For numeric columns, automatically convert to float where needed for statistical testing.

        **Output**
        - Always include relevant test statistics (e.g., t, r, χ²), degrees of freedom (if applicable), and p-values.
        - Always store the final output in a pandas DataFrame called `result` containing all relevant test outcomes or summary statistics.
        - If the analysis cannot be performed with the given schema, assign a short explanatory string to `result` instead.
        - Output **only** executable Python code — no markdown, comments, or explanations.

        {df_summary}

        Question: "{question}"
        """

        # Generate and clean up code
        code = self.llm_client.generate(prompt)
        code = clean_code(code)

        return code

    def execute_code(self, code):
        # Executes LLM-generated statistical code safely and returns the result

        # Restrict variables accessible during execution
        try:
            
            # Provide a copy of the dataframe to avoid modifications
            safe_locals = {"df": self.df.copy()} 
            safe_globals = {"pd": pd, "np": np, "stats": stats, "Counter": Counter, "__builtins__": __builtins__}

            # execute the generated code (it should assign the output to variable `result`)
            exec(code, {**safe_globals, **safe_locals}, safe_locals)

            result = safe_locals.get("result", None)

            if result is None:
                return {
                    "type": "error",
                    "code": code,
                    "data": "No variable named 'result' found. Please ensure your code assigns output to 'result'."
                }
            
            # Handle other result types
            elif isinstance(result, (int, float, np.number, np.generic)):
                display_data = float(result)
                type_str = "number"
            elif isinstance(result, pd.DataFrame):
                display_data = result
                type_str = "dataframe"
            elif isinstance(result, pd.Series):
                # Convert series to DF with index levels 
                display_data = result.reset_index()
                value_col = result.name if result.name else "value"
                index_cols = [
                    c if (isinstance(c, str) and c.strip() != "") else f"index_{i}"
                    for i, c in enumerate(result.index.names or [])
                ]
                # If the Series has no index names, assign generic ones
                if not index_cols or any(name is None for name in index_cols):
                    index_cols = [f"index_{i}" for i in range(display_data.shape[1] - 1)]
                display_data.columns = index_cols + [value_col]
                type_str = "dataframe"
            else:
                display_data = str(result)
                type_str = "text"

            return {"type": type_str,
                    "code": code,
                    "data": display_data}
        
        except Exception as e:

            err = traceback.format_exc()

            return {
                "type": "error",
                "code": code,
                "data": f"Error executing statistical code: {e}\n\n{err}"
                }
