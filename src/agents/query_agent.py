# Class for agent that generates and executes query code based on user question classification
import pandas as pd
import numpy as np
import traceback
from src.utils import clean_code
from scipy import stats
from collections import Counter


class QueryAgent:
    def __init__(self, df, llm_client):
        self.df = df
        self.llm_client = llm_client

    def handle(self, question, df_summary):
        prompt = f"""
        You are a python data analysis assistant. Given the dataframe summary below, generate python code using pandas to answer the user's question.
        
        -- Use the dataframe variable name 'df'.
        -- CRITICAL: Assign the answer to a variable named 'result'.
        -- Base your code only on the columns and data types shown in the dataframe summary. Do NOT assume missing columns.
        -- You may create new temporary DataFrames or Series to compute the answer, but do NOT modify, overwrite, or alter the original 'df'.
        -- Some columns may contain multiple comma-separated values per record. When this is relevant:
            - split those values using `str.split(r'\s*,\s*')` and use `explode()` to analyze them properly.
            - If exploding multiple columns, ensure all have equal-length lists (pad shorter ones with None before exploding).
        -- If the query involves counts or comparisons across groups, make sure to group logically and use `.nunique()` for distinct items where appropriate.
        -- Return the full dataframe subset or summary that answers the question, or a numeric/scalar value if appropriate.
        -- If the query cannot be answered given the schema, assign a polite explanatory string to 'result' instead.
        -- You may only use pandas (pd), numpy (np), and scipy.stats (stats). Do NOT import anything else.
        -- The data may contain missing values (NaNs). Handle them safely by excluding missing entries. Do NOT fill, impute, or alter data values.
        -- Output only executable Python code—no markdown or explanations.

        {df_summary}
 
        Question: "{question}"

        /no_think
        """

        # Generate and clean up code
        code = self.llm_client.generate(prompt)
        code = clean_code(code)

        return code
    
    def execute_code(self, code):
        # Executes LLM-generated query code safely and returns the result
        
        # Restrict variables accessible during execution
        try:

            # Provide a copy of the dataframe to avoid modifications
            safe_globals = {"pd": pd, "np": np, "stats": stats, "Counter": Counter, "__builtins__": __builtins__} 
            safe_locals = {"df" : self.df.copy()}

            # execute the generated code (it should assign the output to variable `result`)
            exec(code, safe_globals, safe_locals)
            result = safe_locals.get("result", None)

            if result is None:
                return {
                    "type": "error",
                    "code": code,
                    "data": "No variable named 'result' found. Please ensure your code assigns output to 'result'."
                }
            
            # Handle potential other result types 
            elif isinstance(result, (int, float, np.number, np.generic)):
                display_data = float(result)
                type_str = "number"
            elif isinstance(result, pd.DataFrame):
                display_data = result.replace("", pd.NA)
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

            return {"type": "error",
                    "code": code,
                    "data": f"Error executing query: {e}\n\n{err}"
                    }