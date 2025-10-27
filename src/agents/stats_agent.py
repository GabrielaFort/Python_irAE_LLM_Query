# Class for statistics expert agent
import pandas as pd 
import numpy as np
from scipy import stats
import traceback
from src.utils import clean_code

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
        You are a python statistical analysis assistant. Given the dataframe summary below, write Python code using pandas, numpy, or scipy.stats to perform the appropriate statistical test that answers the user's question.
        
        -- Use the dataframe variable name 'df'.
        -- Assign the final result (e.g., test statistic, p-value, correlation coefficient, or summary table) to a variable named 'result'.
        -- Base your analysis only on the columns and data types shown in the dataframe summary. Do NOT assume any missing columns.
        -- You may create temporary variables for intermediate calculations, but do NOT modify, overwrite, or alter the original 'df'.
        -- Some columns may contain multiple comma-separated values per record. When this is relevant, split those values using `str.split(r'\\s*,\\s*')` and use `explode()` to analyze them properly.
        -- Choose an appropriate statistical method based on the question (e.g., t-test, chi-square, ANOVA, correlation, regression).       
        -- Include relevant summary statistics if helpful (means, counts, etc.) but keep the output concise and numeric/textual — no plots.
        -- If the request cannot be answered given the dataframe schema, assign a polite explanatory string to 'result' instead.
        -- Use only pandas (pd), numpy (np), and scipy.stats (stats). Do NOT import anything else.
        -- The data may contain missing values (NaNs). Handle them safely by excluding missing entries. Do NOT fill, impute, or alter data values.
        -- Output only executable Python code - no markdown or explanations.

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
            safe_globals = {"pd": pd, "np": np, "stats": stats, "__builtins__": __builtins__}

            # execute the generated code (it should assign the output to variable `result`)
            exec(code, safe_globals, safe_locals)

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
                display_data = result.to_frame()
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
