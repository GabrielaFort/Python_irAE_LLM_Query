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

    def handle(self, question, df_summary, messages=None):
        """
        Handle query questions with conversation history.
        Args:
            question: User's current question
            messages: Previous conversation messages
            df_summary: Summary of df schema from summarize_dataframe
        Returns:
            Dict with type, data, and code.
        """
        if messages is None:
            messages = []

        # Prepare system prompt
        system_prompt = f"""
        You are a python data analysis assistant.
        Given the dataframe summary below, write executable python code using pandas to answer the user's question.
        
        **Rules**
        - Use dataframe name 'df'.
        - Assign the final answer to 'result'.
        - Do **not** include any import statements — assume `pandas (pd)`, `numpy (np)`, and `scipy.stats (stats)` are already imported.
        - Only use columns and data types shown in the summary. Do **not** assume others. 
        - You may create temp DataFrames/Series/columns but never modify 'df'.
        - Handle comma-separated values using `str.split(r'\\s*,\\s*', regex=True)` or `explode()`.
        - If exploding multiple columns, pad shorter lists with `None` before exploding.
        - Prefer using `.str.contains(..., case=False, na=False)` for text matching — particularly for comma-separated values.
        - Exclude missing values safely, but do not drop rows unless necessary.
        - Unless explicitly instructed otherwise, always prefer returning a DataFrame that preserves the full schema of the original data (same column names, same order) — especially for
          filtering or lookup-style questions.
        - For counts/comparisons, group logically and use '.nunique()' where appropriate.
        - Ensure all column names are unique to avoid errors.
        - **SAFETY**: Never include code that writes to disk or removes files, accesses the network, or executes system commands.

        **Output**
            - a DataFrame subset/summary that answers the question, **or**
            - a single numeric/scalar value if suitable.
        - If unanswerable from schema, assign a short explanatory string to 'result'.
        - Output **only** executable **Python code**, no markdown or explanations.

        {df_summary}
        """

        # Build messages for LLM
        full_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        full_messages.extend(messages)

        # Add current user question 
        full_messages.append({"role": "user", "content": question})

        # Generate and clean up code
        code = self.llm_client.generate(messages=full_messages)
        code = clean_code(code)

        return code
    
    def execute_code(self, code):
        # Executes LLM-generated query code safely and returns the result
        
        # Restrict variables accessible during execution
        try:

            # Provide a copy of the dataframe to avoid modifications
            safes = {"pd": pd, "np": np, "stats": stats, "Counter": Counter, "__builtins__": __builtins__,"df" : self.df.copy()} 
            #safe_locals = {"df" : self.df.copy()}

            # execute the generated code (it should assign the output to variable `result`)
            exec(code, safes)

            result = safes.get("result", None)

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
                display_data = result.reset_index(drop=True).to_frame()
                display_data = display_data.replace("", pd.NA)
                type_str = "dataframe"

            # Lists and NumPy arrays
            elif isinstance(result, np.ndarray):
                if result.ndim == 1:
                    display_data = pd.DataFrame(result, columns=["value"])
                    display_data = display_data.replace("", pd.NA)
                    type_str = "dataframe"
                else:
                    display_data = pd.DataFrame(result)
                    display_data = display_data.replace("", pd.NA)
                    type_str = "dataframe"

            elif isinstance(result, list):
            # Convert simple lists to a single-column DataFrame
                try:
                    arr = np.array(result, dtype=object)
                    if arr.ndim == 1:
                        display_data = pd.DataFrame(arr, columns=["value"])
                        display_data = display_data.replace("", pd.NA)
                        type_str = "dataframe"
                    else:
                        display_data = pd.DataFrame(arr)
                        display_data = display_data.replace("", pd.NA)
                        type_str = "dataframe"
                except Exception:
                    display_data = pd.DataFrame(result, columns=["value"])
                    display_data = display_data.replace("", pd.NA)
                    type_str = "dataframe"

            elif isinstance(result, pd.Index):
                display_data = result.tolist()
                display_data = pd.DataFrame(display_data, columns=["value"])
                display_data = display_data.replace("", pd.NA)
                type_str = "dataframe"

            elif isinstance(result, set):
                display_data = pd.DataFrame(sorted(result), columns=["value"])
                display_data = display_data.replace("", pd.NA)
                type_str = "dataframe"
            
            elif isinstance(result, dict):
                display_data = pd.DataFrame(result)
                display_data = display_data.replace("", pd.NA)
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