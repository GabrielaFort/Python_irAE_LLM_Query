# Class for agent that generates and executes query code based on user question classification
import pandas as pd
import numpy as np
import traceback
from src.utils import clean_code

class QueryAgent:
    def __init__(self, df, llm_client):
        self.df = df
        self.llm_client = llm_client

    def handle(self, question, df_summary):
        prompt = f"""
        You are a python data analysis assistant. Given the following dataframe summary, 
        generate python code using pandas to answer the user's question.
        Use the dataframe variable name 'df'. Assign the answer to a variable 'result'.
        CRITICAL: If requested query is not possible given table schema, respond with a polite note saying so.
        Do not include any explanations or markdown, only return the executable code.

        {df_summary}

        Question: "{question}"
        """

        # Generate and clean up code
        code = self.llm_client.generate(prompt)
        code = clean_code(code)

        return code
    
    def execute_code(self, code):

         # Safe execution of LLM-suggested query code
        try:
            local_vars = {"df": self.df, "pd": pd, "np": np}
            # execute the generated code (it should assign the output to variable `result`)
            exec(code, {}, local_vars)
            result = local_vars.get("result", None)

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
                display_data = result.to_frame()
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