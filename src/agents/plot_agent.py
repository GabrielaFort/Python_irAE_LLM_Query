# Class for agent that generates and executes plotting code based on user question classification
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import traceback
from utils import clean_code

class PlotAgent:
    """
    Handles visualization requests.
    Uses LLM to suggest a seaborn plot based on dataset summary and user query.
    Returns a matplotlib fig object to display in streamlit frontend.
    """
    def __init__(self, df, llm_client):
        self.df = df
        self.llm_client = llm_client

        # Set seaborn theme for plots
        sns.set_theme(style="whitegrid",palette="muted")

    def handle(self, question, df_summary):
        prompt = f"""
        You are a python plotting assistant. Given the following dataframe summary, 
        generate python code using seaborn to create a plot that answers the user's question.
        Use the dataframe variable name 'df'. Do not include any explanations, only return the code.

        {df_summary}

        Question: "{question}"

        The plot must be simple, self-contained, and should not include plt.show().
        """

        # Generate and clean up code
        code = self.llm_client.generate(prompt).strip()
        code = clean_code(code)

        return code
    
    def execute_code(self, code):

        # Safe execution of LLM-suggested plotting code
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            local_vars = {"df": self.df, "sns": sns, "plt": plt, "ax": ax, "pd": pd, "np": np}

            # Execute the LLM-suggested seaborn plot command
            exec(code, {}, local_vars)

            # Save plot to a memory buffer
            plt.xticks(rotation=45, fontsize=8, ha='right')
            plt.yticks(fontsize=8)
            plt.tight_layout()


            return {"type": "plot",
                    "code": code,
                    "data": fig}

        except Exception as e:
            err = traceback.format_exc()

            return {
                "type": "error",
                "code": code, 
                "data": f"Error creating plot: {e}\n\n{err}"
            }
