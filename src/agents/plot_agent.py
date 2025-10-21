# Class for agent that generates and executes plotting code based on user question classification
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import traceback
from src.utils import clean_code

class PlotAgent:
    """
    Handles visualization requests.
    Uses LLM to suggest a seaborn plot based on dataset summary and user query.
    If requested plot/data is not possible given table schema, respond with a polite note saying so.
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
        The code must not include plt.show().
        Always ensure the plot has meaningful data (avoid empty plots).
        When plotting categories, rotate x-axis labels if many categories exist.
        When showing overlap (e.g. Venn diagrams), compute intersections properly using sets of patient IDs, not counts.

        {df_summary}

        Question: "{question}"
        """

        # Generate and clean up code
        code = self.llm_client.generate(prompt).strip()
        code = clean_code(code)

        return code
    
    def execute_code(self, code):

        # Safe execution of LLM-suggested plotting code
        try:
            local_vars = {"df": self.df, "sns": sns, "plt": plt, "pd": pd, "np": np}

            # Execute the LLM-suggested seaborn plot command
            exec(code, {}, local_vars)

            fig = plt.gcf()

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
