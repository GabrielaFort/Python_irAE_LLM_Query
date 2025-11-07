# Class for agent that generates and executes plotting code based on user question classification
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import traceback
from scipy import stats
from src.utils import clean_code
import plotly.express as px
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly
from matplotlib_venn import venn2, venn3
from collections import Counter

class PlotAgent:
    """
    Handles visualization requests.
    Uses LLM to suggest a plotly plot based on dataset summary and user query.
    Returns a plotly figure object to display in streamlit frontend.
    """
    def __init__(self, df, llm_client):
        self.df = df
        self.llm_client = llm_client

    def handle(self, question, df_summary):
        prompt = f"""
        You are a python plotting assistant. Given the dataframe summary below, write Python code using using plotly to create an interactive plot that answers the user's question. 
        
        -- Use the dataframe variable name 'df'.
        -- Assign the final plotly figure object to a variable named 'result'.
        -- Use only pandas (pd), numpy (np), plotly.express (px), plotly.graph_objects (go), and scipy.stats (stats). Do NOT import anything else!
        -- The plot must contain meaningful data (avoid empty or all-NaN plots).
        -- You may create temporary variables but do NOT modify or overwrite the original 'df'.
        -- Some columns may contain multiple comma-separated values per record. When this is relevant, split those values using `str.split(r'\\s*,\\s*')` and use `explode()` to analyze them properly.
        -- Choose an appropriate chart type based on the question (e.g., line, bar, box, scatter, pie, donut, venn diagram, etc).
        -- If the requested plot is not possible given the dataframe schema, assign a polite explanatory string to 'result' instead of plotting.
        -- The data may contain missing values (NaNs). Handle them safely by excluding missing entries. Do NOT fill, impute, or alter data values.
        -- When creating bar plots of counts or value frequencies, always convert Series or Counter outputs to a DataFrame using `reset_index()`, and refer to column names in Plotly Express (e.g., `px.bar(df, x='col', y='count')`).
        -- For overlap or co-occurrence plots, you may use matplotlib_venn to create a static Venn diagram. 
                - Import from `matplotlib_venn import venn2` (or venn3 if needed). 
                - Assign the resulting Matplotlib Axes object to 'result'.
                - Do not attempt to recreate Venns in Plotly.
        -- Output only executable Python code—no markdown or explanations.

        {df_summary}

        Question: "{question}"
        """

        # Generate and clean up code
        code = self.llm_client.generate(prompt).strip()
        code = clean_code(code)

        return code
    
    def execute_code(self, code):
        # Safe execution of LLM-suggested plotting code
        # Returns figure or text result

        # Restrict variables accessible during execution
        try:
            safe_locals = {"df": self.df.copy()}
            safe_globals = {"pd":pd, "np":np, "stats":stats, "plt":plt, "sns":sns, "px":px, "go":go, "venn2":venn2, "venn3":venn3, "Counter":Counter, "__builtins__":__builtins__} 

            # Execute the generated code
            exec(code, safe_globals, safe_locals)

            # Retrieve result if defined
            result = safe_locals.get("result", None)

            # If 'result' is a plotly figure
            if hasattr(result, "to_plotly_json"): 
                return {
                    "type": "plotly",
                    "code": code,
                    "data": result}

            # If 'result' is a matplotlib Figure or Axes
            elif isinstance(result, plt.Figure) or hasattr(result, "figure"):
                fig = result.figure if hasattr(result, "figure") else result
                # Check to see if venn diagram, if so keep as matplotlib object
                if "matplotlib_venn" in code.lower() or "venn2" in code.lower() or "venn3" in code.lower():
                    plt.close(fig)
                    return {"type": "plot", "code": code, "data": fig}
                # Else, convert to plotly for interactive display
                plotly_fig = mpl_to_plotly(fig)
                plt.close(fig)
                return {"type": "plotly", "code": code, "data": plotly_fig}
            

            # If 'result' is a string (e.g., polite note)
            elif isinstance(result, str):
                return {
                    "type": "text",
                    "code": code,
                    "data": result
                    }
            
            else:
                return {
                    "type": "text",
                    "code": code, 
                    "data": "No valid figure or text result returned."
                }

        except Exception as e:
            err = traceback.format_exc()

            return {
                "type": "error",
                "code": code, 
                "data": f"Error creating plot: {e}\n\n{err}"
            }
