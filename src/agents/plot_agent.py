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
from matplotlib_venn._common import VennDiagram
from matplotlib_venn import venn2, venn3
from collections import Counter

class PlotAgent:
    """
    Handles visualization requests.
    Uses LLM to suggest a plotly plot based on dataset summary and user query.
    Returns a plotly figure object to display in streamlit frontend.
    Uses conversation history for context
    """
    def __init__(self, df, llm_client):
        self.df = df
        self.llm_client = llm_client

    def handle(self, question, df_summary, messages=None):

        if messages is None:
            messages = []

        # Build system prompt
        system_prompt = f"""
        You are a python plotting assistant.
        Given the dataframe summary below, write Python code using using plotly to create an informative, interactive plot that answers the user's question. 
        
        **Rules**
        - Use the dataframe variable name 'df'.
        - Assign the final figure object to a variable named 'result'.
        - In addition, create a pandas DataFrame named `plot_data` that contains the exact data used to construct the plot (one row per point/bar/etc).
        - Do **not** include any import statements — assume `pandas (pd)`, `numpy (np)`, `plotly.express (px)`, `plotly.graph_objects (go)`, `scipy.stats (stats)` are already imported.
        - Only use columns and data types shown in the summary. Do **not** assume any others.
        - You may create temporary variables, but never modify or overwrite 'df'.
        - When splitting comma-separated values, ALWAYS assign and explode within the SAME DataFrame (e.g., df_copy['col'] = df_copy['col'].str.split(...); df_copy = df_copy.explode('col')). Never create a separate exploded Series and assign it back.
        - All axis update methods must use Plotlys plural API: use update_xaxes(), update_yaxes(). Never use update_xaxis or update_yaxis.
        - All plots must use only px.* or go.*; do not use matplotlib unless producing a Venn diagram.
        - **SAFETY**: Never include code that writes to disk or removes files, accesses the network, or executes system commands.

        **Data Handling**
        - Always handle comma-separated values using `str.split(r'\\s*,\\s*', regex=True)` and `explode()`.
        - Handle missing values safely by excluding them; never impute or alter data.
        - Ensure the plot contains meaningful data — avoid empty or all-NaN visualizations.

        **Special Cases**
        - For overlap or co-occurrence plots:
            - You may import `venn2` or `venn3` from `matplotlib_venn`.
            - Assign the resulting Matplotlib Axes object to `result` instead of a Plotly figure.
            - Do **not** attempt to recreate Venn diagrams using Plotly.

        **Output**
        - Assign the final Plotly (or Matplotlib) figure to the variable `result` and the intermediate DataFrame to `plot_data`.
        - Do not produce textual summaries or markdown.
        - If the requested plot is not feasible given the schema, assign a short explanatory string to `result` instead.
        - Output **only** executable Python code — no markdown, comments, or explanations.

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
        # Safe execution of LLM-suggested plotting code
        # Returns figure or text result

        # Restrict variables accessible during execution
        try:
            #safe_locals = {"df": self.df.copy()}
            #safe_globals = {"pd":pd, "np":np, "stats":stats, "plt":plt, "sns":sns, "px":px, "go":go, "venn2":venn2, "venn3":venn3, "Counter":Counter, "__builtins__":__builtins__} 
            safes = {"pd": pd, "np": np, "stats": stats, "sns":sns, "px":px, "go":go, "plt":plt,"venn2":venn2, "venn3":venn3, "Counter": Counter, "__builtins__": __builtins__,"df" : self.df.copy()} 

            # execute the generated code (it should assign the output to variable `result`)
            exec(code, safes)
            # Execute the generated code
            #exec(code, {**safe_globals, **safe_locals}, safe_locals)

            # Retrieve result if defined
            result = safes.get("result", None)
            plot_data = safes.get("plot_data", None)

            # If 'result' is a plotly figure
            if hasattr(result, "to_plotly_json"): 
                return {
                    "type": "plotly",
                    "code": code,
                    "data": result,
                    "plot_data": plot_data}

            # If 'result' is a matplotlib Figure or Axes
            elif isinstance(result, plt.Figure) or hasattr(result, "figure"):
                fig = result.figure if hasattr(result, "figure") else result
                # Check to see if venn diagram, if so keep as matplotlib object
                if "matplotlib_venn" in code.lower() or "venn2" in code.lower() or "venn3" in code.lower():
                    plt.close(fig)
                    return {"type": "plot", "code": code, "data": fig, "plot_data": plot_data}
                # Else, convert to plotly for interactive display
                plotly_fig = mpl_to_plotly(fig)
                plt.close(fig)
                return {"type": "plotly", "code": code, "data": plotly_fig, "plot_data": plot_data}
            
            elif isinstance(result, VennDiagram):
                fig = result.figure
                plt.close(fig)
                return {"type": "plot", "code": code, "data": fig, "plot_data": plot_data}
            
            # If 'result' is a string (e.g., polite note)
            elif isinstance(result, str):
                return {
                    "type": "text",
                    "code": code,
                    "data": result,
                    "plot_data": plot_data}
            
            else:
                return {
                    "type": "text",
                    "code": code, 
                    "data": "No valid figure or text result returned.",
                    "plot_data": plot_data
                }

        except Exception as e:
            err = traceback.format_exc()

            return {
                "type": "error",
                "code": code, 
                "data": f"Error creating plot: {e}\n\n{err}",
                "plot_data": None
            }
