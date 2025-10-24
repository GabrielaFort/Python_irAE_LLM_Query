# Class for agent that generates and executes plotting code based on user question classification
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import traceback
from scipy import stats
from src.utils import clean_code

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
        You are a python plotting assistant. Given the dataframe summary below, write Python code using seaborn and matplotlib to create a plot that answers the user's question. 
        
        -- Use the dataframe variable name 'df'.
        -- Assign the final plot object (e.g., a matplotlib Figure or Axes) to a variable named 'result'.
        -- Use only the following libraries: pandas (pd), numpy (np), seaborn (sns), matplotlib.pyplot (plt), and scipy.stats (stats). Do NOT import anything else.
        -- Do NOT call plt.show().
        -- The plot must contain meaningful data (avoid empty or all-NaN plots).
        -- You may create temporary variables but do NOT modify or overwrite the original 'df'.
        -- Some columns may contain multiple comma-separated values per record. When this is relevant, split those values using `str.split(r'\\s*,\\s*')` and use `explode()` to analyze them properly.
        -- Rotate x-axis labels if category labels would overlap.
        -- For overlap-type plots (e.g., Venn diagrams), compute intersections using sets of IDs rather than counts.
        -- If the requested plot is not possible given the dataframe schema, assign a polite explanatory string to 'result' instead of plotting.
        -- The data may contain missing values (NaNs). Handle them safely by excluding missing entries. Do NOT fill, impute, or alter data values.
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
            safe_locals = {"df": self.df.copy(), "pd": pd, "np": np, "stats": stats, "plt": plt, "sns": sns} 

            # Execute the generated code
            exec(code, {}, safe_locals)

            # Retrieve result if defined
            result = safe_locals.get("result", None)

            # If 'result' is a matplotlib Figure
            if isinstance(result, plt.Figure):
                fig = result

            # If 'result' is an Axes object, grab its parent Figure
            elif hasattr(result, "figure"):
                fig = result.figure

            # If 'result' is a string (e.g., polite note)
            elif isinstance(result, str):
                return {
                    "type": "text",
                    "code": code,
                    "data": result
                    }
            
            else:
                # If nothing returned, fallback to current figure
                fig = plt.gcf()

            if not fig.axes:
                # If figure is blank, show message instead of empty grid
                return {
                    "type": "text",
                    "code": code,
                    "data": "No plot elements were drawn. The query may not be possible with the given data."    
                }

            # Close the figure after returning it to Streamlit to prevent reuse
            plt.close(fig)

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
