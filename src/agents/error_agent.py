# Class for agent that fixes code that raised an error
from src.utils import clean_code

class ErrorAgent:
    def __init__ (self, llm_client):
        self.llm_client = llm_client

    def handle(self, question, error_message, original_code, df_summary):
        prompt = f"""
You are a python debugging assistant. Given the users question, dataframe summary,
the original python code, and the error traceback, fix the code so it runs successfully 
and still answers the question. Return ONLY valid, correct python code using the variable
'df' for the dataframe. Do not include any explanations, only return the corrected code.

User Question:
{question}

Original Code:
{original_code}

Error Traceback:
{error_message}

Dataframe Summary:
{df_summary}

/no_think
            """

        # Generate and clean up code
        corrected_code = self.llm_client.generate(prompt)
        corrected_code = clean_code(corrected_code)
        return corrected_code