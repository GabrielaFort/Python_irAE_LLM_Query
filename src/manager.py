# This class will manage different agents and route questions accordingly
# It uses the QuestionClassifier to determine the type of question

from question_classifier import QuestionClassifier
from utils import llama3_llm, qwen_plotter_llm, qwen_query_llm, summarize_dataframe
from agents import QueryAgent, PlotAgent, ErrorAgent
import traceback

class Manager:
    def __init__(self,df):
        self.df = df

        # Instantiate LLM clients for classifier and agents
        self.classifier = QuestionClassifier(llama3_llm())
        self.query_agent = QueryAgent(df,qwen_query_llm())
        self.plot_agent = PlotAgent(df,qwen_plotter_llm())
        self.error_agent = ErrorAgent(qwen_query_llm())

        # Summarize the dataframe - will need every time we instantiate this class
        self.df_summary = summarize_dataframe(df)

    def process_question(self, question):
        # Classify the question
        qtype = self.classifier.classify(question)

        if qtype == "query":
            agent = self.query_agent
        elif qtype == "plot":
            agent = self.plot_agent
        else:
            return({"type": "text",
                      "code": None,
                      "data": "Sorry, I couldn’t classify that question."})
        
        # Try to handle the question with the selected agent
        try:
            code = agent.handle(question, self.df_summary)
            result = agent.execute_code(code)
            
            if result["type"] == "error":
                # If there was an error, use the ErrorAgent to fix the code
                corrected_code = self.error_agent.handle(
                    question,
                    result["data"],
                    result["code"],
                    self.df_summary
                )
                # Re-execute the corrected code
                retry_result = agent.execute_code(corrected_code)

                if retry_result["type"] == "error":
                    retry_result["data"] = (
                        "Automatic correction failed:\n\n"
                        + retry_result["data"]
                        + "\n\n"
                        + "(You can review the fixed code above.)"
                    )

                return retry_result
            
            return result 

        except Exception as e:  
            return {
                "type": "error",
                "data": f"Fatal Manager error:\n\n{traceback.format_exc()}"
            }           
        


    




