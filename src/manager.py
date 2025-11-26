# This class will manage different agents and route questions accordingly
# It uses the QuestionClassifier to determine the type of question

from src.question_classifier import QuestionClassifier
from src.utils import question_classifier_llm, plotter_llm, query_llm, stats_llm, error_checker_llm, summarize_dataframe
from src.agents import QueryAgent, PlotAgent, StatsAgent, ErrorAgent
import traceback

class Manager:
    def __init__(self,df):
        self.df = df

        # Instantiate LLM clients for classifier and agents
        self.classifier = QuestionClassifier(question_classifier_llm())
        self.query_agent = QueryAgent(df,query_llm())
        self.plot_agent = PlotAgent(df,plotter_llm())
        self.stats_agent = StatsAgent(df,stats_llm())
        self.error_agent = ErrorAgent(error_checker_llm())

        # Summarize the dataframe - will need every time we instantiate this class
        self.df_summary = summarize_dataframe(df)

    def process_question(self, question, context=None):
        # Classify the question
        qtype = self.classifier.classify(question, context=context)

        if qtype == "tableqa":
            agent = self.query_agent
        elif qtype == "plot":
            agent = self.plot_agent
        elif qtype == "stats":
            agent = self.stats_agent
        else:
            return({"type": "text",
                      "code": None,
                      "data": "Sorry, I couldn’t classify that question."})
        
        # Try to handle the question with the selected agent
        try:
            code = agent.handle(question, self.df_summary, context=context)
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
                        + "\n"
                    )

                return retry_result
            
            return result 

        except Exception as e:

            err = traceback.format_exc()
              
            return {
                "type": "error",
                "data": f"Fatal Manager error: {e}\n\n{err}"
            }           
        


    




