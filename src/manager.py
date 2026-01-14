# This class will manage different agents and route questions accordingly
# It uses the QuestionClassifier to determine the type of question

from src.question_classifier import QuestionClassifier
from src.utils import question_classifier_llm, plotter_llm, query_llm, stats_llm, error_checker_llm, guideline_llm, summarize_dataframe
from src.agents import QueryAgent, PlotAgent, StatsAgent, ErrorAgent, GuidelineAgent
from src.index_manager import IndexManager
import traceback
import numpy as np
from sentence_transformers import SentenceTransformer

class Manager:
    def __init__(self,df):
        self.df = df

        # Initialize IndexManager and load index + metas
        self.index_manager = IndexManager(kb_dir="src/knowledge_base", model_name="NeuML/pubmedbert-base-embeddings")
        try:
            self.index_manager.load()
        except FileNotFoundError:
            raise RuntimeError("Knowledge base not found. Run build_pdf_kb.py to build the index.")

        # export search_fn and embed_fn
        self.search_fn = self.index_manager.search
        self.embed_fn = self.index_manager.embed_fn

        # Instantiate LLM clients for classifier and agents
        self.classifier = QuestionClassifier(question_classifier_llm())
        self.query_agent = QueryAgent(df,query_llm())
        self.plot_agent = PlotAgent(df,plotter_llm())
        self.stats_agent = StatsAgent(df,stats_llm())
        # Pass search_fn (and optionally embed_fn) to GuidelineAgent
        self.guideline_agent = GuidelineAgent(guideline_llm(), search_fn=self.search_fn, embed_fn=self.embed_fn)
        self.error_agent = ErrorAgent(error_checker_llm())

        # Summarize the dataframe - will need every time we instantiate this class
        self.df_summary = summarize_dataframe(df)

    def process_question(self, question, context=None):
        # Classify the question
        qtype = self.classifier.classify(question, context=context)
        print(qtype)

        if qtype == "tableqa":
            agent = self.query_agent
        elif qtype == "plot":
            agent = self.plot_agent
        elif qtype == "stats":
            agent = self.stats_agent
        elif qtype == "guideline":
            return self.guideline_agent.handle(question, context=context)
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
        


    




