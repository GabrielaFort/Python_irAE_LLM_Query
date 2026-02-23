# This class will manage different module LLMs and route questions accordingly
# It uses the QuestionClassifier to determine the type of question before routing to appropriate modules.

from src.question_classifier import QuestionClassifier
from src.utils import question_classifier_llm, plotter_llm, query_llm, stats_llm, error_checker_llm, guideline_llm, summarize_dataframe
from src.agents import QueryAgent, PlotAgent, StatsAgent, ErrorAgent, GuidelineAgent
from src.index_manager import IndexManager
import traceback
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Set up logger
logger = logging.getLogger(__name__)

class Manager:
    def __init__(self,df, shared_index_manager=None):
        self.df = df

        # Initialize IndexManager and load index + metas
        # Use shared index manager or create session specific one
        if shared_index_manager is not None:
            self.index_manager = shared_index_manager
        else:
            self.index_manager = IndexManager(kb_dir="src/knowledge_base", model_name="NeuML/pubmedbert-base-embeddings")
            self.index_manager.load()

        # export search_fn and embed_fn
        self.search_fn = self.index_manager.search
        self.embed_fn = self.index_manager.embed_fn

        # Instantiate LLM clients for classifier and agents
        self.classifier = QuestionClassifier(question_classifier_llm())
        self.query_agent = QueryAgent(df.copy(),query_llm())
        self.plot_agent = PlotAgent(df.copy(),plotter_llm())
        self.stats_agent = StatsAgent(df.copy(),stats_llm())
        self.guideline_agent = GuidelineAgent(llm_client = guideline_llm(), search_fn = self.search_fn, embed_fn = self.embed_fn, top_k = 10)
        self.error_agent = ErrorAgent(error_checker_llm())

        # Summarize the dataframe - will need every time we instantiate this class
        self.df_summary = summarize_dataframe(df)

    def process_question(self, question, context=None):
        """
        Process a user question with conversation history.

        Args:
            question (str): The user's question.
            context: List of message dicts from conversation history.
        
        Returns:
            dict: with result type, data, and code.
        """
        if context is None:
            context = []

        # Classify the question
        qtype = self.classifier.classify(question, messages=context)

        if qtype == "tableqa":
            agent = self.query_agent
            logger.info("Question asked and classified as 'tableqa'. Routing to QueryAgent.")
        elif qtype == "plot":
            agent = self.plot_agent
            logger.info("Question asked and classified as 'plot'. Routing to PlotAgent.")
        elif qtype == "stats":
            agent = self.stats_agent
            logger.info("Question asked and classified as 'stats'. Routing to StatsAgent.")
        elif qtype == "guideline":
            logger.info("Question asked and classified as 'guideline'. Routing to GuidelineAgent.")
            return self.guideline_agent.handle(question, messages=context)
        else:
            logger.warning("Question type not recognized. Returning error.", qtype)
            return({"type": "text",
                      "code": None,
                      "data": "Sorry, I couldn’t classify that question. Please try again or rephrase it."})
        
        # Try to handle the question with the selected agent
        try:
            code = agent.handle(question, self.df_summary, messages=context)
            result = agent.execute_code(code)
            
            if result["type"] == "error":
                # If there was an error, use the ErrorAgent to fix the code
                logger.info("Error detected in code execution, invoking ErrorAgent for correction.")
                corrected_code = self.error_agent.handle(
                    question,
                    result["data"],
                    result["code"],
                    self.df_summary
                )
                # Re-execute the corrected code
                retry_result = agent.execute_code(corrected_code)

                #### add all errors to log but only return friendly messsage to user ###
                ### User does not get to see error traceback or raw error message, but we log it for debugging and improvement purposes.
                if retry_result["type"] == "error":
                    logger.error("Automatic correction failed: %s", retry_result["data"])
                    retry_result["data"] = (
                        "Automatic correction failed:\n\n"
                        + retry_result["data"]
                        + "\n"
                    )
                
                if retry_result["type"] != "error":
                    logger.info("Returning result after error correction attempt, no further error detected.")
                
                return retry_result
            
            return result 

        except Exception as e:
            logger.error("Fatal Manager error: %s", e, exc_info=True)
            err = traceback.format_exc()
              
            return {
                "type": "error",
                "data": f"Fatal Manager error: {e}\n\n{err}"
            }           
        


    




