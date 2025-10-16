# This class will manage different agents and route questions accordingly
# It uses the QuestionClassifier to determine the type of question

from question_classifier import QuestionClassifier
from utils import llama3_llm, qwen_plotter_llm, qwen_query_llm, summarize_dataframe
from agents import QueryAgent, PlotAgent

class Manager:
    def __init__(self,df):
        self.df = df

        # Instantiate LLM clients for classifier and agents
        self.classifier = QuestionClassifier(llama3_llm())
        self.query_agent = QueryAgent(df,qwen_query_llm())
        self.plot_agent = PlotAgent(df,qwen_plotter_llm())

        # Summarize the dataframe - will need every time we instantiate this class
        self.df_summary = summarize_dataframe(df)

    def process_question(self, question):
        # Classify the question
        qtype = self.classifier.classify(question)

        if qtype == "query":
            result = self.query_agent.handle(question, self.df_summary)
        elif qtype == "plot":
            result = self.plot_agent.handle(question, self.df_summary)

        else:
            result = {"type": "text",
                      "code": None,
                      "data": "Sorry, I couldn’t classify that question."}
        
        return result
    




