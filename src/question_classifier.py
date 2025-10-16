class QuestionClassifier:
    """
    Classifies user questions into categories: 'query' or 'plot'.
    Uses lightweight LLM to interpret question if no heuristic matches are found.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def classify(self, question):
        plot_keywords = ["plot", "graph", "chart", "visualize", "histogram", "scatter", "bar", "line", "figure", "volcano","heatmap"]
        query_keywords = ["how many", "count", "number of", "what is the total", "total number of", "average", "median", "mean","list"]
        question_lower = question.lower()

        # Keyword-based detection
        if any(word in question_lower for word in plot_keywords):
            return "plot"
        elif any(word in question_lower for word in query_keywords):
            return "query"

        else:
            # Fallback to LLM classification
            prompt = f"""
            Classify the task type of this question: "{question}"
            Choose one of: 'query' or 'plot'. Return only the category name.
            """
            
            classification = self.llm_client.generate(prompt).strip().lower()
            if classification in ["query", "plot"]:
                return classification

            # Default to 'query' if unsure
            return "query"