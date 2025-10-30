class QuestionClassifier:
    """
    Classifies user questions into categories: 'query', 'plot', or 'stats'.
    Uses keyword detection and falls back to LLM classification if needed.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def classify(self, question):
        plot_keywords = ["plot", "graph", "chart", "visualize", "histogram", "scatter", "bar", "line", "figure", "volcano","heatmap", "piechart", "pie chart","donut"]
        stat_keywords = ["statistical", "correlation", "regression", "significant", "p-value", "anova", "t-test", "chi-square", "average","median","mean"]
        query_keywords = ["how many", "count", "number of", "what is the total", "total number of", "list","find"]
        question_lower = question.lower()

        # Keyword-based detection
        if any(word in question_lower for word in plot_keywords):
            return "plot"
        elif any(word in question_lower for word in stat_keywords):
            return "stats"
        elif any(word in question_lower for word in query_keywords):
            return "query"
        else:
            # Fallback to LLM classification
            prompt = f"""
            Classify the task type of this question: "{question}"
            Choose one of: 'query', 'stats', or 'plot'. Return only the category name.
            /no_think
            """
            
            classification = self.llm_client.generate(prompt).strip().lower()
            if classification in ["query", "plot", "stats"]:
                return classification

            # Default to 'query' if unsure
            return "query"