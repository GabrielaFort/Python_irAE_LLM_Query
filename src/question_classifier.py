class QuestionClassifier:
    """
    Classifies user questions into categories: 'query' or 'plot'.
    Uses LLM to interpret question if no heuristic matches are found.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def classify(self, question):
        plot_keywords = ["plot", "graph", "chart", "visualize", "histogram", "scatter", "bar", "line", "figure", "box","volcano","heatmap", "piechart", "pie chart","donut"]
        stat_keywords = ["statistical", "correlation", "regression", "significant", "p-value", "anova", "t-test", "mann-whitney","wilcoxon","chi-square", "average","median","mean","standard deviation","confidence interval"]
        query_keywords = ["how many", "number of", "what is the total", "total number of", "list all","list unique","find all","retrieve","get all"]
        question_lower = question.lower()

        # Keyword-based detection
        if any(word in question_lower for word in plot_keywords):
            return "plot"
        elif any(word in question_lower for word in stat_keywords):
            return "stats"
        elif any(word in question_lower for word in query_keywords):
            return "tableqa"

        else:
            # Fallback to LLM classification
            prompt = f"""
            Classify the task type of this question: "{question}"
            Choose one of: 'tableQA', 'stats', or 'plot'. Return only the category name.
            """
            
            classification = self.llm_client.generate(prompt).strip().lower()
            if classification in ["tableqa", "plot", "stats"]:
                return classification

            # Default to 'tableqa' if unsure
            return "tableqa"