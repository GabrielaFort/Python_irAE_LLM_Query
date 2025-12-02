class QuestionClassifier:
    """
    Classifies user questions into categories: 'query' or 'plot'.
    Uses LLM to interpret question if no heuristic matches are found.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def classify(self, question, context=None):

        if not context:
            plot_keywords = ["plot", "graph", "chart", "histogram", "scatter", "bar", "line", "box","volcano","heatmap", "piechart", "pie chart", "donut"]
            stat_keywords = ["statistical", "correlation", "regression", "significant", "p-value", "anova", "t-test", "mann-whitney","wilcoxon","chi-square", "average","median","mean","standard deviation","confidence interval"]
            query_keywords = ["how many", "number of", "what is the total", "total number of", "list all","list unique","find all","retrieve"]
            guideline_keywords = ["guidelines", "management", "recommendations"]
            question_lower = question.lower()

            # Keyword-based detection
            if any(word in question_lower for word in plot_keywords):
                return "plot"
            elif any(word in question_lower for word in stat_keywords):
                return "stats"
            elif any(word in question_lower for word in query_keywords):
                return "tableqa"
            elif any(word in question_lower for word in guideline_keywords):
                return "guideline"
            
        # If no match, fall through to LLM classification
        memory_block = f"{context}\n\n" if context else ""

        # Fallback to LLM classification
        prompt = f"""
You are a routing classifier. Your job is to identify what type of task the user's question represents.

Classify the following question into ONE of these categories:

1. tableqa   — questions requiring dataframe queries, filtering, grouping, counting, comparisons, or extracting values from the dataset.
2. stats     — questions requiring statistical tests, correlations, summary statistics, confidence intervals, regressions, distributions.
3. plot      — questions requiring a figure or visualization based on the dataframe.
4. guideline — general questions asking about immune-related adverse events (irAEs), toxicity grading, clinical management, supportive care, or treatment guidance based on SITC/NCCN/ASCO guidelines.

RULES:
- Return ONLY the category name: one of {'tableqa', 'stats', 'plot', 'guideline'}.
- Do NOT explain your reasoning.

Conversation Context (oldest first, newest last):
{memory_block}

QUESTION:
{question}
        """
            
        classification = self.llm_client.generate(prompt).strip().lower()
        if classification in ["tableqa", "plot", "stats","guideline"]:
            return classification

        # Default to 'tableqa' if unsure
        return "tableqa"