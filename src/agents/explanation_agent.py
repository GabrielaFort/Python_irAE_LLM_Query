
class ExplanationAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_explanation(self, messages):
        """
        Generate a short explanation of what the code does.
        Will only be run if the previous question resulted in code generation.
        Args:
            messages - list of previous conversation messages
        """
        system_prompt = f"""
Given a user's question and the Python code that was generated to answer the question,
provide a brief, clear explantion of what the analysis does.

Rules:
- Keep it 1-2 sentences maximum.
- Use plain language, avoid technical jargon.
- Focus on WHAT was done, not HOW.
- Explain what YOU did to answer the question.
- CRITICAL: ONLY INCLUDE THE EXPLANATION, not the question or code or other thoughts.

Examples:
Question: "Show me lung cancer patients with a rash"
Code: df[(df['tumor_type'] == 'lung cancer') & (df['irae'].str.contains('rash', case=False))]
Explanation: "I filtered the dataset to show only lung cancer patients who experienced a rash as an irAE."

Question: "Test for an association between irae type and sex"
Code: df2 = df[['irae_type','sex']].dropna()\ndf2['irae_type'] = df2['irae_type'].astype(str).str.split(r'\s*,\s*')\ndf2 = df2.explode('irae_type')\ntab = pd.crosstab(df2['irae_type'], df2['sex'])\nchi2, p_value, dof, _ = stats.chi2_contingency(tab)\nresult = pd.DataFrame({{'statistic':[chi2], 'p_value':[p_value]}})
Explanation: "I checked whether the distribution of different irAE types varies between male and female patients by building a table of counts and running a chi‑square test of independence.
"""
        # Build messages for LLM
        full_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        full_messages.extend(messages)

        # Generate explanation
        explanation = self.llm_client.generate(messages=full_messages)

        # Clean up the explanation (remove quotes, extra whitespace)
        explanation = explanation.strip().strip('"').strip("'").strip()
        
        return explanation
