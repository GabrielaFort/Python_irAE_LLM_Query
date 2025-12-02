import numpy as np
from src.utils import clean_text

class GuidelineAgent:

    def __init__(self, llm_client, kb_pages, kb_embeddings, top_k=5):
        self.llm = llm_client
        self.pages = kb_pages
        self.emb = kb_embeddings
        self.top_k = top_k

    def retrieve_relevant_pages(self, question, embed_fn):
        # embed_fn should return a 1d normalized vector for the question 
        q_vec = embed_fn(question)

        emb_norm = self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)
        sims = emb_norm @ q_vec

        # Top k similarity indices
        idx = sims.argsort()[-self.top_k:][::-1]
        return [self.pages[i] for i in idx]
    

    def handle(self, question, embed_fn, context=None):

        # Build memory block if context provided
        memory_block = f"{context}\n\n" if context else ""

        retrieved = self.retrieve_relevant_pages(question, embed_fn)

        context = "\n\n".join([f"Source: {p['source']} (Page {p['page']})\n{p['text']}" for p in retrieved])

        prompt = f"""
You are an expert on immune-related adverse events (irAEs) from cancer immunotherapy.
Use the following retrieved guideline exerpts AND general oncology knowledge where relevant to answer the question below.

CRITICAL RULES:
- DO NOT recreate guideline tables.
- DO NOT use markdown tables or vertical bars (|).
- DO NOT output multi-column layouts.
- Write the answer as clean, concise clinical prose.
- No HTML tags, no <br>, no <p>.
- If the question is not relevant to irAEs or guidelines, respond with: "Sorry, I can only answer questions related to immune-related adverse events (irAEs) and their management."

RETRIEVED GUIDELINES:
{context}

QUESTION:
{question}

CONVERSATION MEMORY (most recent message LAST): {memory_block}

FINAL ANSWER (clean prose, no table formatting):
"""
        # LLM client returns text response
        result = self.llm.generate(prompt)

        result = clean_text(result)

        return {
            "type": "text",
            "code": None,
            "data": result
        }
    