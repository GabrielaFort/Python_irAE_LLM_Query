import numpy as np
from src.utils import clean_text

class GuidelineAgent:

    def __init__(self, llm_client, search_fn, embed_fn = None, top_k=10):
        """
        search_fn: callable(query, top_k) -> list[metadata with 'text','doc_key','page' or section info, 'score']
        embed_fn: optional embedding function (if needed elsewhere)
        """
        self.llm = llm_client
        self.search_fn = search_fn
        self.embed_fn = embed_fn
        self.top_k = top_k

    def retrieve_relevant_chunks(self, question, top_k=None):
        top_k = top_k or self.top_k
        results = self.search_fn(question, top_k=top_k)

        # Ensure expected keys exist — normalize to 'source' and 'page' or section_title
        normalized = []
        for r in results:
            # r contains: doc_key, section_index, section_title, chunk_index, text, 
            src = r.get("doc_key")
            section_index = r.get("section_index")
            chunk_index = r.get("chunk_index")
            section_title = r.get("section_title")
            normalized.append({"source": src, "section_index": section_index, "chunk_index": chunk_index, "section_title": section_title, "text": r.get("text",""), "score": r.get("score",0.0)})
        return normalized
    

    def handle(self, question, context=None):
        memory_block = f"{context}\n\n" if context else ""
        retrieved = self.retrieve_relevant_chunks(question)
        context_text = "\n\n".join([f"Source: {p['source']} (Section {p['section_index']}, Chunk {p['chunk_index']})\nSection Title: {p['section_title']}\n{p['text']}" for p in retrieved])
        print(context_text)

        prompt = f"""
You are an expert on immune-related adverse events (irAEs) from cancer immunotherapy.
Use the following retrieved guideline exerpts to answer the question below.


CRITICAL RULES:
- DO NOT recreate guideline tables.
- DO NOT use markdown tables or vertical bars (|).
- DO NOT output multi-column layouts.
- Write the answer as clean, concise clinical prose.
- Cite sources in parentheses with source name only (ASCO), (ASCO;NCCN).
- Every time information is used from the guidelines, cite the source.
- If a user asks specifically about ASCO, SITC, or NCCN guidelines, prioritize answers from that guideline.
- No HTML tags, no <br>, no <p>.
- If the question is not relevant to irAEs or guidelines, respond with: "Sorry, I can only answer questions related to immune-related adverse events (irAEs) and their management."

RETRIEVED GUIDELINES:
{context_text}

QUESTION:
{question}

CONVERSATION MEMORY (most recent message LAST): {memory_block}

FINAL ANSWER (clean prose, no table formatting):
"""
        result = self.llm.generate(prompt)
        result = clean_text(result)
        return {"type": "text", "code": None, "data": result}