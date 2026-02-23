import numpy as np
from src.utils import clean_text
import re

#### Helper functions for guideline linking in LLM output

# Map for RAG sources 
SOURCE_MAP = {
    "ASCO": "https://ascopubs.org/doi/10.1200/JCO.21.01440?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed",
    "NCCN": "https://jnccn.org/view/journals/jnccn/18/3/article-p230.xml",
    "SITC": "https://jitc.bmj.com/content/11/3/e006398"
}

# Simple safe-ish label escape
def esc_label(s):
    return s.replace("<", "&lt;").replace(">", "&gt;")

def make_link(url, label):
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{esc_label(label)}</a>'

# Matches (ASCO) or (ASCO;NCCN;PUBMED) where keys are letters/numbers
PAREN_RE = re.compile(
    r'(?:(?<=\()|(?<=\[)|(?<=【))'                # left bracket lookbehind: ( or [ or 【
    r'([A-Z0-9]+(?:\s*;\s*[A-Z0-9]+)*)'          # inner token: ASCO or ASCO;NCCN etc.
    r'(?=(?:\)|\]|】))'                           # right bracket lookahead: ) or ] or 】
)

def link_short_citations(text):
    """
    text: raw RAG output
    """
    def repl(m):
        items = [it.strip().upper() for it in m.group(1).split(";")]
        linked = []
        for key in items:
            tmpl = SOURCE_MAP.get(key)
            if tmpl:
                linked.append(make_link(tmpl, key))
            else:
                linked.append(key)
        return "; ".join(linked)
    
    return PAREN_RE.sub(repl, text)


class GuidelineAgent:

    def __init__(self, llm_client, search_fn, embed_fn = None, top_k=10):
        """
        search_fn: callable(query, top_k) -> list[metadata with 'text','doc_key','page' or section info, 'score']
        embed_fn: embedding function
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
    

    def handle(self, question, messages=None):
        """Handle guideline questions with RAG and convo history"""
        if messages is None:
            messages = []

        # Rag retrieval 
        retrieved = self.retrieve_relevant_chunks(question)
        context_text = "\n\n".join([f"Source: {p['source']} (Section {p['section_index']}, Chunk {p['chunk_index']})\nSection Title: {p['section_title']}\n{p['text']}\n" for p in retrieved])

        system_prompt = f"""
You are an expert on immune-related adverse events (irAEs) from cancer immunotherapy.
Use the following retrieved guideline exerpts to answer the user's question.

CRITICAL RULES:
- DO NOT recreate guideline tables.
- DO NOT use markdown tables or vertical bars (|).
- DO NOT output multi-column layouts.
- Write the answer as clean, concise clinical prose.
- CRITICAL: Cite sources in parentheses **with source(s) name only** (ASCO), (ASCO;NCCN).
- Every time information is used from the guidelines, cite the source.
- If a user asks specifically about ASCO, SITC, or NCCN guidelines, prioritize answers from that source.
- No HTML tags, no <br>, no <p>.
- If the question is not relevant to irAEs or guidelines, respond with: "Sorry, I can only answer questions related to immune-related adverse events (irAEs) and their management."

RETRIEVED GUIDELINES:
{context_text}
"""
        # Build messages for LLM
        full_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        full_messages.extend(messages)

        # Add current user question 
        full_messages.append({"role": "user", "content": question})

        # Generate and clean textual response
        result = self.llm.generate(messages=full_messages)
        result = clean_text(result)
        
        return {"type": "text", "code": None, "data": result}