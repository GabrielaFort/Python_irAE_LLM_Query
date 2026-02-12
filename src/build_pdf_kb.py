import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pymupdf
import pymupdf.layout
import pymupdf4llm
import re

PDFS = {
    "SITC": "./knowledge_base/SITC_guidelines.pdf",
    "NCCN": "./knowledge_base/NCCN_guidelines.pdf",
    "ASCO": "./knowledge_base/ASCO_guidelines.pdf"
}

MODEL_NAME = "NeuML/pubmedbert-base-embeddings" # sbert style pubmedbert fine tuned for sentence embeddings
OUTPUT_DIR = "./knowledge_base"   
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 250

# Utilize pymupdf4llm to extract markdown from each pdf
# This is documented here: https://github.com/pymupdf/pymupdf4llm
def extract_markdown(pdf_path, header=False, footer=False):
        document = pymupdf.open(pdf_path)
        md = pymupdf4llm.to_markdown(document, header=header, footer=footer, use_ocr = True)
        return md

# Patterns to identify reference sections
REF_HEADERS = [
    r"(?mi)^#{1,6}\s*(?:\*\*)?References(?:\*\*)?\s*$",
    r"(?mi)^#{1,6}\s*(?:\*\*)?Bibliography(?:\*\*)?\s*$",
    r"(?mi)^#{1,6}\s*(?:\*\*)?Works Cited(?:\*\*)?\s*$",
    r"(?mi)^#{1,6}\s*(?:\*\*)?Literature Cited(?:\*\*)?\s*$",
    r"(?mi)^#{1,6}\s*(?:\*\*)?Reference[s]?(?:\*\*)?\s*$",
    r"(?mi)^#{1,6}\s*(?:\*\*)?Acknowledg(?:ement|ements)(?:\*\*)?\s*$"
]

def strip_reference_sections(md_text: str) -> str:
    # find the earliest reference-like header and cut it plus everything after it
    for pat in REF_HEADERS:
        m = re.search(pat, md_text)
        if m:
            print("Stripping reference section starting at header:", m.group(0)[:80])
            return md_text[:m.start()].strip()
    return md_text

# Split by markdown headers, fallback to single big section
def split_into_sections(md_text, doc_id):
    section_pattern = re.compile(r"(?m)^(#{1,6})\s+(.+)$")
    matches = list(section_pattern.finditer(md_text))
    sections = []
    if matches:
        for idx, m in enumerate(matches):
            title = m.group(2).strip()
            start = m.end()
            end = matches[idx+1].start() if idx+1 < len(matches) else len(md_text)
            content = md_text[start:end].strip()
            print("\n\nSection", idx+1)
            print(title)
            print(content[:100])
            sections.append({"title": title, "content": content, "section_index": idx+1, "doc_id": doc_id})
    else:
        sections.append({"title": doc_id, "content": md_text.strip(), "section_index": 1, "doc_id": doc_id})
    return sections

# Overlapping chunker (character-based)
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks


def build_kb():
    model = SentenceTransformer(MODEL_NAME)
    embed_dim = model.get_sentence_embedding_dimension()

    all_metadatas = []
    texts = []

    for source, path in PDFS.items():
        print("Processing", source, path)
        md = extract_markdown(path) 
        md = strip_reference_sections(md)
        doc_basename = os.path.basename(path)
        sections = split_into_sections(md, doc_basename)
        for sec in sections:
            chunks = chunk_text(sec["content"])
            for ci, c in enumerate(chunks):
                meta = {
                    "doc_key": source,
                    "doc_id": sec["doc_id"],
                    "section_index": sec["section_index"],
                    "section_title": sec["title"],
                    "chunk_index": ci,
                    "text": c
                }
                all_metadatas.append(meta)
                
                embed_text = (
                    f"Document: {source}\n"
                    f"Doc ID: {sec['doc_id']}\n"
                    f"Section: {sec['section_index']} — {sec['title']}\n\n"
                    f"{c}"
                )
                texts.append(embed_text)  

    print(f"Total chunks: {len(texts)}")

    print("Embedding chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    # build FAISS inner product index
    index = faiss.IndexFlatIP(embed_dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss.index"))

    # save metadatas line-delimited
    with open(os.path.join(OUTPUT_DIR, "metadatas.jsonl"), "w", encoding="utf-8") as f:
        for m in all_metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # config manifest
    with open(os.path.join(OUTPUT_DIR, "index_config.json"), "w") as f:
        json.dump({"model_name": MODEL_NAME, "embed_dim": embed_dim, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}, f)

    print("Saved faiss.index, metadatas.jsonl, index_config.json to", OUTPUT_DIR)

if __name__ == "__main__":
    build_kb()
