import json
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

PDFS = {
    "SITC": "../data/SITC_guidelines.pdf",
    "NCCN": "../data/NCCN_guidelines.pdf",
    "ASCO": "../data/ASCO_guidelines.pdf"
}

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
OUTPUT_DIR = "./knowledge_base"

# Load pages of PDF
def load_pdf_pages(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({
            "page": i + 1,
            "text": text
        })
    return pages


# Generate embeddings for text chunks
def build_kb():
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []

    # Load all PDFs
    for source, path in PDFS.items():
        pages = load_pdf_pages(path)
        for p in pages:
            p["source"] = source
            all_chunks.append(p)

    print(f"Loaded {len(all_chunks)} chunks from {len(PDFS)} PDFs.")

    # Convert list → array of texts
    texts = [c["text"] for c in all_chunks]

    print("Computing embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Save pages
    with open(f"{OUTPUT_DIR}/pages.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    # Save embeddings
    np.save(f"{OUTPUT_DIR}/embeddings.npy", embeddings)

    print("Knowledge base built successfully:")
    print(f" - saved pages.json")
    print(f" - saved embeddings.npy")


if __name__ == "__main__":
    build_kb()