import os
import json
import re
from pathlib import Path
import nltk

# Make sure NLTK has the sentence tokenizer
nltk.download('punkt')

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# TOKEN ESTIMATOR (rough)
# -----------------------------
# We approximate tokens using a standard rule: ~4 chars/token for English.
# Good enough for sizing chunks before real embedding.

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# -----------------------------
# SENTENCE SPLITTING
# -----------------------------

def split_sentences(text: str):
    """Split text into sentences using NLTK."""
    return nltk.sent_tokenize(text)


# -----------------------------
# BUILD CHUNKS
# -----------------------------

def build_chunks(sentences, max_tokens=900, overlap=120):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)

        # Too long sentence → push alone
        if sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.append(sent)
            continue

        # If adding the sentence would exceed max_tokens → finalize chunk
        if current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))

            # Create overlap window
            overlap_sents = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                t = estimate_tokens(s)
                if overlap_tokens + t > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += t

            current_chunk = overlap_sents.copy()
            current_tokens = sum(estimate_tokens(s) for s in current_chunk)

        # Add sentence normally
        current_chunk.append(sent)
        current_tokens += sent_tokens

    # Final remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# -----------------------------
# MAIN PROCESSOR
# -----------------------------

def chunk_all_books():
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".txt"):
            continue

        in_path = os.path.join(INPUT_DIR, fname)

        # Extract metadata from filename
        # Example: "35924_en.txt"
        base = Path(fname).stem
        parts = base.split("_")

        if len(parts) == 2:
            book_id, lang = parts
        else:
            # fallback
            book_id = parts[0]
            lang = "unknown"

        print(f"\n📚 Chunking Book {book_id} ({lang})")

        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        # Sentence splitting
        sentences = split_sentences(text)

        # Chunk building
        chunks = build_chunks(sentences)

        # Save chunks to JSONL
        out_path = os.path.join(OUTPUT_DIR, f"{book_id}.jsonl")
        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, chunk in enumerate(chunks):
                obj = {
                    "book_id": book_id,
                    "lang": lang,
                    "chunk_index": i,
                    "text": chunk
                }
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"✅ Saved {len(chunks)} chunks → {out_path}")


if __name__ == "__main__":
    chunk_all_books()
    print("\n🔥 All books chunked successfully.")
