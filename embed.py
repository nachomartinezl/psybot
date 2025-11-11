import os
import json
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

# ---------- PATHS ----------
CHUNKS_DIR = Path("data/chunks")
STORE_DIR = Path("vector_store")
STORE_DIR.mkdir(exist_ok=True)

index_path = STORE_DIR / "faiss.index"
meta_path = STORE_DIR / "meta.jsonl"

# ---------- EMBEDDING MODEL (GPU) ----------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME, device="cuda")
EMB_DIM = model.get_sentence_embedding_dimension()
BATCH_SIZE = 32

# ---------- INITIALIZE FAISS ----------
if index_path.exists():
    index = faiss.read_index(str(index_path))
else:
    # Inner product (cosine similarity if vectors normalized)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(EMB_DIM))

# Metadata file append-mode
meta_file = open(meta_path, "a", encoding="utf-8")

# Continue IDs
next_id = index.ntotal


# ---------- HELPERS ----------
def embed_batch(texts):
    """Embed a list of texts on GPU."""
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda"
    )
    return vecs.astype("float32")


# ---------- MAIN PIPELINE ----------
def process_chunks():
    global next_id

    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))

    for chunk_file in tqdm(chunk_files, desc="Embedding chunks"):
        
        with chunk_file.open("r", encoding="utf-8") as f:
            batch_texts = []
            batch_meta = []

            for line in f:
                obj = json.loads(line)
                text = obj["text"].strip()
                if not text:
                    continue

                batch_texts.append(text)
                batch_meta.append(obj)

                # When batch full → embed and store
                if len(batch_texts) >= BATCH_SIZE:
                    vecs = embed_batch(batch_texts)
                    ids = np.arange(next_id, next_id + len(vecs))
                    next_id += len(vecs)

                    index.add_with_ids(vecs, ids)

                    for m, vid in zip(batch_meta, ids):
                        m["vector_id"] = int(vid)
                        meta_file.write(json.dumps(m) + "\n")

                    batch_texts.clear()
                    batch_meta.clear()

            # Tail batch
            if batch_texts:
                vecs = embed_batch(batch_texts)
                ids = np.arange(next_id, next_id + len(vecs))
                next_id += len(vecs)

                index.add_with_ids(vecs, ids)

                for m, vid in zip(batch_meta, ids):
                    m["vector_id"] = int(vid)
                    meta_file.write(json.dumps(m) + "\n")

        # Save index after each chunk file
        faiss.write_index(index, str(index_path))

    meta_file.close()
    faiss.write_index(index, str(index_path))

    print("\n✅ DONE.")
    print(f"→ Total vectors: {index.ntotal}")
    print(f"→ FAISS index saved to: {index_path}")
    print(f"→ Metadata saved to: {meta_path}")


if __name__ == "__main__":
    process_chunks()
