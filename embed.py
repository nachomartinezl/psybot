import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ---------- PATHS ----------
CHUNKS_DIR = Path("data/chunks")
STORE_DIR = Path("vector_store")
STORE_DIR.mkdir(exist_ok=True)


# ---------- EMBEDDING MODEL (GPU) ----------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
)

print("Loaded:", MODEL_NAME)
print("Embedding dim:", model.get_sentence_embedding_dimension())

EMB_DIM = model.get_sentence_embedding_dimension()
BATCH_SIZE = 32


# ---------- CHROMA SETUP ----------
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=str(STORE_DIR)
))

collection = chroma_client.get_or_create_collection(
    name="psychoanalytic_books",
    metadata={"hnsw:space": "cosine"}
)


# ---------- HELPERS ----------
def embed_batch(texts):
    """Embed a list of texts on GPU."""
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda"
    ).astype("float32")


# ---------- MAIN PIPELINE ----------
def process_chunks():

    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    print(f"Found {len(chunk_files)} chunk files.")

    for chunk_file in tqdm(chunk_files, desc="Embedding chunks"):

        batch_texts = []
        batch_meta = []
        batch_ids = []

        with chunk_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                text = obj["text"].strip()
                if not text:
                    continue

                batch_texts.append(text)
                batch_meta.append(obj)
                batch_ids.append(f"{chunk_file.stem}_{i}")

                if len(batch_texts) >= BATCH_SIZE:
                    vecs = embed_batch(batch_texts)

                    collection.add(
                        embeddings=vecs,
                        documents=batch_texts,
                        metadatas=batch_meta,
                        ids=batch_ids
                    )

                    batch_texts, batch_meta, batch_ids = [], [], []

        # tail batch
        if batch_texts:
            vecs = embed_batch(batch_texts)
            collection.add(
                embeddings=vecs,
                documents=batch_texts,
                metadatas=batch_meta,
                ids=batch_ids
            )

        chroma_client.persist()

    print("\n✅ DONE")
    print(f"→ Stored Chroma collection at: {STORE_DIR}")


if __name__ == "__main__":
    process_chunks()
