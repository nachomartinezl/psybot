import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ---- PATHS ----
CHUNKS_DIR = Path("data/chunks")
DB_DIR = "vector_store"

# ---- CHROMA ----
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(
    name="psybot",
    metadata={"hnsw:space": "cosine"}
)

# ---- MODEL ----
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME, device="cuda")
BATCH_SIZE = 32

print("Loaded:", MODEL_NAME)

def embed(texts):
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda"
    ).astype(np.float32)

# ---- MAIN ----
def process():
    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))

    doc_id_counter = 0

    for file in tqdm(chunk_files, desc="Embedding chunks"):

        with file.open() as f:
            buffer_texts = []
            buffer_meta = []

            for line in f:
                obj = json.loads(line)
                text = obj["text"].strip()
                if not text:
                    continue

                buffer_texts.append(text)
                buffer_meta.append(obj)

                if len(buffer_texts) >= BATCH_SIZE:
                    vecs = embed(buffer_texts)
                    ids = [str(doc_id_counter + i) for i in range(len(vecs))]
                    doc_id_counter += len(vecs)

                    collection.add(
                        ids=ids,
                        embeddings=vecs.tolist(),
                        metadatas=buffer_meta,
                        documents=buffer_texts
                    )

                    buffer_texts, buffer_meta = [], []

            # tail batch
            if buffer_texts:
                vecs = embed(buffer_texts)
                ids = [str(doc_id_counter + i) for i in range(len(vecs))]
                doc_id_counter += len(vecs)

                collection.add(
                    ids=ids,
                    embeddings=vecs.tolist(),
                    metadatas=buffer_meta,
                    documents=buffer_texts
                )

    print("✅ DONE — embeddings stored in Chroma DB:", DB_DIR)

if __name__ == "__main__":
    process()
