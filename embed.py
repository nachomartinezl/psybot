import os
import json
from pathlib import Path
from typing import List, Dict

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

# -------------------- CONFIG --------------------
CHUNKS_DIR = Path("data/chunks")          # .jsonl files with {"text", "book_id", ...}
PERSIST_DIR = "chroma_db"                 # on-disk vector store
COLLECTION  = "psybot_multilingual"
MODEL_NAME  = "intfloat/multilingual-e5-large"  # multilingual, retrieval-optimized

BATCH_SIZE = 128                          # RTX A5000 can handle this easily
USE_FP16   = True                         # half precision on GPU
DEVICE     = "cuda"                       # force GPU

# -------------------- SAFETY CHECKS --------------------
assert torch.cuda.is_available(), "CUDA not available. Make sure nvidia-smi shows your GPU."
print("GPU:", torch.cuda.get_device_name(0))

# -------------------- MODEL --------------------
# SentenceTransformer wraps pooling & tokenizer for this HF model.
# E5 expects "passage: ..." for documents (and "query: ..." for queries).
model_kwargs = {}
if USE_FP16:
    # will cast internal forward to fp16 where supported
    model_kwargs["device"] = DEVICE
else:
    model_kwargs["device"] = DEVICE

model = SentenceTransformer(MODEL_NAME, **model_kwargs)
model.max_seq_length = 512  # E5 context length; keep consistent
print(f"Loaded model: {MODEL_NAME} | dim={model.get_sentence_embedding_dimension()}")

# -------------------- CHROMA --------------------
client = chromadb.PersistentClient(
    path=PERSIST_DIR,
    settings=Settings(allow_reset=False)  # keep data
)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}  # cosine works with normalized embeddings
)

# -------------------- HELPERS --------------------
def iter_chunk_files():
    files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL chunk files in {CHUNKS_DIR.resolve()}")
    return files

def load_jsonl(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def embed_passages(texts: List[str]):
    # Prefix for E5 document embeddings
    prefixed = [f"passage: {t}" for t in texts]
    # normalize_embeddings=True gives unit vectors -> cosine ready
    embs = model.encode(
        prefixed,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    if USE_FP16:
        # store as float32 for compatibility; vectors are normalized already
        embs = embs.astype("float32")
    return embs

def already_inserted_ids(ids: List[str]) -> set:
    """Check which ids exist to make the process idempotent."""
    existing = set()
    # Chroma doesn’t have a direct bulk-exists API; we probe in chunks.
    for i in range(0, len(ids), 1000):
        batch = ids[i:i+1000]
        res = collection.get(ids=batch, include=[])
        if res and res.get("ids"):
            existing.update(res["ids"])
    return existing

# -------------------- MAIN --------------------
def main():
    files = iter_chunk_files()
    total_added = 0

    for fp in tqdm(files, desc="Embedding & indexing"):
        docs: List[str] = []
        metas: List[Dict] = []
        ids:   List[str] = []

        for obj in load_jsonl(fp):
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            # Build deterministic ID: <bookid>:<chunk_index or hash>
            # Prefer provided "id"/"chunk_id" if present; else hash text.
            cid = obj.get("id") or obj.get("chunk_id")
            if not cid:
                cid = str(abs(hash(text)) % (10**12))
            book_id = str(obj.get("book_id") or fp.stem)
            vec_id  = f"{book_id}:{cid}"

            ids.append(vec_id)
            docs.append(text)
            metas.append({k: v for k, v in obj.items() if k != "text"})

        if not ids:
            continue

        # Skip already present vector ids (idempotent)
        exist = already_inserted_ids(ids)
        if exist:
            keep = [(i, d, m) for i, d, m in zip(ids, docs, metas) if i not in exist]
            if not keep:
                continue
            ids, docs, metas = map(list, zip(*keep))

        # Embed & upsert in batches
        for i in range(0, len(docs), BATCH_SIZE):
            batch_docs = docs[i:i+BATCH_SIZE]
            batch_ids  = ids[i:i+BATCH_SIZE]
            batch_meta = metas[i:i+BATCH_SIZE]

            embs = embed_passages(batch_docs)  # np.float32 (normalized)
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=embs,
                metadatas=batch_meta,
            )
            total_added += len(batch_ids)

        # Persist after each file to be safe
        # (PersistentClient commits automatically, this is just a log)
        print(f"Indexed {len(ids)} from {fp.name} (total so far: {total_added})")

    print("\n✅ Done.")
    print(f"Chroma path: {Path(PERSIST_DIR).resolve()}")
    print(f"Collection:  {COLLECTION}")
    print(f"Total new vectors: {total_added}")

if __name__ == "__main__":
    main()
