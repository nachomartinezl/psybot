from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import torch

MODEL_NAME = "intfloat/multilingual-e5-large"
PERSIST_DIR = "chroma_db"
COLLECTION = "psybot_multilingual"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load model ----
print(f"Loading {MODEL_NAME} on {device}...")
model = SentenceTransformer(MODEL_NAME, device=device)

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(name=COLLECTION)

def search(query: str, k: int = 5):
    if not query.strip():
        print("Empty query.")
        return

    # Force list type to avoid tokenizer issues
    qtext = [f"query: {query.strip()}"]

    # Encode safely
    qvec = model.encode(
        qtext,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Ensure float32 dtype for Chroma
    qvec = np.array(qvec, dtype=np.float32)

    results = collection.query(
        query_embeddings=qvec,
        n_results=k,
        include=["documents", "metadatas"]
    )

    hits = results["documents"][0]
    metas = results["metadatas"][0]

    print(f"\nTop {len(hits)} results for: {query}")
    for i, (doc, meta) in enumerate(zip(hits, metas), 1):
        print(f"\n#{i}")
        print(doc[:400].replace("\n", " "))
        print("Meta:", meta)

if __name__ == "__main__":
    try:
        while True:
            query = input("\n🔍 Ask something: ").strip()
            if not query:
                break
            search(query)
    except KeyboardInterrupt:
        print("\nBye 👋")
