from sentence_transformers import SentenceTransformer
import chromadb
import torch

MODEL_NAME = "intfloat/multilingual-e5-large"
PERSIST_DIR = "chroma_db"
COLLECTION = "psybot_multilingual"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = SentenceTransformer(MODEL_NAME, device=device)
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(name=COLLECTION)

def search(query: str, k: int = 5):
    qvec = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    results = collection.query(
        query_embeddings=qvec,
        n_results=k,
        include=["documents", "metadatas"]
    )

    for i, doc in enumerate(results["documents"][0]):
        print(f"\n#{i+1}")
        print(doc)
        print(results["metadatas"][0][i])

if __name__ == "__main__":
    query = input("🔍 Ask something: ")
    search(query)
