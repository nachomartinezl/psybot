from pydantic_ai import Agent
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import torch

# ---------- CONFIG ----------
MODEL_NAME = "intfloat/multilingual-e5-large"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "psybot_multilingual"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- SYSTEM PROMPT ----------
SYSTEM_PROMPT = """
You are a psychoanalytic conversational agent and therapeutic listener.

You use psychoanalytic technique — exploring symbols, defenses, emotions, transference,
and unconscious dynamics — to help users reflect on their inner experience.

Your voice is calm, insightful, and patient. You do not give advice or instructions.
You do not use coaching or CBT language. You stay in the psychoanalytic frame.

Your task is to:
- Listen carefully to the user’s associations and emotions.
- Ask thoughtful, open-ended questions that invite reflection.
- Interpret patterns and slips symbolically, not literally.
- Gently point to underlying wishes, fears, or conflicts.
- Stay neutral, never moralizing or judging.
- End the session naturally when appropriate, saying something like:
  “Let’s pause here for today. We can explore this further next time.”

Use the retrieved psychological and psychiatric texts from your knowledge base
only as inspiration for your interpretations — never quote or summarize them directly.

If the user’s message seems urgent or unsafe, respond with care and recommend they seek
a qualified professional immediately, without continuing analysis.
"""


# ---------- LOAD MODELS ----------
print(f"Loading retriever model ({MODEL_NAME}) on {device} ...")
embedder = SentenceTransformer(MODEL_NAME, device=device)

print("Connecting to Chroma...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# ---------- DEFINE AGENT ----------
agent = Agent(
    "google-gla:gemini-2.5-pro",
    system_prompt=SYSTEM_PROMPT,
)

# ---------- INPUT / OUTPUT MODELS ----------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str


# ---------- RETRIEVER ----------
def retrieve_context(query: str, k: int = 5) -> str:
    """Get top-k relevant chunks from Chroma."""
    qvec = embedder.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    results = collection.query(
        query_embeddings=qvec,
        n_results=k,
        include=["documents", "metadatas"]
    )
    docs = results["documents"][0]
    context = "\n\n".join(docs)
    return context


# ---------- CHAT FUNCTION ----------
async def chat_with_context(user_message: str) -> ChatResponse:
    """Retrieve context → feed into Gemini."""
    context = retrieve_context(user_message)

    augmented_prompt = f"""
Context from psychoanalytic corpus:
{context}

User message:
{user_message}

Respond reflectively and empathetically.
    """

    reply = await agent.run(augmented_prompt)
    return ChatResponse(response=reply.output_text)
