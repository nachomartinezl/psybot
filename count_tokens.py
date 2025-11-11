import os
import tiktoken

INPUT_DIR = "data/processed"
ENCODING = "cl100k_base"   # Same tokenizer used by OpenAI, Jina, Voyage

# Load tokenizer
enc = tiktoken.get_encoding(ENCODING)

def count_tokens(text: str) -> int:
    """Return exact tokenizer-based token count."""
    return len(enc.encode(text))

def main():
    total_tokens = 0

    print("\n============================")
    print("📚 TOKEN COUNT PER BOOK")
    print("============================")

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(INPUT_DIR, fname)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        tokens = count_tokens(text)
        total_tokens += tokens

        print(f"{fname:<30} {tokens:>12,} tokens")

    print("\n============================")
    print("📦 TOTAL TOKEN COUNT")
    print("============================")
    print(f"Total tokens: {total_tokens:,}")

    print("\n============================")
    print("💸 COST ESTIMATES")
    print("============================")
    print(f"OpenAI text-embedding-3-small (~$0.02 per 1M):   ${total_tokens / 1_000_000 * 0.02:.4f}")
    print(f"Jina embedding-small (~$0.01 per 1M):           ${total_tokens / 1_000_000 * 0.01:.4f}")
    print(f"Jina embedding-large (~$0.10 per 1M):           ${total_tokens / 1_000_000 * 0.10:.4f}")
    print(f"Voyage large (~$1.00 per 1M):                   ${total_tokens / 1_000_000 * 1.00:.4f}")

if __name__ == "__main__":
    main()
