import os
import re
import csv
import requests
import unicodedata

CSV_PATH = "gutenberg_books.csv"
OUT_DIR = "data/clean"

os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}


# -------------------------------------------------------
# DOWNLOAD TEXT FROM GUTENBERG
# -------------------------------------------------------

def download_book(url: str) -> str:
    """Returns the full text of the book from the .txt.utf-8 URL."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


# -------------------------------------------------------
# PHASE 2: Extract only the inner content
# -------------------------------------------------------

def extract_gutenberg_content(text: str) -> str:
    start_marker = "START OF THE PROJECT GUTENBERG"
    end_marker = "END OF THE PROJECT GUTENBERG"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    # If markers missing, fallback to entire text
    if start_idx == -1 or end_idx == -1:
        return text.strip()

    # Move pointer to next line after start marker
    start_idx = text.find("\n", start_idx)
    if start_idx == -1:
        return text.strip()
    start_idx += 1

    return text[start_idx:end_idx].strip()


# -------------------------------------------------------
# PHASE 3: Neutral Cleaning
# -------------------------------------------------------

def clean_text(text: str) -> str:
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove Windows-style carriage returns
    text = text.replace("\r", "")

    # Collapse 3+ newlines into exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove ASCII decoration lines like ***** or ------
    text = re.sub(r"^[\*\-=]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Strip trailing spaces per-line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Final global trim
    return text.strip()


# -------------------------------------------------------
# MAIN PROCESSOR
# -------------------------------------------------------

def process_all_books():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            book_id = row["book_id"]
            url = row["plain_text_url"]

            print(f"\n=== Processing Book {book_id} ===")
            print(f"Downloading: {url}")

            try:
                # DOWNLOAD
                raw_text = download_book(url)

                # EXTRACT INNER TEXT
                extracted = extract_gutenberg_content(raw_text)

                # CLEAN
                cleaned = clean_text(extracted)

                # SAVE
                out_path = os.path.join(OUT_DIR, f"{book_id}.txt")
                with open(out_path, "w", encoding="utf-8") as out_file:
                    out_file.write(cleaned)

                print(f"Saved cleaned book → {out_path}")

            except Exception as e:
                print(f"Error processing {book_id}: {e}")


if __name__ == "__main__":
    process_all_books()
    print("\nAll books downloaded, extracted, and cleaned.")
