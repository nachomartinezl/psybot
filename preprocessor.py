import os
import re
import unicodedata
from lingua import Language, LanguageDetectorBuilder

INPUT_DIR = "data/clean"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# LANGUAGE DETECTOR (LINGUA)
# -------------------------------------------------------

LANGUAGES = [
    Language.ENGLISH,
    Language.GERMAN,
    Language.FRENCH,
    Language.DUTCH,
    Language.ITALIAN,
    Language.SPANISH,
    Language.SWEDISH,
    Language.DANISH,
    Language.PORTUGUESE,
    Language.FINNISH,
    Language.POLISH,
    Language.CZECH,
    Language.LATIN,
]

detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()

def detect_language(text: str) -> str:
    sample = text[:4000]  # small sample is enough
    lang = detector.detect_language_of(sample)
    return lang.iso_code_639_1.name.lower() if lang else "unknown"


# -------------------------------------------------------
# REMOVE TOC / INDEX / BIBLIOGRAPHY / ETC.
# -------------------------------------------------------

NOISE_HEADERS = [
    r"^CONTENTS$",
    r"^TABLE OF CONTENTS$",
    r"^TOC$",
    r"^INDEX$",
    r"^INDEX OF AUTHORS$",
    r"^INDEX OF SUBJECTS$",
    r"^GENERAL INDEX$",
    r"^BIBLIOGRAPHY$",
    r"^REFERENCES$",
    r"^ILLUSTRATIONS$",
    r"^LIST OF ILLUSTRATIONS$",
    r"^APPENDIX$",
    r"^APPENDICES$",
    r"^GLOSSARY$",
    r"^FOOTNOTES$",
    r"^NOTES$",
    r"^TRANSCRIBERS?",
]

combined_noise_pattern = re.compile("|".join(NOISE_HEADERS), flags=re.IGNORECASE)

def remove_trailing_sections(text: str) -> str:
    lines = text.split("\n")
    cutoff = len(lines)

    for i, line in enumerate(lines):
        clean = line.strip().upper()

        # Flexible match for headers
        if combined_noise_pattern.match(clean):
            # Only cut if this happens after 30% of the text
            if i > len(lines) * 0.30:
                cutoff = i
                break

        # TOC-style rows "CHAPTER I ...... 12"
        if re.match(r".+\.{2,}\s*\d+$", clean) and i < len(lines) * 0.25:
            continue

    return "\n".join(lines[:cutoff]).strip()


# -------------------------------------------------------
# DEDUPLICATE PARAGRAPHS
# -------------------------------------------------------

def dedupe_paragraphs(text: str) -> str:
    seen = set()
    final = []

    for para in text.split("\n\n"):
        cleaned = para.strip()
        if not cleaned:
            continue

        # For dedupe, normalize internal whitespace + lowercase
        key = re.sub(r"\s+", " ", cleaned.lower())

        if key not in seen:
            seen.add(key)
            final.append(cleaned)

    return "\n\n".join(final)


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------

def postprocess_books():
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".txt"):
            continue

        book_id = fname.replace(".txt", "")
        in_path = os.path.join(INPUT_DIR, fname)

        print(f"\n=== Postprocessing {book_id} ===")

        try:
            with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            # Apply additional cleaning layers
            text = remove_trailing_sections(text)
            text = dedupe_paragraphs(text)

            # Detect language
            lang = detect_language(text)

            # Output path
            out_path = os.path.join(OUTPUT_DIR, f"{book_id}_{lang}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Saved processed book → {out_path}")

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")


if __name__ == "__main__":
    postprocess_books()
    print("\n🔥 Phase 3+ cleaning complete for all books.")
