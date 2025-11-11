import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
import time

BASE = "https://www.gutenberg.org"
START_URL = f"{BASE}/ebooks/bookshelf/688"
LAST_BOOK_ID = 586

UA = {"User-Agent": "Mozilla/5.0"}

def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def parse_bookshelf_page(soup: BeautifulSoup):
    rows = []
    reached_last = False

    for li in soup.select("li.booklink"):
        a = li.select_one("a.link")
        if not a or not a.get("href", "").startswith("/ebooks/"):
            continue

        href = a["href"]                         # /ebooks/35924
        book_id = int(href.split("/")[2])

        # Title
        title_tag = li.select_one("span.title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Author (exact subtitle tag)
        subtitle_tag = li.select_one("span.subtitle")
        author = subtitle_tag.get_text(strip=True) if subtitle_tag else ""

        # Correct plain text link
        text_url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

        rows.append((book_id, title, author, text_url))

        if book_id == LAST_BOOK_ID:
            reached_last = True

    return rows, reached_last

def find_next_page(soup: BeautifulSoup):
    nxt = soup.find("a", string="Next")
    return urljoin(BASE, nxt["href"]) if nxt and nxt.get("href") else None

def scrape_all():
    url = START_URL
    out = []
    seen = set()

    while url:
        print(f"Scraping: {url}")
        soup = get_soup(url)

        rows, reached_last = parse_bookshelf_page(soup)
        for book_id, title, author, text_url in rows:
            if book_id not in seen:
                out.append((book_id, title, author, text_url))
                seen.add(book_id)

        if reached_last:
            print(f"Reached book {LAST_BOOK_ID}. Done.")
            break

        url = find_next_page(soup)
        time.sleep(0.5) # Be polite

    out.sort(key=lambda r: r[0])
    return out

def save_to_csv(rows, filename="gutenberg_books.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["book_id", "title", "author", "plain_text_url"])
        w.writerows(rows)
    print(f"\nSaved CSV → {filename}")

if __name__ == "__main__":
    data = scrape_all()
    save_to_csv(data)
