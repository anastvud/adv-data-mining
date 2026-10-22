import time

import requests

headers = {"User-Agent": "AdvancedDataMining/1.0 (your-email@example.com)"}
search_response = requests.get(
    "https://openlibrary.org/search.json",
    params={
        "q": "Pride and Prejudice",
        "fields": "key,title,author_name",
        "limit": 1,
    },
    headers=headers,
    timeout=10,
)
search_response.raise_for_status()
search_results = search_response.json()

books = search_results.get("docs", [])
if not books:
    raise LookupError("No matching books found in Open Library.")

book = books[0]
work_response = requests.get(
    f"https://openlibrary.org{book['key']}.json",
    headers=headers,
    timeout=10,
)
work_response.raise_for_status()
work = work_response.json()

description = work.get("description")
if isinstance(description, dict):
    description = description.get("value")

genres = []
for genre in work.get("genres", []):
    if genre.startswith("/tags/"):
        time.sleep(1)
        genre_response = requests.get(
            f"https://openlibrary.org{genre}.json",
            headers=headers,
            timeout=10,
        )
        genre_response.raise_for_status()
        genres.append(genre_response.json()["name"])
    else:
        genres.append(genre)

print(f"Title: {book.get('title', 'Unknown title')}")
print(f"Authors: {', '.join(book.get('author_name', [])) or 'Unknown'}")
print(f"Description: {description or 'No description available.'}")
print(f"Genres: {', '.join(genres) or 'No genres assigned.'}")
