"""Build a frequency bag-of-words representation of the cleaned book descriptions.

The saved matrix and metadata have the same row order. The vectorizer is fitted
on the full dataset, so refit it on training descriptions before evaluating a
predictive model on held-out books.
"""

import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "BALANCED_CLEANED.parquet"
OUTPUT_DIR = ROOT / "data" / "fbow"
MATRIX_PATH = OUTPUT_DIR / "fbow_matrix.npz"
VECTORIZER_PATH = OUTPUT_DIR / "fbow_vectorizer.pkl"
METADATA_PATH = OUTPUT_DIR / "fbow_metadata.parquet"


def main() -> None:
    books = pd.read_parquet(INPUT_PATH, columns=["work_key", "desc_cleaned", "genres"])
    if books[["work_key", "desc_cleaned", "genres"]].isna().any().any():
        raise ValueError("Book identifiers, descriptions, and genres must not be missing.")
    if books["desc_cleaned"].str.strip().eq("").any():
        raise ValueError("Book descriptions must not be empty.")

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(books["desc_cleaned"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_npz(MATRIX_PATH, matrix)
    with VECTORIZER_PATH.open("wb") as file:
        pickle.dump(vectorizer, file)
    books[["work_key", "genres"]].to_parquet(METADATA_PATH, index=False)
    print(f"Saved {matrix.shape[0]} books and {matrix.shape[1]} frequency features to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()