"""Build a TF-IDF unigram-and-bigram representation of cleaned descriptions.

The saved matrix and metadata have the same row order. The vectorizer is fitted
on the full dataset, so refit it on training descriptions before evaluating a
predictive model on held-out books.
"""

import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "BALANCED_CLEANED.parquet"
OUTPUT_DIR = ROOT / "data" / "tfidf"
MATRIX_PATH = OUTPUT_DIR / "tfidf_matrix.npz"
VECTORIZER_PATH = OUTPUT_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = OUTPUT_DIR / "tfidf_metadata.parquet"


def main() -> None:
    books = pd.read_parquet(INPUT_PATH, columns=["work_key", "desc_cleaned", "genres"])
    if books[["work_key", "desc_cleaned", "genres"]].isna().any().any():
        raise ValueError("Book identifiers, descriptions, and genres must not be missing.")
    if books["desc_cleaned"].str.strip().eq("").any():
        raise ValueError("Book descriptions must not be empty.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(books["desc_cleaned"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_npz(MATRIX_PATH, matrix)
    with VECTORIZER_PATH.open("wb") as file:
        pickle.dump(vectorizer, file)
    books[["work_key", "genres"]].to_parquet(METADATA_PATH, index=False)
    print(
        f"Saved {matrix.shape[0]} books and {matrix.shape[1]} TF-IDF features "
        f"({matrix.nnz} nonzero values) to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()