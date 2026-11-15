"""Build SVD topic vectors from the cleaned book descriptions.

The saved matrix and metadata have the same row order. Both the TF-IDF
vectorizer and SVD model are fitted on the full dataset, so refit them on
training descriptions before evaluating a predictive model on held-out books.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "BALANCED_CLEANED.parquet"
OUTPUT_DIR = ROOT / "data" / "svd"
MATRIX_PATH = OUTPUT_DIR / "svd_matrix.npy"
VECTORIZER_PATH = OUTPUT_DIR / "svd_vectorizer.pkl"
SVD_MODEL_PATH = OUTPUT_DIR / "svd_model.pkl"
METADATA_PATH = OUTPUT_DIR / "svd_metadata.parquet"
N_COMPONENTS = 100


def main() -> None:
    books = pd.read_parquet(INPUT_PATH, columns=["work_key", "desc_cleaned", "genres"])
    if books[["work_key", "desc_cleaned", "genres"]].isna().any().any():
        raise ValueError("Book identifiers, descriptions, and genres must not be missing.")
    if books["desc_cleaned"].str.strip().eq("").any():
        raise ValueError("Book descriptions must not be empty.")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(books["desc_cleaned"])

    n_components = min(N_COMPONENTS, *tfidf_matrix.shape)
    if n_components < 1:
        raise ValueError("SVD requires at least one book and one TF-IDF feature.")

    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    matrix = svd_model.fit_transform(tfidf_matrix)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MATRIX_PATH, matrix)
    with VECTORIZER_PATH.open("wb") as file:
        pickle.dump(vectorizer, file)
    with SVD_MODEL_PATH.open("wb") as file:
        pickle.dump(svd_model, file)
    books[["work_key", "genres"]].to_parquet(METADATA_PATH, index=False)
    print(
        f"Saved {matrix.shape[0]} books and {matrix.shape[1]} SVD topic vectors "
        f"to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()