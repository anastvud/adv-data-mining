"""Build Doc2Vec document vectors from the cleaned book descriptions.

The saved matrix and metadata have the same row order. The Doc2Vec model is
trained on the full dataset, so train it on training descriptions only before
inferring vectors and evaluating a predictive model on held-out books.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "BALANCED_CLEANED.parquet"
OUTPUT_DIR = ROOT / "data" / "doc2vec"
MATRIX_PATH = OUTPUT_DIR / "doc2vec_matrix.npy"
MODEL_PATH = OUTPUT_DIR / "doc2vec_model"
METADATA_PATH = OUTPUT_DIR / "doc2vec_metadata.parquet"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 40
SEED = 42


def main() -> None:
    books = pd.read_parquet(INPUT_PATH, columns=["work_key", "desc_cleaned", "genres"])
    if books[["work_key", "desc_cleaned", "genres"]].isna().any().any():
        raise ValueError("Book identifiers, descriptions, and genres must not be missing.")
    if books["desc_cleaned"].str.strip().eq("").any():
        raise ValueError("Book descriptions must not be empty.")

    tokenized_descriptions = [
        simple_preprocess(description) for description in books["desc_cleaned"]
    ]
    empty_rows = [
        row for row, tokens in enumerate(tokenized_descriptions) if not tokens
    ]
    if empty_rows:
        raise ValueError(
            f"Descriptions produced no tokens after preprocessing; row indexes: "
            f"{empty_rows[:10]}"
        )

    documents = [
        TaggedDocument(words=tokens, tags=[str(row)])
        for row, tokens in enumerate(tokenized_descriptions)
    ]
    model = Doc2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=1,
        epochs=EPOCHS,
        seed=SEED,
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=EPOCHS)

    matrix = np.vstack(
        [model.dv[str(row)] for row in range(len(books))]
    ).astype(np.float32, copy=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MATRIX_PATH, matrix)
    model.save(str(MODEL_PATH))
    books[["work_key", "genres"]].to_parquet(METADATA_PATH, index=False)
    print(
        f"Saved {matrix.shape[0]} books and {matrix.shape[1]} Doc2Vec features "
        f"to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()