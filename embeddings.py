import numpy as np
from typing import Callable, List
from sklearn.feature_extraction.text import TfidfVectorizer


def make_tfidf_embedder(corpus: List[str]) -> Callable[[List[str]], np.ndarray]:
    """
    Returns a pure embedding function acting as the pre-trained contrastive encoder.
    Maps text to the Semantic Manifold (X).
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(corpus)

    def embedder(texts: List[str]) -> np.ndarray:
        vectors = vectorizer.transform(texts).toarray()
        # L2 Normalize to project onto unit hypersphere (Eq. 3 Justification)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)

    return embedder
