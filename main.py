from embeddings import make_tfidf_embedder
from manifolds import make_pca_projector
from distances import cosine_distance, make_mahalanobis_distance
from traversals import semantic_decay_traversal
from visualizations import plot_manifolds


def build_prompt(query: str, corpus: list, hops: dict) -> str:
    primary = [corpus[i] for i in hops.get(0, frozenset())]
    structural = [corpus[i] for t, docs in hops.items() if t > 0 for i in docs]

    return f"""<primary_semantic_context>
{chr(10).join(f"- {doc}" for doc in primary)}
</primary_semantic_context>

<structurally_linked_context>
{chr(10).join(f"- {doc}" for doc in structural)}
</structurally_linked_context>

System: Synthesize the primary context to answer the user's query: "{query}".
Use the structurally linked context to draw broader multi-hop connections if the primary context is missing direct evidence."""


def main():
    # 1. Corpus designed to explicitly test multi-hop bottlenecks
    corpus = [
        "The latest iPhone battery uses lithium and advanced chemistry.",  # D0
        "Lithium is a key component of the global battery supply chain.",  # D1
        "Cobalt mining is essential for the battery supply chain.",  # D2
        "Electric vehicles rely on extensive battery supply chains.",  # D3
        "Apples and bananas are yellow fruits.",  # D4 (Noise)
        "The weather in London is rainy today.",  # D5 (Noise)
    ]
    query = "iPhone battery materials"

    # 2. Dependency Injection: Embeddings & Projections
    embedder = make_tfidf_embedder(corpus + [query])
    X = embedder(corpus)
    q = embedder([query])[0]

    # Project to d=2 bottleneck
    projector = make_pca_projector(d=2)
    Z, inv_cov = projector(X)

    # 3. Distance Metrics passed as pure closures
    dist_X = cosine_distance
    dist_Z = make_mahalanobis_distance(inv_cov)

    # 4. Traversal
    hops = semantic_decay_traversal(
        q=q,
        X=X,
        Z=Z,
        dist_X_fn=dist_X,
        dist_Z_fn=dist_Z,
        tau=0.75,  # Semantic similarity threshold limits starting point to D0 only
        epsilon=1.8,  # Structural Mahalanobis jump limit
        gamma=1.5,  # Decay factor (drift multiplier > 1 allows thematic walking)
        max_hops=3,
    )

    print("=== Multi-hop Traversal Result ===")
    for t in sorted(hops.keys()):
        print(f"Hop {t}:")
        for d in hops[t]:
            print(f"  [D{d}] {corpus[d]}")

    print("\n=== Section 6 LLM Synthesis Prompt ===")
    print(build_prompt(query, corpus, hops))

    # 5. Visualization Map
    plot_manifolds(X, Z, hops, corpus)


if __name__ == "__main__":
    main()
