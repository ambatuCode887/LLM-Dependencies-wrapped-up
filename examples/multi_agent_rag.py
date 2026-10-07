"""Small offline-first RAG example using three configured local model profiles."""
import asyncio
import json
import math
from pathlib import Path
from typing import Any

from local_llm import LocalLLM

CONFIG = Path(__file__).resolve().parents[1] / "config" / "models.yaml"

DOCUMENTS = [
    {
        "id": "local-runtime",
        "text": "Local model inference runs against a model server on the same machine. "
        "After the runtime and model files are installed, inference does not require internet.",
    },
    {
        "id": "fallback",
        "text": "A model fallback chain tries configured local models in order. If a primary "
        "backend request fails, the router can try the next local model.",
    },
    {
        "id": "toon",
        "text": "TOON is a compact text representation for structured data. It can reduce "
        "prompt tokens for uniform arrays of objects, but savings depend on the data.",
    },
    {
        "id": "profiles",
        "text": "A model profile maps a logical agent role to an ordered chain of backend models. "
        "Different agents can use different profiles while sending model=local-llm.",
    },
]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def retrieve(
    documents: list[dict[str, Any]],
    document_vectors: list[list[float]],
    query_vector: list[float],
    top_k: int = 2,
) -> list[dict[str, Any]]:
    if len(documents) != len(document_vectors):
        raise ValueError("each document must have exactly one embedding")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    ranked = [
        {
            **document,
            "score": cosine_similarity(vector, query_vector),
        }
        for document, vector in zip(documents, document_vectors)
    ]
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]


async def main() -> None:
    async with (
        LocalLLM(CONFIG, profile="embedder") as embedder,
        LocalLLM(CONFIG, profile="planner") as researcher,
        LocalLLM(CONFIG, profile="coder") as answerer,
    ):
        document_result = await embedder.embeddings(
            model="local-llm",
            input=[document["text"] for document in DOCUMENTS],
        )
        document_vectors = [
            item["embedding"] for item in document_result["data"]
        ]
        query = "How does the local model fallback work without internet?"
        query_result = await embedder.embeddings(
            model="local-llm",
            input=query,
        )
        query_vector = query_result["data"][0]["embedding"]
        retrieved = retrieve(DOCUMENTS, document_vectors, query_vector)

        research_response = await researcher.chat(
            model="local-llm",
            messages=[
                {
                    "role": "system",
                    "content": "Use only the provided passages. Give concise, "
                    "evidence-based research notes in plain text. Do not invent facts.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": query, "passages": retrieved},
                        separators=(",", ":"),
                    ),
                },
            ],
            temperature=0,
            stream=False,
        )
        research_notes = research_response["choices"][0]["message"]["content"]
        if not isinstance(research_notes, str) or not research_notes.strip():
            raise ValueError("researcher returned empty or non-text notes")

        final_response = await answerer.chat(
            model="local-llm",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using only the research notes and "
                    "retrieved passages. If they do not contain the answer, say so.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": query,
                            "research_notes": research_notes,
                            "passages": retrieved,
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            temperature=0,
            stream=False,
        )

        print("Retrieved passages:")
        for passage in retrieved:
            print(f"- {passage['id']} (similarity {passage['score']:.3f})")
        print("\nAnswer:")
        print(final_response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
