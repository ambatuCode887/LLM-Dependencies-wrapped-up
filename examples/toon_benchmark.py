"""Compare JSON and TOON token counts using tiktoken's cl100k tokenizer."""
import tiktoken

from local_llm.toon.codec import TOON_AVAILABLE, to_toon


def main():
    if not TOON_AVAILABLE:
        raise SystemExit('Install TOON support first: python -m pip install -e ".[toon,benchmark]"')

    data = {
        "documents": [
            {"id": index, "source": f"manual-{index % 3}.md", "score": round(0.99 - index * 0.01, 2),
             "text": f"Retrieved passage {index}: local models can serve multiple applications."}
            for index in range(10)
        ]
    }
    json_text = __import__("json").dumps(data, separators=(",", ":"), ensure_ascii=False)
    toon_text = to_toon(data)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    json_tokens = len(tokenizer.encode(json_text))
    toon_tokens = len(tokenizer.encode(toon_text))
    reduction = (1 - toon_tokens / json_tokens) * 100 if json_tokens else 0

    print(f"Tokenizer: cl100k_base (estimate; use your model's tokenizer for exact counts)")
    print(f"JSON: {json_tokens} tokens")
    print(f"TOON: {toon_tokens} tokens")
    print(f"Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
