from src.ingestion import ingest_path
from src.chunking import convert_py_to_chunks, convert_to_chunks
from src.tokenizer import prepare_chunks, score_chunks, get_top_chunks, tokenize
import json
from pathlib import Path


def main() -> int:
    """Prepare corpus statistics once and print ranked matches for a question."""
    # Stop here if ingestion fails; later steps need valid file contents.
    try:
        files = ingest_path("data/raw/vllm-0.10.1")
    except Exception as error:
        print(f"Could not read the corpus: {error}")
        return 1

    if files is None:
        print("Could not read the corpus.")
        return 1

    if not files:
        print("No supported files were found in the corpus.")
        return 0

    # print("Files read:", len(files))
    # print("Characters read:", sum(len(text) for text in files.values()))

    # for file_path in list(files)[:5]:
        # print(file_path)

    # Stop if chunking fails instead of using an undefined all_chunks.
    try:
        all_chunks = convert_py_to_chunks(files) + convert_to_chunks(files)
        # print("Chunks len:", len(all_chunks))
        # print(
        #     "Largest chunk:",
        #     max((len(chunk["text"]) for chunk in all_chunks), default=0),
        # )
    except Exception as error:
        print(f"Could not create chunks: {error}")
        return 1

    if not all_chunks:
        print("No searchable chunks were created.")
        return 0

    # Prepare corpus statistics once so later questions can reuse them.
    try:
        chunks_tf, idf = prepare_chunks(all_chunks)
        # Create the output folder if it doesn't exist.
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # Collect everything needed to search later.
        index = {
            "chunks": all_chunks,
            "chunks_tf": chunks_tf,
            "idf": idf,
        }

        # Save it to a JSON file.
        with open("data/processed/index.json", "w", encoding="utf-8") as file:
            json.dump(index, file, indent=2)
    except Exception as error:
        print(f"Could not prepare the corpus: {error}")
        return 1

    # Scores stay in the same order as all_chunks until ranking.
    question = "What command can be used to evaluate the accuracy of a quantized model using lm_eval with vLLM?"
    try:
        scores = score_chunks(question, chunks_tf, idf)
        for chunk, score in zip(all_chunks, scores):
            if chunk["file_path"].endswith(("fp8.md", "int8.md")):
                print(chunk["file_path"], chunk["first_character_index"], score)
        for chunk, counts in zip(all_chunks, chunks_tf):
            if chunk["first_character_index"] in (1907, 1873):
                if chunk["file_path"].endswith(("fp8.md", "int8.md")):
                    print(chunk["file_path"])
                    for word in set(tokenize(question)):
                        print(word, counts.get(word, 0))
        top_chunks = get_top_chunks(all_chunks, scores, k=5)
        for chunk in top_chunks:
            print(chunk["file_path"], f": {chunk["first_character_index"]}-{chunk["last_character_index"]}")
    except Exception as error:
        print(f"Search failed: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
