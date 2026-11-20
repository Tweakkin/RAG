from src.ingestion import ingest_path
from src.chunking import convert_py_to_chunks, convert_to_chunks
from src.tokenizer import (
    prepare_chunks, score_chunks, get_top_chunks, tokenize_question,
)
import hashlib
import json
from pathlib import Path


def main() -> int:
    """Load or build the index and print ranked matches for a question."""
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

    # Store one current content hash under each file's path.
    file_hashes: dict[str, str] = {}
    for file_path, text in files.items():
        # Convert the file's text into the bytes SHA-256 expects.
        contents_bytes = text.encode("utf-8")

        # Calculate a hash object from those bytes.
        hash_object = hashlib.sha256(contents_bytes)

        # Get a hash string that can be saved in JSON and compared later.
        file_hashes[file_path] = hash_object.hexdigest()

    # Load the saved index if it already exists.
    cache_path = Path("data/processed/index.json")
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                index = json.load(file)
            all_chunks = index["chunks"]
            chunks_tf = index["chunks_tf"]
            idf = index["idf"]
            # print("Loaded saved index.")

            # An older index may have been saved before we added hashes.
            saved_hashes = index.get("file_hashes")
            if saved_hashes is None:
                print("No saved file hashes. Delete index.json to rebuild once.")
            else:
                # Compare every current file with its previously saved hash.
                for file_path, current_hash in file_hashes.items():
                    # A path absent from the saved dictionary returns None.
                    saved_hash = saved_hashes.get(file_path)

                    # No previous hash means this file is new to the index.
                    if saved_hash is None:
                        print(file_path, "is new")
                    # A different hash means the file's contents changed.
                    elif current_hash != saved_hash:
                        print(file_path, "was modified")
                    # An equal hash means the contents are unchanged.
                    else:
                        print(file_path, "is unchanged")

                # Reporting only: saved chunks are still reused at this step.
        except Exception as error:
            print(f"Could not load the saved index: {error}")
            return 1
    else:
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
                # Save hashes of the same files used to build these chunks.
                "file_hashes": file_hashes,
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
                    for word in set(tokenize_question(question)):
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
