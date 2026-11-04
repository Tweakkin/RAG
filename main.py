from src.ingestion import ingest_path, convert_to_chunks, convert_py_to_chunks

def main():
    with open("practice.txt", "r") as f:
        text = f.read()
    print(text)
    passage = text[48:-1]
    print(passage)
try:
    files = ingest_path("data/raw/vllm-0.10.1")
except Exception as e:
    print(f"Error caught in main: {e}")

if files is not None:
    print("Files read:", len(files))
    print("Characters read:", sum(len(text) for text in files.values()))

for file_path in list(files)[:5]:
    print(file_path)
try:
    all_chunks = convert_py_to_chunks(files) + convert_to_chunks(files)
    print(f"Chunks len: {len(all_chunks)}")
    print("Largest chunk:", max((len(c["text"]) for c in all_chunks), default=0))
except Exception as e:
    print(f"Error caught in main: {e}")


for chunk in all_chunks:
    original = files[chunk["file_path"]]
    start = chunk["first_character_index"]
    end = chunk["last_character_index"]

    assert original[start:end] == chunk["text"]

print("All chunk positions match their source text.")