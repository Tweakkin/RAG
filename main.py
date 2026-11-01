from src.ingestion import ingest_path, convert_to_chunks

def main():
    with open("practice.txt", "r") as f:
        text = f.read()
    print(text)
    passage = text[48:-1]
    print(passage)

files = ingest_path("data/raw/vllm-0.10.1")

if files is not None:
    print("Files read:", len(files))
    print("Characters read:", sum(len(text) for text in files.values()))

for file_path in list(files)[:5]:
    print(file_path)

print(convert_to_chunks(files, 2000)[:5])
