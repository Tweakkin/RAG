import os
from typing import Dict, List
import ast

def ingest_path(path: str) -> Dict[str, str] | None:
    try:
        if not os.path.isdir(path):
            print(f"Not a valid directory: {path}")
            return None
        files_saved: Dict[str, str] = {}
        for curr_folder, sub_folders, filenames in os.walk(path):
            for filename in filenames:
                try:
                    if filename.endswith((".txt", ".py", ".md")):
                        file_path = os.path.join(curr_folder, filename)
                        with open(file_path, "r", encoding="utf-8", newline="") as f:
                            content = f.read()
                        files_saved[file_path] = content
                except (OSError, UnicodeDecodeError) as e:
                    print(f"Could not read {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error: {e}")
        return None

    return files_saved

def convert_to_chunks(files_content: Dict[str, str], max_chunk_size: int) -> List[Dict[str, str | int]]:
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("Chunk size must be between 1 and 2000.")
    result: List[Dict[str, str | int]] = []
    for file_path, content in files_content.items():
        counter: int = 0
        paragraphs = content.split("\n\n")
        for paragraph in paragraphs:
            if paragraph == "":
                counter += 2
                continue
            for num in range(0, len(paragraph), max_chunk_size):
                piece = paragraph[num: num+max_chunk_size]
                start = counter + num
                end = start + len(piece)
                last = result[-1] if result else None
                if (last is not None and last["file_path"] == file_path and end - last["first_character_index"] <= max_chunk_size):
                    last["last_character_index"] = end
                    last["text"] = content[last["first_character_index"]:end]
                else:
                    result.append({
                        "file_path": file_path,
                        "first_character_index": start,
                        "last_character_index": end,
                        "text": piece,
                    })
            counter += len(paragraph) + 2
    return result

def convert_py_to_chunks(files_content: Dict[str, str]) -> List[Dict[str, str | int]]:
    result: List[Dict[str, str | int]] = []
    for file_path, content in files_content.items():
        if file_path.endswith(".py"):
            splitted = content.splitlines(keepends=True)
            tree = ast.parse(content)
            tree_list = tree.body
            for node in tree_list:
                text = "".join(splitted[node.lineno - 1:node.end_lineno])
                first_index = sum(len(line) for line in splitted[:node.lineno - 1])
                end_index = first_index + len(text)
                result.append({
                    "file_path": file_path,
                    "first_character_index": first_index,
                    "last_character_index": end_index,
                    "text": text
                })
        else:
            continue
    return result
