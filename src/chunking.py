from typing import Dict, List
import ast

# Split .md and .txt files into paragraphs if they fit and split
# oversized paragraphs
def convert_to_chunks(files_content: Dict[str, str], max_chunk_size: int = 2000) -> List[Dict[str, str | int]]:
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("Chunk size must be between 1 and 2000.")
    result: List[Dict[str, str | int]] = []
    for file_path, content in files_content.items():
        if file_path.endswith((".md", ".txt")):
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
        else:
            continue
    return result


# Splitting .py files, using Python AST to identify statements, functions...
# Split oversized sections by char count
def convert_py_to_chunks(files_content: Dict[str, str], max_chunk_size: int = 2000) -> List[Dict[str, str | int]]:
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("Chunk size must be between 1 and 2000.")
    result: List[Dict[str, str | int]] = []
    for file_path, content in files_content.items():
        if file_path.endswith(".py"):
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                print(f"Could not parse {file_path}: {e}")
                continue
            splitted = content.splitlines(keepends=True)
            tree_list = tree.body
            for node in tree_list:
                text = "".join(splitted[node.lineno - 1:node.end_lineno])
                first_index = sum(len(line) for line in splitted[:node.lineno - 1])
                for num in range(0, len(text), max_chunk_size):
                    piece = text[num:num + max_chunk_size]
                    start = first_index + num
                    result.append({
                        "file_path": file_path,
                        "first_character_index": start,
                        "last_character_index": start + len(piece),
                        "text": piece
                    })
        else:
            continue
    return result
