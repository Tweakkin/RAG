*This project has been created as part of the 42 curriculum by yahya.*

# RAG against the machine

## Description

A local retrieval-augmented generation pipeline for the vLLM 0.10.1 source
tree. It searches Python code and documentation for relevant excerpts, reports
their original file paths and character offsets, and uses Qwen3-0.6B to answer
from those excerpts. It can also measure retrieval recall against labeled
datasets.

## Instructions

Use Python 3.10 or later and a recent version of [uv](https://docs.astral.sh/uv/).
Run commands from the repository root.

```bash
make install   # uv sync
make lint
```

`uv sync` creates `uv.lock` if it is missing. The defense checklist requires
committing the generated lockfile.

Place the supplied vLLM repository under `data/raw/vllm-0.10.1/`. Place the
question files under `data/datasets/UnansweredQuestions/` and their references
under `data/datasets/AnsweredQuestions/`. If the attachment extracted to
`datasets_public/public/`, copy those two directories into `data/datasets/`.
The evaluation files, index, results, and model weights stay out of Git.

The first generation command loads `Qwen/Qwen3-0.6B` with Transformers and
PyTorch on CPU. If the weights are cached, generation can run offline.
`--model` accepts a local model directory or a compatible Hugging Face model.
Retrieval and evaluation run independently of the model.

## Example usage

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How do I enable prefix caching?" --k 5
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results/UnansweredQuestions --k 10
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run python -m src answer "How do I enable prefix caching?" --k 5
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Use `uv run python -m src <command> -- --help` to see options for a command.
`--corpus_path` and `--index_path` override the default input corpus and index
where applicable. Single-query results and evaluation support `--output_path`.
Batch commands preserve the input filename and accept `--save_directory`.
By default, batch outputs are grouped by the input's parent directory.
`make run ARGS='search "How does paging work?" --k 5'` is a shortcut.
`make debug ARGS='...'` runs the CLI under pdb.

## System architecture

Ingestion reads UTF-8 files. The chunker hands each snippet and its exact file
span to the indexer, which saves per-file SHA-256 hashes, chunks and token
counts. Retrieval loads the saved data and builds term-to-chunk postings.
Pydantic validates the index, question datasets, search results, and answers.
The answer stage packs the highest-ranked excerpts that fit the model's input
budget. Its exported sources are the excerpts passed to the model.

## Chunking strategy

Python files use the AST to find top-level statements and include decorators.
Comments between statements remain searchable. If Python parsing fails, prose
chunking still indexes the file. Markdown and text use blank lines to find
paragraphs and group adjacent ones up to the chunk limit. Both strategies split
oversized blocks and store half-open character ranges in the original text.

## Retrieval method

TF-IDF scores matching chunks using log-scaled term frequency, smoothed inverse
document frequency, and mild chunk-length normalization. Full identifiers and
their snake_case or camelCase components are searchable. The scorer downweights
common question words, gives extra weight to exact code symbols, and adds a
filename-match bonus. It scores both code and prose for every query without
using dataset labels or answers.

## Design decisions and challenges

- Character offsets must survive newline handling and Windows path separators.
  Reading unchanged UTF-8 text preserves offsets. Project-relative paths use
  forward slashes so they match the grader's source locations.
- The index reuses chunks and token counts for unchanged files and removes
  deleted files. Corpus-wide weights are refreshed when search loads the updated
  index. Atomic writes keep a failed run from leaving a half-written JSON file.
- Long excerpts can crowd out evidence. The generator checks the full formatted
  prompt against the token budget and reserves room for the answer. Qwen runs
  with thinking disabled and a prompt that asks for excerpt citations and an
  explicit acknowledgment when the evidence is insufficient.

## Performance analysis

Public vLLM datasets, scored with this project's evaluator using exact file
paths and character IoU of at least 0.05.

| Maximum chunk size | Dataset | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2,000 | Docs | 100 | 55.0% | 85.0% | 88.0% | 90.0% |
| 2,000 | Code | 99 | 46.5% | 67.7% | 75.8% | 82.8% |
| 1,200 | Docs | 100 | 57.0% | 78.0% | 82.0% | 86.0% |
| 1,200 | Code | 99 | 44.4% | 62.6% | 66.7% | 81.8% |

The default 2,000-character configuration indexed 1,969 files in about 11
seconds. The 199 public queries took about 12 seconds of retrieval time after
loading the index. This exceeds the subject's public recall thresholds of 80%
for docs and 50% for code within its indexing and retrieval time limits.
The official defense score is computed separately by the supplied grader.

## Resources

- [Python AST documentation](https://docs.python.org/3/library/ast.html)
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/)

AI assistance was used to review the subject, finish the CLI and JSON models,
refine chunking and retrieval, implement the Qwen generation stage, and
benchmark the pipeline and check edge cases.
