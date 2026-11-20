"""Expose the assignment's indexing, retrieval, and generation commands."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import RootModel
from tqdm import tqdm

from .evaluation import recall
from .generation import (
    AnswerGenerator, DEFAULT_MODEL, INSUFFICIENT, SourceReader,
)
from .index import INDEX_NAME, build_index, load_index
from .models import (
    MAX_CHUNK_SIZE, MinimalAnswer, MinimalSearchResults, MinimalSource,
    RagDataset,
    StudentSearchResults, StudentSearchResultsAndAnswer, UnansweredQuestion,
)
from .retrieval import Retriever, location, validate_k
from .storage import write_model

DEFAULT_CORPUS = "data/raw"
DEFAULT_INDEX = f"data/processed/{INDEX_NAME}"
SEARCH_DIRECTORY = "data/output/search_results"
ANSWER_DIRECTORY = "data/output/search_results_and_answer"


class RagCLI:
    """Build and query the vLLM retrieval-augmented generation pipeline."""

    def index(
        self,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        corpus_path: str = DEFAULT_CORPUS,
        index_path: str = DEFAULT_INDEX,
    ) -> None:
        """Build the index, refreshing only new or changed files."""
        def progress(paths: list[Path]) -> Iterable[Path]:
            """Report progress through the corpus."""
            yield from tqdm(paths, desc="Indexing", unit="file")

        files, chunks, updated = build_index(
            Path(corpus_path), Path(index_path), Path.cwd(),
            max_chunk_size, progress,
        )
        print(f"Indexed {files} files and {chunks} chunks "
              f"({updated} files rebuilt). Index saved to {index_path}")

    def search(
        self,
        query: str,
        k: int = 10,
        index_path: str = DEFAULT_INDEX,
        output_path: str = f"{SEARCH_DIRECTORY}/single.json",
    ) -> None:
        """Find relevant source locations for one question."""
        validate_k(k)
        retriever = Retriever(load_index(Path(index_path)))
        question = UnansweredQuestion(question=query)
        sources = [location(chunk) for chunk in retriever.search(query, k)]
        result = MinimalSearchResults(
            question_id=question.question_id, question=query,
            retrieved_sources=sources,
        )
        output = StudentSearchResults(search_results=[result], k=k)
        write_model(Path(output_path), output)
        print(output.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str | None = None,
        index_path: str = DEFAULT_INDEX,
    ) -> None:
        """Find sources for each dataset question and write one JSON file."""
        validate_k(k)
        path = Path(dataset_path)
        dataset = RagDataset.model_validate_json(path.read_bytes())
        retriever = Retriever(load_index(Path(index_path)))
        questions = tqdm(
            dataset.rag_questions, desc="Retrieving", unit="query",
        )
        results = [
            MinimalSearchResults(
                question_id=item.question_id,
                question=item.question,
                retrieved_sources=[
                    location(chunk)
                    for chunk in retriever.search(item.question, k)
                ],
            )
            for item in questions
        ]
        directory = (Path(SEARCH_DIRECTORY) / path.parent.name
                     if save_directory is None else Path(save_directory))
        output = directory / path.name
        write_model(output, StudentSearchResults(search_results=results, k=k))
        print(f"Saved student_search_results to {output}")

    def answer(
        self,
        query: str,
        k: int = 10,
        index_path: str = DEFAULT_INDEX,
        output_path: str = f"{ANSWER_DIRECTORY}/single.json",
        model: str = DEFAULT_MODEL,
        context_tokens: int = 3072,
        max_new_tokens: int = 192,
    ) -> None:
        """Answer one question using the most relevant source snippets."""
        validate_k(k)
        retriever = Retriever(load_index(Path(index_path)))
        question = UnansweredQuestion(question=query)
        chunks = retriever.search(query, k)
        text = INSUFFICIENT
        if chunks:
            generator = AnswerGenerator(model, context_tokens, max_new_tokens)
            text, chunks = generator.answer(query, chunks)
        result = MinimalAnswer(
            question_id=question.question_id, question=query, answer=text,
            retrieved_sources=[location(chunk) for chunk in chunks],
        )
        output = StudentSearchResultsAndAnswer(search_results=[result], k=k)
        write_model(Path(output_path), output)
        print(output.model_dump_json(indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str | None = None,
        corpus_path: str = DEFAULT_CORPUS,
        model: str = DEFAULT_MODEL,
        context_tokens: int = 3072,
        max_new_tokens: int = 192,
    ) -> None:
        """Generate a batch of answers from saved retrieval results."""
        path = Path(student_search_results_path)
        previous = StudentSearchResults.model_validate_json(path.read_bytes())
        reader = SourceReader(Path.cwd(), Path(corpus_path))
        generator: AnswerGenerator | None = None
        results: list[MinimalAnswer] = []
        questions = tqdm(
            previous.search_results, desc="Answering", unit="query",
        )
        for item in questions:
            text = INSUFFICIENT
            sources: list[MinimalSource] = []
            sources_to_read = item.retrieved_sources[:previous.k]
            if item.question.strip() and sources_to_read:
                chunks = [reader.chunk(source)
                          for source in sources_to_read]
                if generator is None:
                    generator = AnswerGenerator(
                        model, context_tokens, max_new_tokens,
                    )
                text, used = generator.answer(item.question, chunks)
                sources = [location(chunk) for chunk in used]
            results.append(MinimalAnswer(
                question_id=item.question_id, question=item.question,
                answer=text, retrieved_sources=sources,
            ))
        directory = (Path(ANSWER_DIRECTORY) / path.parent.name
                     if save_directory is None else Path(save_directory))
        output = directory / path.name
        write_model(output, StudentSearchResultsAndAnswer(
            search_results=results, k=previous.k,
        ))
        print(f"Saved student_search_results_and_answer to {output}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        output_path: str = "data/output/evaluation.json",
    ) -> None:
        """Print Recall@1, @3, @5, and @10 against a reference dataset."""
        results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_bytes(),
        )
        dataset_path_obj = Path(dataset_path)
        dataset = RagDataset.model_validate_json(dataset_path_obj.read_bytes())
        metrics = RootModel(recall(results, dataset))
        write_model(Path(output_path), metrics)
        print(metrics.model_dump_json(indent=2))
