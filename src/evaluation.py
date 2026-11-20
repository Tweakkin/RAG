"""Evaluate average per-question retrieval recall against reference spans."""

from .models import (
    AnsweredQuestion, MinimalSearchResults, MinimalSource, RagDataset,
    StudentSearchResults,
)

CUTOFFS = (1, 3, 5, 10)


def matches(result: MinimalSource, source: MinimalSource) -> bool:
    """Check for an identical file path and character IoU of at least 0.05."""
    if result.file_path != source.file_path:
        return False
    overlap = min(result.last_character_index, source.last_character_index)
    overlap -= max(result.first_character_index, source.first_character_index)
    union = max(result.last_character_index, source.last_character_index)
    union -= min(result.first_character_index, source.first_character_index)
    return overlap > 0 and 20 * overlap >= union


def recall(
    results: StudentSearchResults,
    dataset: RagDataset,
) -> dict[str, float]:
    """Average the fraction of expected sources found for each question."""
    by_id: dict[str, MinimalSearchResults] = {}
    for result in results.search_results:
        if result.question_id in by_id:
            raise ValueError(f"Duplicate result ID: {result.question_id}")
        by_id[result.question_id] = result

    totals = {cutoff: 0.0 for cutoff in CUTOFFS}
    seen: set[str] = set()
    count = 0
    for item in dataset.rag_questions:
        if not isinstance(item, AnsweredQuestion):
            raise ValueError("Evaluation needs a ground-truth answer dataset.")
        if item.question_id in seen:
            raise ValueError(f"Duplicate dataset ID: {item.question_id}")
        seen.add(item.question_id)
        if not item.sources:
            continue
        count += 1
        result = by_id.get(item.question_id)
        found = [] if result is None else result.retrieved_sources
        first_matches = [
            next(
                (rank for rank, match in enumerate(found[:max(CUTOFFS)], 1)
                 if matches(match, source)),
                max(CUTOFFS) + 1,
            )
            for source in item.sources
        ]
        for cutoff in CUTOFFS:
            totals[cutoff] += (
                sum(rank <= cutoff for rank in first_matches)
                / len(item.sources)
            )
    return {
        f"Recall@{cutoff}": total / count if count else 0.0
        for cutoff, total in totals.items()
    }
