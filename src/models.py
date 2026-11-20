"""Validated data exchanged between the RAG pipeline stages."""

from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_CHUNK_SIZE = 2000


class MinimalSource(BaseModel):
    """Locate a source using a half-open range of character offsets."""

    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "MinimalSource":
        """Require a nonempty forward range."""
        if self.last_character_index <= self.first_character_index:
            raise ValueError("Source end must be greater than its start.")
        return self


class Chunk(MinimalSource):
    """Store the original snippet alongside its location."""

    text: str


class UnansweredQuestion(BaseModel):
    """Store a question and its stable identifier."""

    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str

    @model_validator(mode="before")
    @classmethod
    def reject_partial_answer(cls, data: object) -> object:
        """Keep invalid answered questions from becoming unanswered ones."""
        if (cls is UnansweredQuestion and isinstance(data, dict)
                and ("answer" in data or "sources" in data)):
            raise ValueError("Answered questions need an answer and sources.")
        return data


class AnsweredQuestion(UnansweredQuestion):
    """Store a reference question, answer, and supporting sources."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Read either a reference dataset or a dataset of questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(UnansweredQuestion):
    """Record the sources ranked for one question."""

    question_id: str
    retrieved_sources: list[MinimalSource]

    @field_validator("retrieved_sources")
    @classmethod
    def validate_spans(
        cls, sources: list[MinimalSource],
    ) -> list[MinimalSource]:
        """Enforce the grader's limit on retrieved spans."""
        for source in sources:
            size = source.last_character_index - source.first_character_index
            if size > MAX_CHUNK_SIZE:
                raise ValueError(f"Source exceeds {MAX_CHUNK_SIZE} characters")
        return sources


class MinimalAnswer(MinimalSearchResults):
    """Record an answer alongside the sources shown to the model."""

    answer: str


class StudentSearchResults(BaseModel):
    """Export ranked sources with the requested result limit."""

    search_results: list[MinimalSearchResults]
    k: int = Field(ge=0)


class StudentSearchResultsAndAnswer(BaseModel):
    """Export answers and their supporting sources."""

    search_results: list[MinimalAnswer]
    k: int = Field(ge=0)


class IndexedFile(BaseModel):
    """Store a file fingerprint, chunks, and reusable term counts."""

    digest: str
    chunks: list[Chunk]
    term_counts: list[dict[str, int]]


class IndexData(BaseModel):
    """Persist a versioned index for an identified corpus and chunk size."""

    version: int
    corpus: str
    max_chunk_size: int = Field(ge=1, le=MAX_CHUNK_SIZE)
    files: dict[str, IndexedFile]
