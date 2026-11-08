from dataclasses import dataclass

@dataclass(frozen=True)
class Page:
    document: str
    page: int
    text: str

@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    page: int
    text: str

@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

@dataclass(frozen=True)
class Answer:
    text: str
    hits: list[Hit]
    citation_ids: list[int]
