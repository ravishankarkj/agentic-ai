from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    force_reembed: bool = False


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int
    reused_existing: bool


class QueryRequest(BaseModel):
    requestId: str = "req-12345"
    userId: str = "005xx000001SvU"
    query: str = Field(min_length=2, max_length=5000)


class SourceDocument(BaseModel):
    source: str
    score: float | None = None
    snippet: str


class QueryResult(BaseModel):
    query: str
    answer: str
    sources: list[SourceDocument]


class QueryResponse(BaseModel):
    requestId: str
    userId: str
    response: QueryResult
