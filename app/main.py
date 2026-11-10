"""
FastAPI backend for the AI Research Assistant.

Run locally with:
    uvicorn app.main:app --reload

Then POST to /research, e.g.:
    curl -X POST http://localhost:8000/research \
      -H "Content-Type: application/json" \
      -d '{"query": "AI bias hiring algorithms", "limit": 5}'
"""
from typing import List, Optional
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.ranker import rank_papers
# On a clean clone, base mode works without separately downloaded adapter files.
# Set RESEARCH_ASSISTANT_MODEL=finetuned to use the saved research LoRA adapter.
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
MODEL_MODE = os.environ.get("RESEARCH_ASSISTANT_MODEL", "base").lower().strip()
if MODEL_MODE == "finetuned":
    from app.query_rewriter import rewrite_query
elif MODEL_MODE == "base":
    from app.query_rewriter_base import rewrite_query
else:
    raise ValueError("RESEARCH_ASSISTANT_MODEL must be 'base' or 'finetuned'")
from app.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarError,
    SemanticScholarRateLimitError,
)

app = FastAPI(title="AI Research Assistant")
scholar = SemanticScholarClient()


class QueryRequest(BaseModel):
    query: str
    limit: int = 10
    intent: Optional[str] = None  # "foundational" | "gaps" | None


class PaperResult(BaseModel):
    title: str
    authors: List[str]
    year: Optional[int]
    url: Optional[str]
    citation_count: Optional[int]
    abstract: Optional[str]


class ResearchResponse(BaseModel):
    original_query: str
    rewritten_query: str
    papers: List[PaperResult]
    summary: str
    response_time_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: QueryRequest):
    start_time = time.perf_counter()
    rewritten = rewrite_query(request.query)
    

    try:
        # Retrieve more candidates than the final requested count so that
        # the ranker has several papers to compare.
        candidate_limit = min(max(request.limit * 4, 20), 100)

        raw_papers = scholar.search_papers(
            rewritten,
            limit=candidate_limit,
        )

    except SemanticScholarRateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
        ) from exc

    except SemanticScholarError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    ranked = rank_papers(
        raw_papers,
        query=f"{request.query} {rewritten}",
        intent=request.intent,
    )[: request.limit]

    papers = [
        PaperResult(
            title=paper.get("title") or "Untitled",
            authors=[
                author.get("name", "")
                for author in (paper.get("authors") or [])
                if author.get("name")
            ],
            year=paper.get("year"),
            url=paper.get("url"),
            citation_count=paper.get("citationCount"),
            abstract=paper.get("abstract"),
        )
        for paper in ranked
    ]
    elapsed = time.perf_counter() - start_time

    return ResearchResponse(
        original_query=request.query,
        rewritten_query=rewritten,
        papers=papers,
        summary=_build_summary(papers),
        response_time_seconds=round(elapsed, 3) # type: ignore
    )


def _build_summary(papers: List[PaperResult]) -> str:
    if not papers:
        return "No relevant papers found. Try rephrasing your query."
    top = papers[0]
    return (
        f"Found {len(papers)} relevant papers. "
        f"The most relevant is '{top.title}'"
        + (f" ({top.year})" if top.year else "")
        + "."
    )
