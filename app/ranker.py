"""
Ranks Semantic Scholar papers using topic relevance, Semantic Scholar's
original search order, citation impact, recency, and user intent.

Relevance is deliberately weighted most heavily so that highly cited but
off-topic papers do not outrank papers that directly match the user's query.
"""

import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


CURRENT_YEAR = datetime.now().year

STOPWORDS = {
    "about",
    "after",
    "among",
    "and",
    "are",
    "based",
    "between",
    "for",
    "from",
    "into",
    "its",
    "of",
    "on",
    "or",
    "the",
    "their",
    "this",
    "through",
    "to",
    "using",
    "with",
}


def _query_terms(query: str) -> Set[str]:
    words = re.findall(r"[a-zA-Z0-9-]+", query.lower())

    return {
        word
        for word in words
        if len(word) >= 4 and word not in STOPWORDS
    }


def _text_relevance(text: str, terms: Set[str]) -> float:
    if not text or not terms:
        return 0.0

    text = text.lower()

    matches = sum(
        1
        for term in terms
        if term in text
    )

    return matches / len(terms)


def _score(
    paper: Dict[str, Any],
    query_terms: Set[str],
    intent: Optional[str],
    original_position: int,
) -> float:

    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""

    title_relevance = _text_relevance(
        title,
        query_terms,
    )

    abstract_relevance = _text_relevance(
        abstract,
        query_terms,
    )

    # Semantic Scholar already returns results in a relevance-oriented order.
    # Earlier papers therefore receive a modest bonus.
    semantic_scholar_bonus = 1.0 / (1 + original_position)

    citations = paper.get("citationCount") or 0

    # Logarithm prevents enormous citation counts from dominating everything.
    citation_score = math.log1p(citations)

    year = paper.get("year")
    age = (
        max(0, CURRENT_YEAR - year)
        if year
        else 20
    )

    recency_score = max(
        0.0,
        1.0 - (age / 10.0),
    )

    has_abstract = 1.0 if abstract else 0.0

    # Relevance dominates the default score.
    score = (
        title_relevance * 8.0
        + abstract_relevance * 4.0
        + semantic_scholar_bonus * 2.0
        + citation_score * 0.4
        + recency_score * 0.8
        + has_abstract * 0.2
    )

    if intent == "foundational":
        # Foundational requests care more about established impact.
        score += citation_score * 1.5

    elif intent == "gaps":
        # Research-gap requests favor recent literature.
        score += recency_score * 2.5

    return score


def rank_papers(
    papers: List[Dict[str, Any]],
    query: str = "",
    intent: Optional[str] = None,
) -> List[Dict[str, Any]]:

    if not papers:
        return []

    terms = _query_terms(query)

    indexed_papers = list(enumerate(papers))

    ranked = sorted(
        indexed_papers,
        key=lambda item: _score(
            item[1],
            terms,
            intent,
            item[0],
        ),
        reverse=True,
    )

    return [
        paper
        for _, paper in ranked
    ]