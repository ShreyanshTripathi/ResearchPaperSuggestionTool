from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Paper(BaseModel):
    paper_id: int
    title: str
    authors: List[str]
    abstract: str
    url: str
    source: str
    published_date: Optional[datetime] = None
    relevance_score: float = 0.0
    categories: List[str] = Field(default_factory=list)
    model_config = {"populate_by_name": True}

    def short_repr(self) -> str:
        authors_str = ", ".join(self.authors[:2])
        if len(self.authors) > 2:
            authors_str += " et al."
        return f"[{self.paper_id}] {self.title} — {authors_str}"
    
class SearchRecord(BaseModel):
    query: str                                  # Original user query string
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    paper_ids: list[str] = Field(default_factory=list)   # IDs of papers found
    expanded_topics: list[str] = Field(default_factory=list)  # Agent 2 expansions

    def to_chroma_metadata(self) -> dict:
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "paper_ids": ",".join(self.paper_ids),
            "expanded_topics": ",".join(self.expanded_topics),
        }


class AgentResult(BaseModel):
    agent_name: str
    papers: list[Paper]
    reasoning: str = ""     # LLM explanation of why these papers were chosen
    metadata: dict = Field(default_factory=dict)


class ResearchSession(BaseModel):
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    focused_papers: list[Paper] = Field(default_factory=list)    # Agent 1
    broader_papers: list[Paper] = Field(default_factory=list)    # Agent 2
    interest_papers: list[Paper] = Field(default_factory=list)   # Agent 3
    interest_map_summary: str = ""