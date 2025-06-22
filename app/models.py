from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query in Bangla", min_length=1)
    limit: Optional[int] = Field(10, description="Maximum number of results", ge=1, le=50)
    min_score: Optional[float] = Field(0.5, description="Minimum similarity score", ge=0.0, le=1.0)

class HybridSearchQuery(BaseModel):
    query: str = Field(..., description="Search query in Bangla", min_length=1)
    limit: Optional[int] = Field(10, description="Maximum number of results", ge=1, le=50)
    alpha: Optional[float] = Field(0.7, description="Weight for semantic vs keyword search", ge=0.0, le=1.0)

class DocumentResponse(BaseModel):
    title: str
    sub_title: str
    answer: str
    similarity_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    distance: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    results: List[DocumentResponse]
    total_results: int
    search_time_ms: float
    timestamp: str

class StatusResponse(BaseModel):
    status: str
    message: str
    statistics: Optional[Dict[str, Any]] = None