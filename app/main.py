import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .search_engine import BanglaSemanticSearch
from .models import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global search system instance
search_system: Optional[BanglaSemanticSearch] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global search_system

    try:
        # Initialize search system
        logger.info("Initializing Bangla Semantic Search system...")
        search_system = BanglaSemanticSearch()

        # Check if we need to create schema and load data
        try:
            stats = search_system.get_statistics()
            stats["total_documents"] = 0
            if stats["total_documents"] == 0:
                logger.info("No documents found. Creating schema and loading data...")
                search_system.create_schema()
                search_system.load_and_index_documents()
            else:
                logger.info(f"Found {stats['total_documents']} existing documents")
        except Exception:
            logger.info("Creating new schema and loading documents...")
            search_system.create_schema()
            search_system.load_and_index_documents()

        logger.info("Search system initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize search system: {e}")
        yield
    finally:
        # Cleanup
        if search_system:
            search_system.close()
            logger.info("Search system closed")


# Initialize FastAPI app
app = FastAPI(
    title="Bangla Semantic Search API",
    description="A semantic search API for Bangla religious documents using fine-tuned BERT models",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint with API information"""
    return StatusResponse(
        status="active",
        message="Bangla Semantic Search API is running",
        statistics=search_system.get_statistics() if search_system else None,
    )


@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")

    try:
        stats = search_system.get_statistics()
        return StatusResponse(
            status="healthy", message="All systems operational", statistics=stats
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/search/semantic", response_model=SearchResponse)
async def semantic_search(search_query: SearchQuery):
    """Perform semantic search on Bangla documents"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")

    start_time = asyncio.get_event_loop().time()

    try:
        results = search_system.semantic_search(
            query=search_query.query,
            limit=search_query.limit,
            min_score=search_query.min_score,
        )

        search_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return SearchResponse(
            query=search_query.query,
            results=[DocumentResponse(**result) for result in results],
            total_results=len(results),
            search_time_ms=round(search_time, 2),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(search_query: HybridSearchQuery):
    """Perform hybrid search (semantic + keyword) on Bangla documents"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")

    start_time = asyncio.get_event_loop().time()

    try:
        results = search_system.hybrid_search(
            query=search_query.query, limit=search_query.limit, alpha=search_query.alpha
        )

        search_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return SearchResponse(
            query=search_query.query,
            results=[DocumentResponse(**result) for result in results],
            total_results=len(results),
            search_time_ms=round(search_time, 2),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search/semantic", response_model=SearchResponse)
async def semantic_search_get(
    q: str = Query(..., description="Search query in Bangla"),
    limit: int = Query(10, description="Maximum number of results", ge=1, le=50),
    min_score: float = Query(
        0.5, description="Minimum similarity score", ge=0.0, le=1.0
    ),
):
    """GET endpoint for semantic search"""
    search_query = SearchQuery(query=q, limit=limit, min_score=min_score)
    return await semantic_search(search_query)


@app.get("/search/hybrid", response_model=SearchResponse)
async def hybrid_search_get(
    q: str = Query(..., description="Search query in Bangla"),
    limit: int = Query(10, description="Maximum number of results", ge=1, le=50),
    alpha: float = Query(
        0.7, description="Weight for semantic vs keyword search", ge=0.0, le=1.0
    ),
):
    """GET endpoint for hybrid search"""
    search_query = HybridSearchQuery(query=q, limit=limit, alpha=alpha)
    return await hybrid_search(search_query)


@app.get("/documents/{document_id}/similar", response_model=SearchResponse)
async def find_similar_documents(
    document_id: str,
    limit: int = Query(
        5, description="Maximum number of similar documents", ge=1, le=20
    ),
):
    """Find documents similar to a given document"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")

    start_time = asyncio.get_event_loop().time()

    try:
        results = search_system.find_similar_documents(document_id, limit)
        search_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return SearchResponse(
            query=f"Similar to document {document_id}",
            results=[DocumentResponse(**result) for result in results],
            total_results=len(results),
            search_time_ms=round(search_time, 2),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Similar documents search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/statistics", response_model=Dict[str, Any])
async def get_statistics():
    """Get collection and system statistics"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")

    try:
        return search_system.get_statistics()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )


@app.post("/admin/reindex")
async def reindex_documents(background_tasks: BackgroundTasks):
    """Reindex all documents (admin endpoint)"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")

    def reindex_task():
        try:
            logger.info("Starting reindexing process...")
            search_system.create_schema()
            search_system.load_and_index_documents()
            logger.info("Reindexing completed successfully")
        except Exception as e:
            logger.error(f"Reindexing failed: {e}")

    background_tasks.add_task(reindex_task)

    return StatusResponse(status="accepted", message="Reindexing started in background")