"""Query processing API endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.models.query import (
    BatchQueryRequest,
    BatchQueryResponse,
    EntityInfoResponse,
    QueryHistoryResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.query_service import AliasSelectionRequiredError, QueryService


router = APIRouter()


def get_query_service() -> QueryService:
    return QueryService()


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
):
    """Run the end-to-end question answering pipeline."""

    try:
        return await service.ask_question(request)
    except AliasSelectionRequiredError as exc:
        detail = {
            "needs_alias_confirmation": True,
            "ambiguous_aliases": [
                {"alias": alias, "candidates": names}
                for alias, names in exc.aliases.items()
            ],
        }
        if exc.question:
            detail["question"] = exc.question
        raise HTTPException(status_code=409, detail=detail) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to client
        raise HTTPException(status_code=500, detail=f"Error answering question: {exc}") from exc


@router.post("/batch", response_model=BatchQueryResponse)
async def batch_query(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,  # Reserved for future async exports
    service: QueryService = Depends(get_query_service),
):
    """Process multiple questions sequentially."""

    try:
        return await service.batch_query(request)
    except AliasSelectionRequiredError as exc:
        detail = {
            "needs_alias_confirmation": True,
            "ambiguous_aliases": [
                {"alias": alias, "candidates": names}
                for alias, names in exc.aliases.items()
            ],
        }
        if exc.question:
            detail["question"] = exc.question
        raise HTTPException(status_code=409, detail=detail) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Error running batch query: {exc}") from exc


@router.get("/entities/{entity_name}", response_model=EntityInfoResponse)
async def get_entity_info(
    entity_name: str,
    include_neighbors: bool = True,
    max_distance: int = 1,
    service: QueryService = Depends(get_query_service),
):
    """Return graph information for a specific entity."""

    try:
        return service.get_entity_info(
            entity_name,
            include_neighbors=include_neighbors,
            max_distance=max_distance,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Error retrieving entity info: {exc}") from exc


@router.get("/history", response_model=QueryHistoryResponse)
async def get_query_history(
    limit: int = 100,
    service: QueryService = Depends(get_query_service),
):
    """Return stored question history and aggregate statistics."""

    try:
        return service.get_query_history(limit)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Error fetching history: {exc}") from exc
