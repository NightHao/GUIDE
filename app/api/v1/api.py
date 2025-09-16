"""
API v1 router aggregation.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import graph, query, admin

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(graph.router, prefix="/graph", tags=["graph"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
