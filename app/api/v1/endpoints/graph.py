"""
Graph processing API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Optional

from app.models.graph import (
    GraphBuildRequest, GraphProcessRequest, GraphOptimizeRequest, AliasListRequest,
    GraphOperationResponse, GraphStatusResponse
)
from app.services.graph_service import GraphService

router = APIRouter()

def get_graph_service() -> GraphService:
    """Dependency to get graph service instance."""
    return GraphService()

@router.post("/build", response_model=GraphOperationResponse)
async def build_graph(
    request: GraphBuildRequest,
    background_tasks: BackgroundTasks,
    service: GraphService = Depends(get_graph_service)
):
    """
    Build entity graph from raw entity extraction results.

    This endpoint takes entity extraction results and builds a NetworkX graph,
    handling entity and relationship records, merging duplicates, and converting
    to dictionary format.
    """
    try:
        result = await service.build_graph(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building graph: {str(e)}")

@router.post("/process", response_model=GraphOperationResponse)
async def process_graph(
    request: GraphProcessRequest,
    background_tasks: BackgroundTasks,
    service: GraphService = Depends(get_graph_service)
):
    """
    Process entity graph with abbreviation handling and pattern extraction.

    This endpoint identifies abbreviations using GPT-4o-mini, extracts patterns
    using o3-mini, clusters similar patterns, and merges abbreviation nodes
    with their full name counterparts.
    """
    try:
        result = await service.process_graph(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing graph: {str(e)}")

@router.post("/optimize", response_model=GraphOperationResponse)
async def optimize_graph(
    request: GraphOptimizeRequest,
    background_tasks: BackgroundTasks,
    service: GraphService = Depends(get_graph_service)
):
    """
    Optimize entity graph descriptions through sentence clustering and merging.

    This endpoint splits descriptions into sentences, uses embeddings to cluster
    similar sentences, employs LLM to merge similar clusters, and compresses
    the overall graph size while maintaining information quality.
    """
    try:
        result = await service.optimize_graph(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing graph: {str(e)}")

@router.post("/build-aliases", response_model=GraphOperationResponse)
async def build_alias_list(
    request: AliasListRequest,
    background_tasks: BackgroundTasks,
    service: GraphService = Depends(get_graph_service)
):
    """
    Build bidirectional alias dictionary from processing logs.

    This endpoint extracts abbreviation information from processing logs
    and creates a bidirectional mapping between abbreviations and full names.
    """
    try:
        result = await service.build_alias_list(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building alias list: {str(e)}")

@router.get("/status", response_model=GraphStatusResponse)
async def get_graph_status(
    graph_path: str = "data/output/optimized_entity_graph.json",
    service: GraphService = Depends(get_graph_service)
):
    """
    Get status information for a graph file.

    Returns information about whether the graph file exists, its size,
    last modification time, validity, and basic statistics.
    """
    try:
        result = await service.get_graph_status(graph_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting graph status: {str(e)}")

@router.get("/pipeline", response_model=dict)
async def run_full_pipeline(
    entities_file: str = "data/input/entities_result.json",
    background_tasks: BackgroundTasks = None,
    service: GraphService = Depends(get_graph_service)
):
    """
    Run the complete graph processing pipeline.

    This endpoint runs all graph processing steps in sequence:
    1. Build graph from entities
    2. Process graph (abbreviations and patterns)
    3. Optimize graph descriptions
    4. Build alias dictionary
    """
    try:
        results = {}

        # Step 1: Build graph
        build_request = GraphBuildRequest(entities_file=entities_file)
        build_result = await service.build_graph(build_request)
        results["build"] = build_result

        if not build_result.success:
            return {"success": False, "message": "Pipeline failed at build step", "results": results}

        # Step 2: Process graph
        process_request = GraphProcessRequest(input_graph=build_result.output_path)
        process_result = await service.process_graph(process_request)
        results["process"] = process_result

        if not process_result.success:
            return {"success": False, "message": "Pipeline failed at process step", "results": results}

        # Step 3: Optimize graph
        optimize_request = GraphOptimizeRequest(input_graph=process_result.output_path)
        optimize_result = await service.optimize_graph(optimize_request)
        results["optimize"] = optimize_result

        if not optimize_result.success:
            return {"success": False, "message": "Pipeline failed at optimize step", "results": results}

        # Step 4: Build aliases
        alias_request = AliasListRequest()
        alias_result = await service.build_alias_list(alias_request)
        results["aliases"] = alias_result

        return {
            "success": True,
            "message": "Full pipeline completed successfully",
            "final_graph": optimize_result.output_path,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running pipeline: {str(e)}")