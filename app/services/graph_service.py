"""Graph service providing high-level graph operations."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.graph import builder as graph_builder
from app.core.graph.alias_builder import AliasBuilder
from app.core.graph.optimizer import NodeDescriptionOptimizer
from app.core.graph.processor import EntityGraphProcessor
from app.models.graph import (
    AliasListRequest,
    GraphBuildRequest,
    GraphOperationResponse,
    GraphOptimizeRequest,
    GraphProcessRequest,
    GraphStatsResponse,
    GraphStatusResponse,
)


class GraphService:
    """Service for graph operations."""

    def __init__(self) -> None:
        self.data_dir = settings.DATA_DIR

    async def build_graph(self, request: GraphBuildRequest) -> GraphOperationResponse:
        """Build an entity graph from extraction results."""

        start_time = time.time()
        entities_file = self._resolve_path(request.entities_file)
        output_path = self._resolve_path(request.output_path)

        if not entities_file.exists():
            return GraphOperationResponse(
                success=False,
                message=f"Input file not found: {entities_file}",
                processing_time=time.time() - start_time,
            )

        try:
            graph_dict = await asyncio.to_thread(
                graph_builder.build_entity_graph,
                str(entities_file),
                str(output_path),
            )
            processing_time = time.time() - start_time
            statistics = {
                "input_file": str(entities_file),
                "output_file": str(output_path),
                **self._graph_overview(graph_dict),
            }
            return GraphOperationResponse(
                success=True,
                message="Graph built successfully",
                output_path=str(output_path),
                processing_time=processing_time,
                statistics=statistics,
            )
        except Exception as exc:  # pragma: no cover - propagated to client
            return GraphOperationResponse(
                success=False,
                message=f"Error building graph: {exc}",
                processing_time=time.time() - start_time,
            )

    async def process_graph(self, request: GraphProcessRequest) -> GraphOperationResponse:
        """Process entity graph with abbreviation handling and pattern extraction."""

        start_time = time.time()
        input_graph = self._resolve_path(request.input_graph)
        output_graph = self._resolve_path(request.output_graph)
        output_log = self._resolve_path(request.output_log)

        if not input_graph.exists():
            return GraphOperationResponse(
                success=False,
                message=f"Input graph not found: {input_graph}",
                processing_time=time.time() - start_time,
            )

        processor = EntityGraphProcessor(log_file_path=str(output_log))
        try:
            processed_graph = await processor.process_entity_graph(
                str(input_graph),
                str(output_graph),
            )
            processing_time = time.time() - start_time
            statistics = {
                "input_file": str(input_graph),
                "output_graph": str(output_graph),
                "output_log": str(output_log),
                **self._graph_overview(processed_graph),
            }
            return GraphOperationResponse(
                success=True,
                message="Graph processed successfully",
                output_path=str(output_graph),
                processing_time=processing_time,
                statistics=statistics,
            )
        except Exception as exc:  # pragma: no cover - propagated to client
            return GraphOperationResponse(
                success=False,
                message=f"Error processing graph: {exc}",
                processing_time=time.time() - start_time,
            )

    async def optimize_graph(self, request: GraphOptimizeRequest) -> GraphOperationResponse:
        """Optimize entity graph descriptions through sentence clustering."""

        start_time = time.time()
        input_graph = self._resolve_path(request.input_graph)
        output_graph = self._resolve_path(request.output_graph)

        if not input_graph.exists():
            return GraphOperationResponse(
                success=False,
                message=f"Input graph not found: {input_graph}",
                processing_time=time.time() - start_time,
            )

        optimizer = NodeDescriptionOptimizer()
        try:
            entity_graph = optimizer.load_entity_graph(str(input_graph))
            optimized_graph = await optimizer.async_optimize_entity_graph(
                entity_graph,
                similarity_threshold=request.similarity_threshold,
            )
            optimizer.save_entity_graph(optimized_graph, str(output_graph))

            processing_time = time.time() - start_time
            statistics = {
                "input_file": str(input_graph),
                "output_file": str(output_graph),
                "similarity_threshold": request.similarity_threshold,
                **self._graph_overview(optimized_graph),
            }
            return GraphOperationResponse(
                success=True,
                message="Graph optimized successfully",
                output_path=str(output_graph),
                processing_time=processing_time,
                statistics=statistics,
            )
        except Exception as exc:  # pragma: no cover - propagated to client
            return GraphOperationResponse(
                success=False,
                message=f"Error optimizing graph: {exc}",
                processing_time=time.time() - start_time,
            )

    async def build_alias_list(self, request: AliasListRequest) -> GraphOperationResponse:
        """Build a bidirectional alias dictionary from processing logs."""

        start_time = time.time()
        log_file = self._resolve_path(request.log_file)
        output_file = self._resolve_path(request.output_file)

        if not log_file.exists():
            return GraphOperationResponse(
                success=False,
                message=f"Log file not found: {log_file}",
                processing_time=time.time() - start_time,
            )

        builder = AliasBuilder(str(log_file), str(output_file))
        try:
            alias_dict = await asyncio.to_thread(self._build_alias_dict, builder)
            processing_time = time.time() - start_time
            statistics = {
                "log_file": str(log_file),
                "output_file": str(output_file),
                "abbreviation_count": len(alias_dict.get("abbreviations", {})),
                "full_name_count": len(alias_dict.get("full_names", {})),
            }
            return GraphOperationResponse(
                success=True,
                message="Alias list built successfully",
                output_path=str(output_file),
                processing_time=processing_time,
                statistics=statistics,
            )
        except Exception as exc:  # pragma: no cover - propagated to client
            return GraphOperationResponse(
                success=False,
                message=f"Error building alias list: {exc}",
                processing_time=time.time() - start_time,
            )

    async def get_graph_status(self, graph_path: str) -> GraphStatusResponse:
        """Get status information for a graph file."""
        try:
            resolved_path = self._resolve_path(graph_path)
            if not resolved_path.exists():
                return GraphStatusResponse(
                    exists=False,
                    file_path=str(resolved_path),
                    is_valid=False
                )

            # Get file info
            file_size = resolved_path.stat().st_size
            last_modified = resolved_path.stat().st_mtime

            # Check if valid JSON
            is_valid = False
            stats = None
            try:
                with resolved_path.open('r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                is_valid = True

                # Calculate basic stats
                if isinstance(graph_data, dict):
                    stats = self._graph_stats_model(graph_data)

            except json.JSONDecodeError:
                is_valid = False

            return GraphStatusResponse(
                exists=True,
                file_path=str(resolved_path),
                file_size=file_size,
                last_modified=str(last_modified),
                is_valid=is_valid,
                stats=stats
            )

        except Exception as e:
            return GraphStatusResponse(
                exists=False,
                file_path=graph_path,
                is_valid=False
            )

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the project root."""

        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = self.data_dir.parent / path_obj
        return path_obj.resolve()

    def _graph_overview(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Produce lightweight graph statistics."""

        node_count = len(graph_dict)
        edge_count = sum(len(node.get("connections", [])) for node in graph_dict.values())
        undirected_edges = edge_count // 2
        avg_degree = edge_count / node_count if node_count else 0
        max_degree = max((len(node.get("connections", [])) for node in graph_dict.values()), default=0)

        return {
            "node_count": node_count,
            "edge_count": undirected_edges,
            "avg_node_degree": avg_degree,
            "max_node_degree": max_degree,
        }

    def _graph_stats_model(self, graph_dict: Dict[str, Any]) -> GraphStatsResponse:
        """Convert graph overview stats into a response model."""

        overview = self._graph_overview(graph_dict)
        node_count = overview["node_count"]
        edge_count = overview["edge_count"]
        density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0

        return GraphStatsResponse(
            node_count=node_count,
            edge_count=edge_count,
            avg_node_degree=overview["avg_node_degree"],
            max_node_degree=overview["max_node_degree"],
            graph_density=density,
            largest_component_size=node_count,
        )

    @staticmethod
    def _build_alias_dict(builder: AliasBuilder) -> Dict[str, Any]:
        """Helper for running alias generation in a background thread."""

        abbr_dict = builder.load_abbr_dict()
        alias_dict = builder.build_alias_dict(abbr_dict)
        builder.save_alias_dict(alias_dict)
        return alias_dict
