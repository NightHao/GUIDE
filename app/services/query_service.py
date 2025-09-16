"""Query service bridging FastAPI endpoints with query processor logic."""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.query.processor import AliasResolutionRequired, QueryProcessor
from app.models.query import (
    BatchQueryRequest,
    BatchQueryResponse,
    EntityInfo,
    EntityInfoResponse,
    QueryHistoryResponse,
    QueryIntentResponse,
    QueryRequest,
    QueryResponse,
    QueryType,
)


class AliasSelectionRequiredError(Exception):
    """Bubble up alias disambiguation requests to the API layer."""

    def __init__(self, aliases: Dict[str, List[str]], question: Optional[str] = None):
        self.aliases = aliases
        self.question = question
        super().__init__("Alias selection required")


class QueryService:
    """Service layer orchestrating query-related operations."""

    def __init__(self) -> None:
        self.data_dir = settings.DATA_DIR
        self.input_dir = settings.INPUT_DIR
        self.intermediate_dir = settings.INTERMEDIATE_DIR
        self.output_dir = settings.OUTPUT_DIR

    async def ask_question(self, request: QueryRequest) -> QueryResponse:
        """Execute the full query pipeline for a single question."""

        processor = self._create_processor(request.graph_path, request.subgraph_distance)
        start_time = time.time()
        try:
            result_payload = await processor.ask_question(
                request.question,
                alias_overrides=request.alias_overrides or {},
            )
        except AliasResolutionRequired as exc:
            raise AliasSelectionRequiredError(exc.candidates, request.question)
        processing_time = time.time() - start_time
        return self._prepare_response(processor, request.question, result_payload, processing_time)

    async def batch_query(self, request: BatchQueryRequest) -> BatchQueryResponse:
        """Process multiple questions sequentially using a shared processor."""

        processor = self._create_processor(request.graph_path, request.subgraph_distance)
        responses: List[QueryResponse] = []
        total_processing_time = 0.0
        successful = 0

        for question in request.questions:
            start_time = time.time()
            try:
                payload = await processor.ask_question(question)
            except AliasResolutionRequired as exc:
                raise AliasSelectionRequiredError(exc.candidates, question)
            elapsed = time.time() - start_time
            total_processing_time += elapsed
            response = self._prepare_response(processor, question, payload, elapsed)
            if response.answer:
                successful += 1
            responses.append(response)

        saved_to: Optional[str] = None
        if request.save_results and responses:
            saved_path = self._save_batch_results(responses)
            saved_to = str(saved_path)

        return BatchQueryResponse(
            total_questions=len(request.questions),
            successful_answers=successful,
            failed_answers=len(request.questions) - successful,
            total_processing_time=total_processing_time,
            results=responses,
            saved_to=saved_to,
        )

    def get_entity_info(
        self,
        entity_name: str,
        *,
        graph_path: Optional[str] = None,
        include_neighbors: bool = True,
        max_distance: int = 1,
    ) -> EntityInfoResponse:
        """Retrieve detailed information for an entity from the graph."""

        graph_data = self._load_graph(graph_path)
        normalized = entity_name.strip().upper()
        entity_key = self._resolve_entity_key(graph_data, normalized)
        if not entity_key:
            raise ValueError(f"Entity '{entity_name}' not found in graph")

        entity = graph_data[entity_key]
        connections = [conn.get("target") for conn in entity.get("connections", [])]
        entity_info = EntityInfo(
            name=entity_key,
            description=entity.get("description", ""),
            type=entity.get("type"),
            distance=0,
            connections=connections,
        )

        neighbors: Optional[List[EntityInfo]] = None
        subgraph_stats: Optional[Dict[str, int]] = None
        if include_neighbors:
            bfs = self._bfs(graph_data, entity_key, max_distance)
            neighbors = []
            for node, distance in bfs.items():
                if node == entity_key or distance == 0:
                    continue
                node_data = graph_data.get(node, {})
                neighbors.append(
                    EntityInfo(
                        name=node,
                        description=node_data.get("description", ""),
                        type=node_data.get("type"),
                        distance=distance,
                        connections=[conn.get("target") for conn in node_data.get("connections", [])],
                    )
                )
            subgraph_stats = {
                "total_nodes": len(bfs),
                "total_edges": sum(len(graph_data[n].get("connections", [])) for n in bfs) // 2,
                "max_distance": max_distance,
            }

        return EntityInfoResponse(
            entity=entity_info,
            neighbors=neighbors,
            subgraph_stats=subgraph_stats,
        )

    def get_query_history(self, limit: int = 100) -> QueryHistoryResponse:
        """Return lightweight query history derived from stored chunks."""

        history_file = self.intermediate_dir / "entities_chunks.json"
        if not history_file.exists():
            return QueryHistoryResponse(
                total_queries=0,
                recent_queries=[],
                most_common_entities=[],
                average_processing_time=0.0,
            )

        with history_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        questions = list(data.keys())
        total_queries = len(questions)
        recent_items = []
        for question in questions[-limit:]:
            recent_items.append(
                {
                    "question": question,
                    "timestamp": None,
                    "processing_time": None,
                }
            )

        entity_counter: Counter[str] = Counter()
        for chunk in data.values():
            for entity_chunk in chunk.values():
                entity_counter.update(entity_chunk.keys())

        most_common_entities = [entity for entity, _ in entity_counter.most_common(10)]

        return QueryHistoryResponse(
            total_queries=total_queries,
            recent_queries=recent_items[-limit:],
            most_common_entities=most_common_entities,
            average_processing_time=0.0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_processor(self, graph_path: Optional[str], subgraph_distance: Optional[int]) -> QueryProcessor:
        resolved_graph = self._resolve_graph_path(graph_path)
        processor = QueryProcessor(
            subgraph_distance=subgraph_distance if subgraph_distance is not None else settings.DEFAULT_SUBGRAPH_DISTANCE,
            graph_path=str(resolved_graph),
        )
        processor.retrieved_chunks_path = str(self.input_dir / "retrieved_chunks_15.json")
        processor.entities_chunks_path = str(self.intermediate_dir / "entities_chunks.json")
        processor.flow_constructor.flow_operations.set_alias_path(
            str(self.intermediate_dir / "alias_dict.json")
        )
        processor.flow_constructor.flow_operations.set_alias_overrides({})
        return processor

    def _prepare_response(
        self,
        processor: QueryProcessor,
        question: str,
        payload: Dict[str, Any],
        processing_time: float,
    ) -> QueryResponse:
        category, explanation = processor.extract_intention(question)
        try:
            category_enum = QueryType(category)
        except ValueError:
            category_enum = QueryType.GENERAL

        intent = QueryIntentResponse(
            category=category_enum,
            explanation=explanation,
            confidence=None,
        )

        entities_used = payload.get("entities_used") or []
        if isinstance(entities_used, dict):
            entities_used = list(entities_used.keys())

        return QueryResponse(
            question=question,
            answer=str(payload.get("answer", "")),
            intent=intent,
            processing_time=processing_time,
            entities_used=list(entities_used),
            token_usage=None,
        )

    def _resolve_graph_path(self, graph_path: Optional[str]) -> Path:
        if graph_path:
            path = Path(graph_path)
            if not path.is_absolute():
                path = self.data_dir.parent / path
        else:
            path = Path(settings.DEFAULT_GRAPH_PATH)
            if not path.is_absolute():
                path = self.data_dir.parent / path
        return path.resolve()

    def _load_graph(self, graph_path: Optional[str]) -> Dict[str, Dict]:
        path = self._resolve_graph_path(graph_path)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_entity_key(self, graph: Dict[str, Dict], name: str) -> Optional[str]:
        if name in graph:
            return name
        for key in graph.keys():
            if key.strip().upper() == name:
                return key
        return None

    def _bfs(self, graph: Dict[str, Dict], start: str, max_distance: int) -> Dict[str, int]:
        distances = {start: 0}
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            if current_distance >= max_distance:
                continue
            for connection in graph[current].get("connections", []):
                neighbor = connection.get("target")
                if neighbor and neighbor not in distances:
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        return distances

    def _save_batch_results(self, responses: List[QueryResponse]) -> Path:
        output_file = self.output_dir / "batch_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump([response.model_dump() for response in responses], f, indent=2)
        return output_file
