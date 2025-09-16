"""
Pydantic models for query-related API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum

class QueryType(str, Enum):
    """Enumeration of query types."""
    GENERAL = "General Information Query"
    COMPARISON = "Comparison Query"
    COMMONALITY = "Commonality Query"

# Request Models
class QueryRequest(BaseModel):
    """Request model for single query."""
    question: str = Field(description="The question to ask")
    graph_path: Optional[str] = Field(
        default="data/output/optimized_entity_graph.json",
        description="Path to entity graph file"
    )
    subgraph_distance: Optional[int] = Field(
        default=1,
        description="Distance for subgraph extraction"
    )
    use_agentic_flow: Optional[bool] = Field(
        default=True,
        description="Whether to use agentic flow for processing"
    )
    alias_overrides: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping of abbreviations to preferred full names when multiple alias candidates exist",
        examples=[{"PA": "Platform Adapter"}]
    )

class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""
    questions: List[str] = Field(description="List of questions to ask")
    graph_path: Optional[str] = Field(
        default="data/output/optimized_entity_graph.json",
        description="Path to entity graph file"
    )
    subgraph_distance: Optional[int] = Field(
        default=1,
        description="Distance for subgraph extraction"
    )
    use_agentic_flow: Optional[bool] = Field(
        default=True,
        description="Whether to use agentic flow for processing"
    )
    save_results: Optional[bool] = Field(
        default=True,
        description="Whether to save results to file"
    )

class EntityInfoRequest(BaseModel):
    """Request model for entity information."""
    entity_name: str = Field(description="Name of the entity to retrieve")
    graph_path: Optional[str] = Field(
        default="data/output/optimized_entity_graph.json",
        description="Path to entity graph file"
    )
    include_neighbors: Optional[bool] = Field(
        default=True,
        description="Whether to include neighbor entities"
    )
    max_distance: Optional[int] = Field(
        default=1,
        description="Maximum distance for neighbor inclusion"
    )

# Response Models
class QueryIntentResponse(BaseModel):
    """Response model for query intent analysis."""
    category: QueryType = Field(description="Detected query category")
    explanation: str = Field(description="Explanation of the categorization")
    confidence: Optional[float] = Field(description="Confidence score")

class QueryResponse(BaseModel):
    """Response model for single query."""
    question: str = Field(description="The original question")
    answer: str = Field(description="The generated answer")
    intent: QueryIntentResponse = Field(description="Query intent analysis")
    processing_time: float = Field(description="Processing time in seconds")
    entities_used: List[str] = Field(description="List of entities used in answering")
    token_usage: Optional[Dict[str, int]] = Field(description="Token usage statistics")

class BatchQueryResponse(BaseModel):
    """Response model for batch queries."""
    total_questions: int = Field(description="Total number of questions processed")
    successful_answers: int = Field(description="Number of successful answers")
    failed_answers: int = Field(description="Number of failed answers")
    total_processing_time: float = Field(description="Total processing time in seconds")
    results: List[QueryResponse] = Field(description="Individual query results")
    saved_to: Optional[str] = Field(description="File path where results were saved")

class EntityInfo(BaseModel):
    """Model for entity information."""
    name: str = Field(description="Entity name")
    description: str = Field(description="Entity description")
    type: Optional[str] = Field(description="Entity type")
    distance: Optional[int] = Field(description="Distance from query entity")
    connections: List[str] = Field(description="Connected entity names")

class EntityInfoResponse(BaseModel):
    """Response model for entity information."""
    entity: EntityInfo = Field(description="Main entity information")
    neighbors: Optional[List[EntityInfo]] = Field(description="Neighbor entities")
    subgraph_stats: Optional[Dict[str, Any]] = Field(description="Subgraph statistics")

class QueryHistoryResponse(BaseModel):
    """Response model for query history."""
    total_queries: int = Field(description="Total number of queries")
    recent_queries: List[Dict[str, Any]] = Field(description="Recent query history")
    most_common_entities: List[str] = Field(description="Most commonly queried entities")
    average_processing_time: float = Field(description="Average processing time")
