"""
Pydantic models for graph-related API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from pathlib import Path

# Request Models
class GraphBuildRequest(BaseModel):
    """Request model for building entity graph."""
    entities_file: Optional[str] = Field(
        default="data/input/entities_result.json",
        description="Path to entities result JSON file"
    )
    output_path: Optional[str] = Field(
        default="data/intermediate/entity_graph.json",
        description="Output path for the built graph"
    )

class GraphProcessRequest(BaseModel):
    """Request model for processing entity graph."""
    input_graph: Optional[str] = Field(
        default="data/intermediate/entity_graph.json",
        description="Path to input entity graph"
    )
    output_graph: Optional[str] = Field(
        default="data/intermediate/processed_entity_graph.json",
        description="Output path for processed graph"
    )
    output_log: Optional[str] = Field(
        default="data/intermediate/log.json",
        description="Output path for processing log"
    )

class GraphOptimizeRequest(BaseModel):
    """Request model for optimizing entity graph."""
    input_graph: Optional[str] = Field(
        default="data/intermediate/processed_entity_graph.json",
        description="Path to input processed graph"
    )
    output_graph: Optional[str] = Field(
        default="data/output/optimized_entity_graph.json",
        description="Output path for optimized graph"
    )
    similarity_threshold: Optional[float] = Field(
        default=0.8,
        description="Similarity threshold for sentence clustering"
    )

class AliasListRequest(BaseModel):
    """Request model for building alias list."""
    log_file: Optional[str] = Field(
        default="data/intermediate/log.json",
        description="Path to processing log file"
    )
    output_file: Optional[str] = Field(
        default="data/intermediate/alias_dict.json",
        description="Output path for alias dictionary"
    )

# Response Models
class GraphOperationResponse(BaseModel):
    """Response model for graph operations."""
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Operation result message")
    output_path: Optional[str] = Field(description="Path to output file if applicable")
    processing_time: Optional[float] = Field(description="Processing time in seconds")
    statistics: Optional[Dict[str, Any]] = Field(description="Operation statistics")

class GraphStatsResponse(BaseModel):
    """Response model for graph statistics."""
    node_count: int = Field(description="Number of nodes in the graph")
    edge_count: int = Field(description="Number of edges in the graph")
    avg_node_degree: float = Field(description="Average node degree")
    max_node_degree: int = Field(description="Maximum node degree")
    graph_density: float = Field(description="Graph density")
    largest_component_size: int = Field(description="Size of largest connected component")

class GraphStatusResponse(BaseModel):
    """Response model for graph status."""
    exists: bool = Field(description="Whether the graph file exists")
    file_path: str = Field(description="Path to the graph file")
    file_size: Optional[int] = Field(description="File size in bytes")
    last_modified: Optional[str] = Field(description="Last modification time")
    is_valid: bool = Field(description="Whether the graph is valid JSON")
    stats: Optional[GraphStatsResponse] = Field(description="Graph statistics if available")