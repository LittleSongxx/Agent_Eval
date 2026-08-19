from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FieldDefinition(BaseModel):
    name: str
    type: str  # text / number / text_list / conversation / tool_call_list / tags / json
    required: bool = False
    description: str = ""


class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    sample_type: str
    field_schema: List[FieldDefinition]


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    sample_type: str
    field_schema: Optional[List[FieldDefinition]] = None
    row_count: int
    version: Optional[int] = 1
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetRowCreate(BaseModel):
    data: Dict[str, Any]


class DatasetRowResponse(BaseModel):
    id: int
    dataset_id: int
    row_index: int
    data: Dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetRowsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[DatasetRowResponse]


class AgentTraceImportRequest(BaseModel):
    """Raw Agent traces accepted by the normalized import endpoint."""

    traces: List[Dict[str, Any]] = Field(min_length=1)


class AgentTraceImportResponse(BaseModel):
    dataset_id: int
    imported_count: int
    skipped_duplicates: int
    row_count: int
    dataset_version: int
    trace_ids: List[str] = Field(default_factory=list)
