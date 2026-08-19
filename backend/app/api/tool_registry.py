from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.trace_lint import lint_agent_trace
from app.models.tool_registry import ToolDefinition
from app.schemas.tool_registry import (
    ToolDefinitionCreate,
    ToolDefinitionResponse,
    ToolDefinitionUpdate,
    TraceLintRequest,
    TraceLintResponse,
)

router = APIRouter(prefix="/tool-registry", tags=["Tool Registry"])


def _get_tool(tool_id: int, db: Session) -> ToolDefinition:
    tool = db.query(ToolDefinition).filter(ToolDefinition.id == tool_id).first()
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@router.get("", response_model=List[ToolDefinitionResponse])
def list_tools(db: Session = Depends(get_db)):
    return db.query(ToolDefinition).order_by(ToolDefinition.name).all()


@router.post("", response_model=ToolDefinitionResponse, status_code=201)
def create_tool(payload: ToolDefinitionCreate, db: Session = Depends(get_db)):
    if db.query(ToolDefinition).filter(ToolDefinition.name == payload.name).first():
        raise HTTPException(status_code=400, detail="Tool name already exists")
    tool = ToolDefinition(**payload.model_dump())
    db.add(tool)
    db.commit()
    db.refresh(tool)
    return tool


@router.put("/{tool_id}", response_model=ToolDefinitionResponse)
def update_tool(tool_id: int, payload: ToolDefinitionUpdate, db: Session = Depends(get_db)):
    tool = _get_tool(tool_id, db)
    values = payload.model_dump(exclude_unset=True)
    if values.get("name") and db.query(ToolDefinition).filter(
        ToolDefinition.name == values["name"], ToolDefinition.id != tool_id
    ).first():
        raise HTTPException(status_code=400, detail="Tool name already exists")
    for key, value in values.items():
        setattr(tool, key, value)
    db.commit()
    db.refresh(tool)
    return tool


@router.delete("/{tool_id}", status_code=204)
def delete_tool(tool_id: int, db: Session = Depends(get_db)):
    tool = _get_tool(tool_id, db)
    db.delete(tool)
    db.commit()
    return None


@router.post("/lint", response_model=TraceLintResponse)
def lint_trace(payload: TraceLintRequest, db: Session = Depends(get_db)):
    registry = payload.tool_registry
    if registry is None:
        registry = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_schema": tool.parameters_schema,
                "risk_level": tool.risk_level,
                "has_side_effect": tool.has_side_effect,
                "idempotency_required": tool.idempotency_required,
                "timeout_ms": tool.timeout_ms,
                "enabled": tool.enabled,
            }
            for tool in db.query(ToolDefinition).filter(ToolDefinition.enabled.is_(True)).all()
        ]
    result = lint_agent_trace(
        payload.trajectory,
        payload.available_tools,
        registry,
        payload.max_steps,
    )
    return result
