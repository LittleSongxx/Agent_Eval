from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.endpoint_eval import extract_eval_fields, invoke_endpoint
from app.models.endpoint_target import EndpointTarget
from app.schemas.endpoint_target import (
    EndpointTargetCreate,
    EndpointTargetResponse,
    EndpointTargetTestRequest,
    EndpointTargetTestResponse,
    EndpointTargetUpdate,
)

router = APIRouter(prefix="/endpoint-targets", tags=["Endpoint Targets"])


def _target_config(target: EndpointTarget) -> dict:
    return {
        "endpoint_url": target.endpoint_url,
        "transport_mode": target.transport_mode or "json",
        "authorization": target.authorization or "",
        "extra_headers": target.extra_headers or "{}",
        "request_body_template": target.request_body_template or "",
    }


def _get_target_or_404(target_id: int, db: Session) -> EndpointTarget:
    target = db.query(EndpointTarget).filter(EndpointTarget.id == target_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Endpoint target not found")
    return target


@router.get("", response_model=List[EndpointTargetResponse])
def list_endpoint_targets(db: Session = Depends(get_db)):
    return db.query(EndpointTarget).order_by(EndpointTarget.created_at.desc()).all()


@router.post("", response_model=EndpointTargetResponse, status_code=201)
def create_endpoint_target(payload: EndpointTargetCreate, db: Session = Depends(get_db)):
    if not payload.endpoint_url.strip():
        raise HTTPException(status_code=400, detail="endpoint_url is required")
    target = EndpointTarget(**payload.model_dump())
    db.add(target)
    db.commit()
    db.refresh(target)
    return target


@router.put("/{target_id}", response_model=EndpointTargetResponse)
def update_endpoint_target(
    target_id: int,
    payload: EndpointTargetUpdate,
    db: Session = Depends(get_db),
):
    target = _get_target_or_404(target_id, db)
    data = payload.model_dump(exclude_unset=True)
    if "endpoint_url" in data and not str(data["endpoint_url"] or "").strip():
        raise HTTPException(status_code=400, detail="endpoint_url is required")
    if "authorization" in data and not str(data["authorization"] or "").strip():
        data.pop("authorization")
    for key, value in data.items():
        setattr(target, key, value)
    db.commit()
    db.refresh(target)
    return target


@router.delete("/{target_id}", status_code=204)
def delete_endpoint_target(target_id: int, db: Session = Depends(get_db)):
    target = _get_target_or_404(target_id, db)
    db.delete(target)
    db.commit()
    return None


@router.post("/{target_id}/test", response_model=EndpointTargetTestResponse)
async def test_endpoint_target(
    target_id: int,
    payload: EndpointTargetTestRequest,
    db: Session = Depends(get_db),
):
    target = _get_target_or_404(target_id, db)
    try:
        response_payload = await invoke_endpoint(payload.row_data or {}, _target_config(target))
        extracted_fields, mapping_errors = extract_eval_fields(
            response_payload,
            target.response_mapping or {},
        )
        return EndpointTargetTestResponse(
            success=True,
            message="接口调用成功",
            request_body=response_payload.get("request_body"),
            status_code=response_payload.get("status_code"),
            latency_ms=response_payload.get("latency_ms"),
            raw_response=response_payload.get("raw_response"),
            extracted_fields=extracted_fields,
            mapping_errors=mapping_errors,
        )
    except Exception as exc:
        return EndpointTargetTestResponse(success=False, message=str(exc))
