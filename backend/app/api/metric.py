from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.metric_definition import MetricDefinition
from app.schemas.metric_definition import MetricCreate, MetricResponse

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("", response_model=List[MetricResponse])
def list_metrics(db: Session = Depends(get_db)):
    return db.query(MetricDefinition).all()


@router.post("", response_model=MetricResponse, status_code=201)
def create_metric(payload: MetricCreate, db: Session = Depends(get_db)):
    metric = MetricDefinition(
        name=payload.name,
        display_name=payload.display_name,
        metric_type=payload.metric_type,
        config=payload.config,
        category=payload.category,
        is_builtin=False,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


@router.delete("/{metric_id}", status_code=204)
def delete_metric(metric_id: int, db: Session = Depends(get_db)):
    metric = (
        db.query(MetricDefinition)
        .filter(MetricDefinition.id == metric_id)
        .first()
    )
    if not metric:
        raise HTTPException(status_code=404, detail="Metric not found")
    if metric.is_builtin:
        raise HTTPException(
            status_code=400, detail="Cannot delete a built-in metric"
        )
    db.delete(metric)
    db.commit()
    return None
