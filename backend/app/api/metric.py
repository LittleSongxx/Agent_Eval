from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.metric_definition import MetricDefinition
from app.schemas.metric_definition import MetricCreate, MetricResponse, MetricUpdate

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("", response_model=List[MetricResponse])
def list_metrics(db: Session = Depends(get_db)):
    return db.query(MetricDefinition).all()


@router.post("", response_model=MetricResponse, status_code=201)
def create_metric(payload: MetricCreate, db: Session = Depends(get_db)):
    existing = (
        db.query(MetricDefinition)
        .filter(MetricDefinition.name == payload.name)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Metric name already exists")
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


@router.put("/{metric_id}", response_model=MetricResponse)
def update_metric(metric_id: int, payload: MetricUpdate, db: Session = Depends(get_db)):
    metric = (
        db.query(MetricDefinition)
        .filter(MetricDefinition.id == metric_id)
        .first()
    )
    if not metric:
        raise HTTPException(status_code=404, detail="Metric not found")
    if metric.is_builtin:
        raise HTTPException(
            status_code=400, detail="Cannot update a built-in metric"
        )

    existing = (
        db.query(MetricDefinition)
        .filter(
            MetricDefinition.name == payload.name,
            MetricDefinition.id != metric_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Metric name already exists")

    metric.name = payload.name
    metric.display_name = payload.display_name
    metric.metric_type = payload.metric_type
    metric.config = payload.config
    metric.category = payload.category
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
