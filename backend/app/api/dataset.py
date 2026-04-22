import io
import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.dataset import Dataset, DatasetRow
from app.schemas.dataset import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    DatasetRowCreate,
    DatasetRowResponse,
    DatasetRowsResponse,
)


def _validate_row_data(data: Dict[str, Any], field_schema: list) -> List[str]:
    errors = []
    schema_fields = {f["name"]: f for f in field_schema}
    for f in field_schema:
        if f.get("required") and f["name"] not in data:
            errors.append(f"缺少必填字段: {f['name']}")
    for key in data:
        if key not in schema_fields:
            continue
        field_type = schema_fields[key].get("type", "text")
        value = data[key]
        if value is None:
            continue
        if field_type == "number" and not isinstance(value, (int, float)):
            try:
                float(value)
            except (ValueError, TypeError):
                errors.append(f"字段 {key} 应为数值类型")
        if field_type in ("text_list", "tags") and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    errors.append(f"字段 {key} 应为列表")
                else:
                    data[key] = parsed
            except json.JSONDecodeError:
                pass
        if field_type == "conversation" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    data[key] = parsed
            except json.JSONDecodeError:
                errors.append(f"字段 {key} 应为 JSON 对话数组")
        if field_type == "tool_call_list" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    data[key] = parsed
            except json.JSONDecodeError:
                errors.append(f"字段 {key} 应为 JSON 工具调用数组")
    return errors

router = APIRouter(prefix="/datasets", tags=["Datasets"])


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------

@router.get("", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).all()


@router.post("", response_model=DatasetResponse, status_code=201)
def create_dataset(payload: DatasetCreate, db: Session = Depends(get_db)):
    dataset = Dataset(
        name=payload.name,
        description=payload.description,
        sample_type=payload.sample_type,
        field_schema=[f.model_dump() for f in payload.field_schema],
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.put("/{dataset_id}", response_model=DatasetResponse)
def update_dataset(
    dataset_id: int, payload: DatasetUpdate, db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(dataset, key, value)

    db.commit()
    db.refresh(dataset)
    return dataset


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(dataset)
    db.commit()
    return None


# ---------------------------------------------------------------------------
# Dataset Rows
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/rows", response_model=DatasetRowsResponse)
def list_dataset_rows(
    dataset_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    total = db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset_id).count()
    offset = (page - 1) * page_size
    items = (
        db.query(DatasetRow)
        .filter(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.row_index)
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return DatasetRowsResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@router.post("/{dataset_id}/rows", response_model=DatasetRowResponse, status_code=201)
def add_dataset_row(
    dataset_id: int, payload: DatasetRowCreate, db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.field_schema:
        errors = _validate_row_data(payload.data, dataset.field_schema)
        if errors:
            raise HTTPException(status_code=422, detail="; ".join(errors))

    # Determine next row_index
    max_index = (
        db.query(DatasetRow.row_index)
        .filter(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.row_index.desc())
        .first()
    )
    next_index = (max_index[0] + 1) if max_index else 0

    row = DatasetRow(
        dataset_id=dataset_id,
        row_index=next_index,
        data=payload.data,
    )
    db.add(row)

    dataset.row_count = (
        db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset_id).count() + 1
    )

    db.commit()
    db.refresh(row)
    return row


@router.delete("/{dataset_id}/rows/{row_id}", status_code=204)
def delete_dataset_row(
    dataset_id: int, row_id: int, db: Session = Depends(get_db)
):
    row = (
        db.query(DatasetRow)
        .filter(DatasetRow.id == row_id, DatasetRow.dataset_id == dataset_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Dataset row not found")

    db.delete(row)

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset:
        dataset.row_count = (
            db.query(DatasetRow)
            .filter(DatasetRow.dataset_id == dataset_id)
            .count()
            - 1
        )

    db.commit()
    return None


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

@router.post("/{dataset_id}/import")
def import_dataset_rows(
    dataset_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    content = file.file.read()
    filename = file.filename or ""

    records: list[dict] = []

    if filename.lower().endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(io.BytesIO(content))
        records = df.where(df.notna(), None).to_dict(orient="records")
    elif filename.lower().endswith(".json"):
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise HTTPException(
                status_code=422,
                detail="JSON file must contain a list of objects",
            )
        records = parsed
    else:
        raise HTTPException(
            status_code=422,
            detail="Unsupported file format. Please upload a CSV or JSON file.",
        )

    # Determine starting row_index
    max_index = (
        db.query(DatasetRow.row_index)
        .filter(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.row_index.desc())
        .first()
    )
    start_index = (max_index[0] + 1) if max_index else 0

    for i, record in enumerate(records):
        row = DatasetRow(
            dataset_id=dataset_id,
            row_index=start_index + i,
            data=record,
        )
        db.add(row)

    dataset.row_count = (
        db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset_id).count()
        + len(records)
    )

    db.commit()

    return {"imported_count": len(records)}
