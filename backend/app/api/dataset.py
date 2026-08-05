import io
import json
from typing import Any, Dict, List
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
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
                    data[key] = [value]
                else:
                    data[key] = parsed
            except json.JSONDecodeError:
                # 测试同学在页面或 CSV 中经常用普通文本/换行文本填写评估标准。
                # 对 text_list/tags 做宽松兼容：JSON 数组优先，其次按换行拆分，单行文本作为单元素列表。
                data[key] = [item.strip() for item in value.splitlines() if item.strip()] or [value]
        if field_type == "conversation" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    data[key] = parsed
                else:
                    errors.append(f"字段 {key} 应为 JSON 对话数组")
            except json.JSONDecodeError:
                errors.append(f"字段 {key} 应为 JSON 对话数组")
        if field_type == "tool_call_list" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    data[key] = parsed
                else:
                    errors.append(f"字段 {key} 应为 JSON 工具调用数组")
            except json.JSONDecodeError:
                errors.append(f"字段 {key} 应为 JSON 工具调用数组")
        if field_type == "json" and isinstance(value, str):
            try:
                data[key] = json.loads(value)
            except json.JSONDecodeError:
                errors.append(f"字段 {key} 应为 JSON 对象或数组")
    return errors


def _bump_dataset_version(dataset: Dataset) -> None:
    """数据集内容变更（增删改/导入）后版本自增，供评测任务追溯口径。"""
    dataset.version = (dataset.version or 1) + 1


def _normalize_record(record: Dict[str, Any], field_schema: list) -> Dict[str, Any]:
    normalized = dict(record or {})
    if field_schema:
        errors = _validate_row_data(normalized, field_schema)
        if errors:
            raise HTTPException(status_code=422, detail="; ".join(errors))
    return normalized


def _canonicalize_record(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_import_records(filename: str, content: bytes) -> list[dict[str, Any]]:
    filename_lower = filename.lower()
    if filename_lower.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(io.BytesIO(content))
        return df.where(df.notna(), None).to_dict(orient="records")
    if filename_lower.endswith(".json"):
        parsed = json.loads(content)
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise HTTPException(
                status_code=422,
                detail="JSON file must contain a list of objects",
            )
        return parsed
    raise HTTPException(
        status_code=422,
        detail="Unsupported file format. Please upload a CSV or JSON file.",
    )


def _collect_export_columns(dataset: Dataset, rows: list[DatasetRow]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for field in dataset.field_schema or []:
        name = field.get("name")
        if name and name not in seen:
            seen.add(name)
            columns.append(name)
    for row in rows:
        for key in (row.data or {}).keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def _serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _build_content_disposition(dataset_name: str, suffix: str) -> str:
    safe_name = (dataset_name or "dataset").strip()
    ascii_fallback = "".join(ch if ord(ch) < 128 and ch not in {'"', "\\"} else "_" for ch in safe_name).strip(" ._")
    ascii_fallback = ascii_fallback or "dataset"
    utf8_name = quote(f"{safe_name}.{suffix}")
    return f"attachment; filename=\"{ascii_fallback}.{suffix}\"; filename*=UTF-8''{utf8_name}"

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
    _bump_dataset_version(dataset)

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
        _bump_dataset_version(dataset)

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
    records = _load_import_records(filename, content)

    existing_keys = {
        _canonicalize_record(row.data or {})
        for row in db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset_id).all()
    }
    seen_import_keys: set[str] = set()
    normalized_records: list[dict[str, Any]] = []
    skipped_duplicates = 0
    for record in records:
        normalized = _normalize_record(record, dataset.field_schema or [])
        canonical = _canonicalize_record(normalized)
        if canonical in existing_keys or canonical in seen_import_keys:
            skipped_duplicates += 1
            continue
        seen_import_keys.add(canonical)
        normalized_records.append(normalized)

    # Determine starting row_index
    max_index = (
        db.query(DatasetRow.row_index)
        .filter(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.row_index.desc())
        .first()
    )
    start_index = (max_index[0] + 1) if max_index else 0

    for i, record in enumerate(normalized_records):
        row = DatasetRow(
            dataset_id=dataset_id,
            row_index=start_index + i,
            data=record,
        )
        db.add(row)

    dataset.row_count = (
        db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset_id).count()
        + len(normalized_records)
    )
    if normalized_records:
        _bump_dataset_version(dataset)

    db.commit()

    return {
        "imported_count": len(normalized_records),
        "skipped_duplicates": skipped_duplicates,
    }


@router.get("/{dataset_id}/export")
def export_dataset_rows(
    dataset_id: int,
    format: str = Query("csv"),
    db: Session = Depends(get_db),
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    rows = (
        db.query(DatasetRow)
        .filter(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.row_index)
        .all()
    )
    records = [dict(row.data or {}) for row in rows]
    export_format = (format or "csv").lower()
    dataset_name = (dataset.name or f"dataset-{dataset_id}").strip().replace("/", "-")

    if export_format == "json":
        payload = json.dumps(records, ensure_ascii=False, indent=2)
        return StreamingResponse(
            iter([payload.encode("utf-8")]),
            media_type="application/json; charset=utf-8",
            headers={
                "Content-Disposition": _build_content_disposition(dataset_name, "json")
            },
        )

    if export_format == "csv":
        import csv

        buffer = io.StringIO()
        columns = _collect_export_columns(dataset, rows)
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _serialize_csv_value(record.get(key)) for key in columns})
        return StreamingResponse(
            iter([buffer.getvalue().encode("utf-8-sig")]),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": _build_content_disposition(dataset_name, "csv")
            },
        )

    raise HTTPException(
        status_code=422,
        detail="Unsupported export format. Please choose csv or json.",
    )
