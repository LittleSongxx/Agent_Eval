import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.llm_config import LLMConfig
from app.schemas.llm_config import (
    LLMConfigCreate,
    LLMConfigUpdate,
    LLMConfigResponse,
    LLMTestResult,
)

router = APIRouter(prefix="/llm-configs", tags=["LLM Configs"])


def _run_llm_smoke_test(config_like) -> LLMTestResult:
    try:
        import openai

        client = openai.OpenAI(
            base_url=config_like.api_base_url,
            api_key=config_like.api_key,
        )

        start = time.time()
        response = client.chat.completions.create(
            model=config_like.model_name,
            messages=[{"role": "user", "content": "请用一句话介绍你自己。"}],
            max_tokens=min(int(getattr(config_like, "max_tokens", 128) or 128), 128),
            temperature=getattr(config_like, "temperature", 0.01) or 0.01,
        )
        latency_ms = (time.time() - start) * 1000
        sample_output = response.choices[0].message.content or ""

        return LLMTestResult(
            success=True,
            message="Connection successful",
            latency_ms=round(latency_ms, 2),
            sample_output=sample_output[:500],
        )
    except Exception as exc:
        return LLMTestResult(
            success=False,
            message=f"Connection failed: {str(exc)}",
            latency_ms=None,
            sample_output=None,
        )


@router.get("")
def list_llm_configs(db: Session = Depends(get_db)):
    configs = db.query(LLMConfig).all()
    return [LLMConfigResponse.from_orm_with_mask(c) for c in configs]


@router.post("", status_code=201)
def create_llm_config(payload: LLMConfigCreate, db: Session = Depends(get_db)):
    config = LLMConfig(**payload.model_dump())
    db.add(config)
    db.commit()
    db.refresh(config)
    return LLMConfigResponse.from_orm_with_mask(config)


@router.get("/{config_id}")
def get_llm_config(config_id: int, db: Session = Depends(get_db)):
    config = db.query(LLMConfig).filter(LLMConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="LLM config not found")
    return LLMConfigResponse.from_orm_with_mask(config)


@router.put("/{config_id}")
def update_llm_config(
    config_id: int, payload: LLMConfigUpdate, db: Session = Depends(get_db)
):
    config = db.query(LLMConfig).filter(LLMConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="LLM config not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(config, key, value)

    db.commit()
    db.refresh(config)
    return LLMConfigResponse.from_orm_with_mask(config)


@router.delete("/{config_id}", status_code=204)
def delete_llm_config(config_id: int, db: Session = Depends(get_db)):
    config = db.query(LLMConfig).filter(LLMConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="LLM config not found")
    db.delete(config)
    db.commit()
    return None


@router.post("/{config_id}/test", response_model=LLMTestResult)
def test_llm_config(config_id: int, db: Session = Depends(get_db)):
    config = db.query(LLMConfig).filter(LLMConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="LLM config not found")

    return _run_llm_smoke_test(config)


@router.post("/test-draft", response_model=LLMTestResult)
def test_llm_config_draft(payload: LLMConfigCreate):
    return _run_llm_smoke_test(payload)
