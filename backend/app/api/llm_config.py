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


@router.get("", response_model=List[LLMConfigResponse])
def list_llm_configs(db: Session = Depends(get_db)):
    return db.query(LLMConfig).all()


@router.post("", response_model=LLMConfigResponse, status_code=201)
def create_llm_config(payload: LLMConfigCreate, db: Session = Depends(get_db)):
    config = LLMConfig(**payload.model_dump())
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


@router.get("/{config_id}", response_model=LLMConfigResponse)
def get_llm_config(config_id: int, db: Session = Depends(get_db)):
    config = db.query(LLMConfig).filter(LLMConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="LLM config not found")
    return config


@router.put("/{config_id}", response_model=LLMConfigResponse)
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
    return config


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

    try:
        import openai

        client = openai.OpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key,
        )

        start = time.time()
        client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=16,
        )
        latency_ms = (time.time() - start) * 1000

        return LLMTestResult(
            success=True,
            message="Connection successful",
            latency_ms=round(latency_ms, 2),
        )
    except Exception as exc:
        return LLMTestResult(
            success=False,
            message=f"Connection failed: {str(exc)}",
            latency_ms=None,
        )
