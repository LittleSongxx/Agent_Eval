from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict


class EndpointTargetBase(BaseModel):
    name: str
    description: Optional[str] = None
    endpoint_url: str
    transport_mode: Literal["json", "sse"] = "json"
    authorization: Optional[str] = None
    extra_headers: Optional[str] = "{}"
    request_body_template: Optional[str] = None
    response_mapping: Optional[Dict[str, Any]] = None
    default_test_input: Optional[str] = None


class EndpointTargetCreate(EndpointTargetBase):
    pass


class EndpointTargetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    endpoint_url: Optional[str] = None
    transport_mode: Optional[Literal["json", "sse"]] = None
    authorization: Optional[str] = None
    extra_headers: Optional[str] = None
    request_body_template: Optional[str] = None
    response_mapping: Optional[Dict[str, Any]] = None
    default_test_input: Optional[str] = None


class EndpointTargetResponse(EndpointTargetBase):
    id: int
    authorization_masked: str = ""
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        data = super().model_validate(obj, *args, **kwargs)
        auth = data.authorization or ""
        if auth:
            data.authorization_masked = auth[:8] + "****" + auth[-4:] if len(auth) > 16 else "****"
        data.authorization = auth
        return data


class EndpointTargetTestRequest(BaseModel):
    row_data: Dict[str, Any]


class EndpointTargetTestResponse(BaseModel):
    success: bool
    message: str
    request_body: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    latency_ms: Optional[int] = None
    raw_response: Optional[str] = None
    extracted_fields: Optional[Dict[str, Any]] = None
    mapping_errors: Optional[Dict[str, str]] = None
