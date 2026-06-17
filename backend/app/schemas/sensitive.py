from __future__ import annotations

from copy import deepcopy
from typing import Any


SENSITIVE_FIELD_NAMES = {
    "authorization",
    "api_key",
    "apikey",
    "api-key",
    "access_token",
    "refresh_token",
    "secret",
    "password",
}


def mask_secret(value: Any) -> str:
    secret = str(value or "")
    if not secret:
        return ""
    return f"{secret[:8]}****{secret[-4:]}" if len(secret) > 16 else "****"


def redact_sensitive_mapping(value: Any) -> Any:
    """Return a copy with known secret fields hidden for API responses."""

    if isinstance(value, list):
        return [redact_sensitive_mapping(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)

    redacted: dict[str, Any] = {}
    for key, item in value.items():
        normalized = key.lower().replace("-", "_")
        if normalized in SENSITIVE_FIELD_NAMES:
            masked = mask_secret(item)
            redacted[key] = None
            if masked:
                redacted[f"{key}_masked"] = masked
            continue
        redacted[key] = redact_sensitive_mapping(item)
    return redacted
