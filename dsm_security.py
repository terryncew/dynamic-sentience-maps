"""Stdlib-only network boundary used by the Dynamic Sentience Maps server."""

from __future__ import annotations

import hmac
import ipaddress
from typing import Optional


DEFAULT_ORIGINS = ("http://127.0.0.1:8000", "http://localhost:8000")


def allowed_origins(configured: Optional[str]) -> list[str]:
    """Return explicit browser origins; wildcard credential sharing is forbidden."""
    origins = [
        value.strip()
        for value in (configured or "").split(",")
        if value.strip()
    ] if configured else list(DEFAULT_ORIGINS)
    if "*" in origins:
        raise ValueError("DSM_ALLOWED_ORIGINS may not contain '*' when credentials are enabled")
    return origins


def is_loopback(client_host: Optional[str]) -> bool:
    if not client_host:
        return False
    try:
        return ipaddress.ip_address(client_host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def mutation_authorization(
    client_host: Optional[str],
    authorization: Optional[str],
    configured_token: Optional[str],
) -> tuple[bool, int, str]:
    """Authorize a mutation using the transport peer and an opt-in bearer token."""
    if is_loopback(client_host):
        return True, 200, "loopback_peer"
    expected = (configured_token or "").strip()
    if not expected:
        return False, 403, "remote_mutation_disabled"
    scheme, separator, presented = (authorization or "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not presented:
        return False, 401, "bearer_token_required"
    if not hmac.compare_digest(presented, expected):
        return False, 401, "bearer_token_invalid"
    return True, 200, "bearer_token_valid"
