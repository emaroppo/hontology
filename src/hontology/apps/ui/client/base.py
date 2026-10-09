"""The client's transport: one HTTP request to the API, and its errors."""

from __future__ import annotations

from typing import Any

import httpx

from hontology.config import get_settings


class ApiError(RuntimeError):
    """A non-2xx response, carrying the API's own message where it gave one."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ApiBase:
    """The connection: where the API is, and one request to it. Each API area's
    calls are a mixin over this (see the package's ``Api``)."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or get_settings().api_base_url).rstrip("/")
        self.timeout = timeout

    def _request(
        self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any
    ) -> Any:
        try:
            response = httpx.request(
                method, f"{self.base_url}{path}", timeout=timeout or self.timeout, **kwargs
            )
        except httpx.HTTPError as exc:
            raise ApiError(0, f"cannot reach the API at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                detail = payload.get("detail", detail)
                if isinstance(detail, list):  # pydantic validation errors
                    detail = "; ".join(
                        f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg')}"
                        for e in detail
                    )
            except ValueError:
                pass
            raise ApiError(response.status_code, str(detail))

        if response.status_code == 204 or not response.content:
            return None
        if "json" not in response.headers.get("content-type", ""):
            return response.text
        return response.json()

    def healthy(self) -> bool:
        try:
            return self._request("GET", "/health").get("status") == "ok"
        except ApiError:
            return False
