import time
import uuid
from datetime import datetime, timezone

from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message

from app.core.logging import get_logger

log = get_logger("audit")


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Intercepta todas as requisições e loga:
    - timestamp, request_id, método, path
    - request_payload
    - response_payload, status_code
    - latency_ms
    """

    def __init__(self, app: ASGIApp) -> None:
        """Registra o middleware na aplicação ASGI."""
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Intercepta a requisição, loga payload/resposta/latência e repassa ao próximo handler."""
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Lê body e re-injeta no receive para os handlers conseguirem ler
        raw_body = await request.body()
        request_payload = raw_body.decode("utf-8") or None

        async def receive() -> Message:
            return {"type": "http.request", "body": raw_body, "more_body": False}

        request = Request(request.scope, receive)

        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        # Lê response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        response_payload = response_body.decode("utf-8")

        log.info(
            "http_request",
            request_id=request_id,
            timestamp=timestamp,
            method=request.method,
            path=request.url.path,
            request_payload=request_payload,
            response_payload=response_payload,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        # Reconstrói a response pois o body_iterator já foi consumido
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
