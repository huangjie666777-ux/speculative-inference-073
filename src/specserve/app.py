"""FastAPI application: streaming speculative-decoding endpoint and UI."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from .models import engine
from .service import BusyError, GenerateRequest, Generation, GenerationService


ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="Speculative Inference")
service = GenerationService(engine)


@app.get("/health")
def health():
    return {"status": "ok", "busy": service.busy}


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/static/app.js")
def app_js():
    return FileResponse(ROOT / "web" / "app.js", media_type="application/javascript")


@app.get("/static/style.css")
def style_css():
    return FileResponse(ROOT / "web" / "style.css", media_type="text/css")


def _sse(event: str, data: dict) -> bytes:
    return (
        f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    ).encode("utf-8")


def _decode_piece(tokenizer, prefix_text: str, token_ids: list[int]) -> tuple[str, str]:
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    if full_text.startswith(prefix_text):
        return full_text[len(prefix_text):], full_text
    return full_text, full_text


@app.post("/api/generate")
async def generate(request: Request):
    try:
        payload = GenerateRequest.model_validate_json(await request.body())
    except ValidationError as exc:
        # Validation happens before acquiring the single running slot.
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    try:
        generation: Generation = service.start(payload)
    except BusyError:
        return JSONResponse(status_code=409, content={"detail": "server busy"})
    except (ValueError, OSError) as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def event_stream():
        token_ids: list[int] = []
        decoded_text = ""
        loop = asyncio.get_running_loop()
        finished_cleanly = False
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    token_id = await loop.run_in_executor(
                        None, lambda: next(generation.token_ids, None)
                    )
                except Exception as exc:
                    yield _sse("error", {"message": str(exc)})
                    return
                if token_id is None:
                    finished_cleanly = True
                    break
                token_ids.append(token_id)
                piece, decoded_text = _decode_piece(
                    generation.tokenizer, decoded_text, token_ids
                )
                yield _sse("token", {"text": piece, "token_id": token_id})
            if finished_cleanly:
                generation.stats.elapsed_seconds = time.perf_counter() - generation.start
                yield _sse("done", generation.stats.as_dict())
        finally:
            # Stop/disconnect path: signal cancellation, then release on a
            # worker thread. Closing the generator waits only for a forward
            # already in flight to complete; after that the per-request
            # DynamicCache frames are garbage collected. The request slot is
            # released as soon as that bounded cleanup finishes.
            service.stop(generation)

            def _finalize() -> None:
                try:
                    generation.token_ids.close()
                except Exception:
                    pass
                service.finish(generation)

            await loop.run_in_executor(None, _finalize)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
