"""FastAPI app.

POST /intent {"text": "make it darker"} -> {"action": "dark", "via": "llm"|"rules"|"embeddings"}
POST /chat   {"question": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}
             -> text/event-stream of  data: {"t": "<text piece>"}
                and a final           data: {"done": true, "sources": [...]}
GET  /health -> status, model, chunk count
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app import config, intent, rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    rag.store()  # load the index + embedding model once, before the first visitor
    log.info("Index loaded (%d chunks). LLM: %s", rag.store().index.ntotal,
             config.LLM_MODEL if config.GROQ_API_KEY else "none (extractive fallback)")
    yield


app = FastAPI(title="Ask Harshil API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------- Request model ----------
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("question is empty")
        return v


# ---------- Rate limit (sliding windows per IP) ----------
_hits: dict[str, deque[float]] = defaultdict(deque)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


_intent_hits: dict[str, deque[float]] = defaultdict(deque)


def check_rate(ip: str, hits=_hits, per_minute=config.RATE_PER_MINUTE, per_day=config.RATE_PER_DAY) -> None:
    now = time.time()
    q = hits[ip]
    while q and now - q[0] > 86400:
        q.popleft()
    last_minute = sum(1 for t in q if now - t < 60)
    if last_minute >= per_minute or len(q) >= per_day:
        raise HTTPException(429, "You're asking a lot! Please wait a minute and try again.")
    q.append(now)


# ---------- Routes ----------
@app.get("/")
def root():
    return {"name": "Ask Harshil API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": config.LLM_MODEL if config.GROQ_API_KEY else None,
        "chunks": rag.store().index.ntotal,
    }


class IntentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)


@app.post("/intent")
async def intent_route(body: IntentRequest, request: Request):
    """Map a spoken sentence to a portfolio action (used by the mic)."""
    check_rate(client_ip(request), _intent_hits, config.INTENT_PER_MINUTE, config.INTENT_PER_MINUTE * 20)
    return await intent.classify(" ".join(body.text.split()))


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    check_rate(client_ip(request))
    docs = rag.retrieve(body.question)
    history = [t.model_dump() for t in body.history]

    async def events():
        async for piece in rag.stream_answer(body.question, docs, history):
            yield sse({"t": piece})
        yield sse({"done": True, "sources": rag.sources(docs)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
