"""
main.py — FastAPI application for QueueStorm Investigator.

Exposes POST /analyze-ticket which takes a customer complaint plus recent
transaction history and returns a routed, classified, evidence-checked, and
safety-vetted analysis.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment from .env then .env.local (local overrides for dev secrets).
load_dotenv()
load_dotenv(".env.local", override=True)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

from app import logic, safety

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("queuestorm")

app = FastAPI(
    title="QueueStorm Investigator",
    description="Support-ticket analysis API for a digital-payments company.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    # All fields optional so a single malformed transaction entry degrades
    # gracefully instead of 400-ing the whole request (reliability).
    model_config = {"extra": "ignore"}

    transaction_id: Optional[str] = None
    timestamp: Optional[str] = None
    type: Optional[str] = None
    amount: Optional[float] = None
    counterparty: Optional[str] = None
    status: Optional[str] = None


class TicketRequest(BaseModel):
    model_config = {"extra": "ignore"}

    ticket_id: str = Field(..., min_length=1)
    complaint: str
    language: Optional[str] = None
    channel: Optional[str] = None
    user_type: Optional[str] = None
    campaign_context: Optional[str] = None
    transaction_history: List[Transaction] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("complaint")
    @classmethod
    def complaint_not_empty(cls, value: str) -> str:
        if value is None or not value.strip():
            raise ValueError("Complaint cannot be empty")
        return value


class TicketResponse(BaseModel):
    ticket_id: str
    relevant_transaction_id: Optional[str]
    evidence_verdict: str
    case_type: str
    severity: str
    department: str
    agent_summary: str
    recommended_next_action: str
    customer_reply: str
    human_review_required: bool
    confidence: Optional[float] = None
    reason_codes: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Error handlers — never leak stack traces, always clean JSON.
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Map validation errors to the spec'd 422 / 400 responses."""
    errors = exc.errors()
    for err in errors:
        msg = str(err.get("msg", ""))
        loc = err.get("loc", ())
        if "Complaint cannot be empty" in msg or ("complaint" in loc and err.get("type") == "missing"):
            return JSONResponse(status_code=422, content={"error": "Complaint cannot be empty"})
    # Missing/invalid structural fields → treat as malformed request.
    logger.info("Validation error: %s", errors)
    return JSONResponse(status_code=422, content={"error": "Invalid request format"})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so stack traces never reach the client."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "QueueStorm Investigator",
        "status": "ok",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "endpoints": ["POST /analyze-ticket", "GET /health"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    # The judge harness requires EXACTLY {"status":"ok"} — do not change this.
    return {"status": "ok"}


@app.post("/analyze-ticket", response_model=TicketResponse)
async def analyze_ticket(request: Request) -> JSONResponse:
    """
    Analyze a support ticket. We parse the body manually so we can return the
    exact spec'd error envelopes for malformed JSON vs. empty complaint.
    """
    started = time.perf_counter()

    # --- Parse raw JSON (malformed → 400) ---
    try:
        raw = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid request format"})

    if not isinstance(raw, dict):
        return JSONResponse(status_code=400, content={"error": "Invalid request format"})

    # --- Missing required fields → 400 (per Section 4.1) ---
    if "ticket_id" not in raw or not str(raw.get("ticket_id") or "").strip():
        return JSONResponse(status_code=400, content={"error": "Missing required field: ticket_id"})

    if "complaint" not in raw:
        return JSONResponse(status_code=400, content={"error": "Missing required field: complaint"})

    complaint = raw.get("complaint")
    if not isinstance(complaint, str):
        return JSONResponse(status_code=400, content={"error": "Invalid request format"})

    # --- Schema valid but semantically empty complaint → 422 ---
    if not complaint.strip():
        return JSONResponse(status_code=422, content={"error": "Complaint cannot be empty"})

    # --- Validate against the schema (structural issues → 400) ---
    try:
        ticket = TicketRequest(**raw)
    except Exception as exc:
        logger.info("Request validation failed: %s", exc)
        return JSONResponse(status_code=400, content={"error": "Invalid request format"})

    payload = ticket.model_dump()

    # --- Analyze (Groq with rule-based fallback) ---
    try:
        analysis = logic.analyze_ticket(payload)
    except Exception as exc:
        logger.exception("Analysis pipeline error, using rule-based fallback: %s", exc)
        analysis = logic.rule_based_analysis(payload)

    # --- Enforce hardcoded safety guardrails (overrides AI) ---
    try:
        safe_result = safety.enforce_safety(analysis, payload["complaint"], payload["ticket_id"])
    except Exception as exc:
        logger.exception("Safety enforcement error: %s", exc)
        # Minimal guaranteed-safe response.
        safe_result = {
            "ticket_id": payload["ticket_id"],
            "relevant_transaction_id": None,
            "evidence_verdict": "insufficient_data",
            "case_type": "other",
            "severity": "medium",
            "department": "customer_support",
            "agent_summary": "Automated analysis unavailable; manual review needed.",
            "recommended_next_action": "Manually review the complaint and transaction history.",
            "customer_reply": safety.SAFE_FALLBACK_REPLY.format(ticket_id=payload["ticket_id"]),
            "human_review_required": True,
            "confidence": 0.0,
            "reason_codes": ["safety_failsafe"],
        }

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "ticket=%s case=%s verdict=%s dept=%s severity=%s %sms",
        safe_result.get("ticket_id"), safe_result.get("case_type"),
        safe_result.get("evidence_verdict"), safe_result.get("department"),
        safe_result.get("severity"), elapsed_ms,
    )

    return JSONResponse(status_code=200, content=safe_result)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
