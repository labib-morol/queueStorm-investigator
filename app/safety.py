"""
safety.py — Hardcoded safety guardrails for QueueStorm Investigator.

These rules OVERRIDE any AI output. Every response is passed through
`enforce_safety()` before it leaves the API. The guarantees:

  1. The customer reply NEVER asks for PIN / OTP / password / credentials.
  2. The customer reply NEVER promises a refund, reversal, or unblock.
  3. Prompt-injection / social-engineering attempts are forced into the
     phishing path with mandatory human review.
  4. Replies always point the customer to official channels only.

Nothing here calls an LLM. It is deterministic, fast, and the last line
of defence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Banned content
# ---------------------------------------------------------------------------

# Terms that must NEVER appear in a customer reply because they solicit secret
# credentials. Matched with WORD BOUNDARIES (so "pin" does not match "shopping").
# Includes Bangla-script forms for multilingual robustness.
CREDENTIAL_PHRASES: Tuple[str, ...] = (
    "pin",
    "otp",
    "password",
    "passcode",
    "share your",
    "provide your",
    "enter your",
    "confirm your credentials",
    "one time password",
    "one-time password",
    "security code",
    "verification code",
    "cvv",
    "card number",
    "পিন",
    "ওটিপি",
    "পাসওয়ার্ড",
)

# Phrases that must NEVER appear because they promise an outcome the company
# cannot guarantee before investigation. English + Banglish + Bangla.
PROMISE_PHRASES: Tuple[str, ...] = (
    "we will refund",
    "you will receive a refund",
    "we will reverse",
    "we will reverse the transaction",
    "account will be unblocked",
    "we will unblock",
    "we guarantee",
    "your money will be returned",
    "guaranteed refund",
    "money back guaranteed",
    "will be refunded",
    "we'll refund",
    "we will return your money",
    "definitely get your money back",
    # Banglish / Bangla promise phrasings
    "taka ferot dibo",
    "taka ferot debo",
    "ferot dibo",
    "ferot debo",
    "refund dibo",
    "refund debo",
    "reverse kore dibo",
    "taka ferot paben",
    "টাকা ফেরত দেব",
    "টাকা ফেরত দিব",
    "ফেরত দিয়ে দেব",
    "রিফান্ড দেব",
)

# Signals of prompt injection / social engineering inside the *complaint*.
INJECTION_PATTERNS: Tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous",
    "ignore the above",
    "disregard previous",
    "disregard all instructions",
    "disregard your",
    "you are now",
    "you're now",
    "pretend you are",
    "pretend to be",
    "act as a",
    "act as an",
    "act like",
    "jailbreak",
    "system prompt",
    "developer mode",
    "forget your instructions",
    "forget all previous",
    "new instructions:",
    "override your",
    "reveal your prompt",
    "print your prompt",
    "you must now",
)

# A refund that is being contested / disputed routes to dispute_resolution
# rather than plain customer_support (per taxonomy 7.2).
CONTESTED_REFUND_SIGNALS: Tuple[str, ...] = (
    "dispute", "denied", "rejected", "refused", "you said no", "not eligible",
    "unfair", "demand", "escalat", "complain again", "second time", "still not",
    "promised", "already asked", "many times", "barbar", "বারবার", "again and again",
)

# Words that suggest a customer is being phished / scammed (used to bump
# severity and force fraud routing).
PHISHING_SIGNALS: Tuple[str, ...] = (
    "otp",
    "shared my pin",
    "shared my otp",
    "someone called",
    "asked for my",
    "scam",
    "fraud",
    "hacked",
    "unauthorized",
    "unauthorised",
    "impersonat",
    "phishing",
    "gave my code",
    "told me to send",
    "fake agent",
    "fake call",
)

# A guaranteed-safe reply used when the model output cannot be repaired.
SAFE_FALLBACK_REPLY: str = (
    "Thank you for reaching out. We have noted your concern and our team will "
    "investigate this matter. Any eligible amount will be processed through "
    "official channels. For your security, never disclose your confidential "
    "credentials to anyone. We will only contact you through our official "
    "support. Reference: {ticket_id}."
)

# A phishing-specific reply.
PHISHING_SAFE_REPLY: str = (
    "Thank you for contacting us. We have noted your concern and flagged it for "
    "urgent review by our fraud and risk team. Please do not disclose any "
    "confidential verification details to anyone — our staff will never ask for "
    "them. We will only contact you through our official support channels. "
    "Reference: {ticket_id}."
)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase + collapse whitespace for robust phrase matching."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def detect_injection(complaint: str) -> bool:
    """True if the complaint tries to hijack the model's instructions."""
    text = _norm(complaint)
    return any(pat in text for pat in INJECTION_PATTERNS)


def detect_phishing(complaint: str) -> bool:
    """True if the complaint describes a likely phishing / fraud scenario."""
    text = _norm(complaint)
    return any(sig in text for sig in PHISHING_SIGNALS)


def _contains_term(text: str, term: str) -> bool:
    """
    True if `term` appears in `text` as a whole token, not as a fragment of a
    larger word. This prevents "pin" from matching inside "shopping" while still
    catching it in "your pin" or "pin:".
    """
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text) is not None


def find_credential_violations(reply: str) -> List[str]:
    """Return the list of credential-soliciting terms found in a reply."""
    text = _norm(reply)
    return [p for p in CREDENTIAL_PHRASES if _contains_term(text, p)]


def find_promise_violations(reply: str) -> List[str]:
    """Return the list of forbidden promise phrases found in a reply."""
    text = _norm(reply)
    return [p for p in PROMISE_PHRASES if _contains_term(text, p)]


def is_reply_safe(reply: str) -> bool:
    """True if a reply violates none of the hardcoded content rules."""
    if not reply or not reply.strip():
        return False
    return not find_credential_violations(reply) and not find_promise_violations(reply)


# ---------------------------------------------------------------------------
# Reply sanitisation
# ---------------------------------------------------------------------------

def sanitize_reply(reply: str, ticket_id: str, *, phishing: bool = False) -> Tuple[str, List[str]]:
    """
    Return a guaranteed-safe customer reply plus a list of reason codes for
    any corrections that were applied.

    If the model's reply contains a forbidden phrase we do NOT try to surgically
    edit it (that risks leaving fragments) — we replace it wholesale with a
    vetted safe template. This is the conservative, audit-friendly choice.
    """
    reasons: List[str] = []
    safe = (reply or "").strip()

    if phishing:
        # Phishing always gets the dedicated reply, regardless of model output.
        return PHISHING_SAFE_REPLY.format(ticket_id=ticket_id), reasons

    if not safe:
        reasons.append("empty_reply_replaced")
        return SAFE_FALLBACK_REPLY.format(ticket_id=ticket_id), reasons

    cred_hits = find_credential_violations(safe)
    promise_hits = find_promise_violations(safe)

    if cred_hits:
        reasons.append("credential_solicitation_blocked")
    if promise_hits:
        reasons.append("refund_promise_blocked")

    if cred_hits or promise_hits:
        return SAFE_FALLBACK_REPLY.format(ticket_id=ticket_id), reasons

    return safe, reasons


# ---------------------------------------------------------------------------
# Full-response enforcement
# ---------------------------------------------------------------------------

# Department routing is part of the safety contract: case_type fully determines
# the department, so we re-derive it here and never trust the model.
DEPARTMENT_BY_CASE_TYPE: Dict[str, str] = {
    "wrong_transfer": "dispute_resolution",
    "payment_failed": "payments_ops",
    "duplicate_payment": "payments_ops",
    "merchant_settlement_delay": "merchant_operations",
    "agent_cash_in_issue": "agent_operations",
    "phishing_or_social_engineering": "fraud_risk",
    "refund_request": "customer_support",
    "other": "customer_support",
}

VALID_VERDICTS = {"consistent", "inconsistent", "insufficient_data"}
VALID_CASE_TYPES = set(DEPARTMENT_BY_CASE_TYPE.keys())
VALID_SEVERITIES = {"low", "medium", "high", "critical"}
VALID_DEPARTMENTS = set(DEPARTMENT_BY_CASE_TYPE.values())


def route_department(case_type: str, severity: str, *, contested: bool = False) -> str:
    """Deterministic department routing from case_type / severity."""
    # A contested refund is a dispute, not a routine support request.
    if case_type == "refund_request" and contested:
        return "dispute_resolution"
    # Low-severity refund or vague 'other' → customer support.
    if case_type in ("refund_request", "other") and severity == "low":
        return "customer_support"
    return DEPARTMENT_BY_CASE_TYPE.get(case_type, "customer_support")


def enforce_safety(result: Dict[str, Any], complaint: str, ticket_id: str) -> Dict[str, Any]:
    """
    Validate and repair a (possibly model-generated) analysis dict so the
    returned object is always schema-correct and policy-safe.

    This is idempotent and never raises on bad input — it coerces instead.
    """
    out: Dict[str, Any] = dict(result or {})
    reason_codes: List[str] = list(out.get("reason_codes") or [])

    # --- Hard override: prompt injection / social engineering ---
    injection = detect_injection(complaint)
    phishing_text = detect_phishing(complaint)

    if injection:
        out["case_type"] = "phishing_or_social_engineering"
        out["human_review_required"] = True
        reason_codes.append("prompt_injection_detected")

    # --- Normalise enum fields, coercing anything invalid ---
    out["ticket_id"] = ticket_id

    verdict = out.get("evidence_verdict")
    if verdict not in VALID_VERDICTS:
        out["evidence_verdict"] = "insufficient_data"
        reason_codes.append("verdict_defaulted")

    case_type = out.get("case_type")
    if case_type not in VALID_CASE_TYPES:
        out["case_type"] = "other"
        reason_codes.append("case_type_defaulted")
    case_type = out["case_type"]

    # If the complaint clearly looks like phishing but the model didn't flag it,
    # and there is no stronger case type, force fraud handling.
    if phishing_text and case_type == "other":
        out["case_type"] = "phishing_or_social_engineering"
        case_type = out["case_type"]
        out["human_review_required"] = True
        reason_codes.append("phishing_signal_detected")

    severity = out.get("severity")
    if severity not in VALID_SEVERITIES:
        out["severity"] = "medium"
        reason_codes.append("severity_defaulted")
    severity = out["severity"]

    # Phishing is always at least high severity.
    if case_type == "phishing_or_social_engineering" and severity in ("low", "medium"):
        out["severity"] = "critical"
        severity = "critical"

    # --- Department is ALWAYS re-derived (never trust the model) ---
    complaint_norm = _norm(complaint)
    contested = case_type == "refund_request" and any(
        sig in complaint_norm for sig in CONTESTED_REFUND_SIGNALS
    )
    out["department"] = route_department(case_type, severity, contested=contested)
    if contested:
        reason_codes.append("contested_refund")

    # --- relevant_transaction_id must be str or null ---
    rtid = out.get("relevant_transaction_id")
    if rtid in ("", "null", "none", "None"):
        rtid = None
    out["relevant_transaction_id"] = rtid if (rtid is None or isinstance(rtid, str)) else str(rtid)

    # If we have no matching transaction, verdict cannot be consistent/inconsistent.
    if out["relevant_transaction_id"] is None and out["evidence_verdict"] != "insufficient_data":
        out["evidence_verdict"] = "insufficient_data"
        reason_codes.append("no_transaction_match")

    # --- Text fields ---
    out["agent_summary"] = (out.get("agent_summary") or "").strip() or (
        "Customer complaint received; see details and recommended action."
    )
    out["recommended_next_action"] = (out.get("recommended_next_action") or "").strip() or (
        "Review the complaint and transaction history, then follow standard procedure."
    )

    # --- Customer reply: the most important guardrail ---
    is_phish = case_type == "phishing_or_social_engineering" or injection
    safe_reply, reply_reasons = sanitize_reply(
        out.get("customer_reply", ""), ticket_id, phishing=is_phish
    )
    out["customer_reply"] = safe_reply
    reason_codes.extend(reply_reasons)

    # --- human_review_required ---
    hrr = bool(out.get("human_review_required", False))
    if severity in ("high", "critical"):
        hrr = True
    if case_type in ("wrong_transfer", "phishing_or_social_engineering"):
        hrr = True
    if out["evidence_verdict"] == "inconsistent":
        hrr = True
    out["human_review_required"] = hrr

    # --- confidence ---
    conf = out.get("confidence")
    try:
        conf = float(conf)
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 0.5
    out["confidence"] = round(conf, 2)

    # --- reason_codes: dedupe, keep order ---
    seen = set()
    deduped: List[str] = []
    for code in reason_codes:
        if code and code not in seen:
            seen.add(code)
            deduped.append(code)
    out["reason_codes"] = deduped

    # Final paranoia check: if somehow the reply is still unsafe, replace it.
    if not is_reply_safe(out["customer_reply"]):
        out["customer_reply"] = SAFE_FALLBACK_REPLY.format(ticket_id=ticket_id)
        if "reply_failsafe_applied" not in out["reason_codes"]:
            out["reason_codes"].append("reply_failsafe_applied")

    return out
