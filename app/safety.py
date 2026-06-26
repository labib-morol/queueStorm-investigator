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

# Bangla fallbacks (used when the complaint is in Bangla and a reply must be replaced).
SAFE_FALLBACK_REPLY_BN: str = (
    "আপনার অভিযোগটি আমরা পেয়েছি এবং আমাদের টিম বিষয়টি যাচাই করে দেখবে। যেকোনো প্রযোজ্য "
    "পরিমাণ অফিসিয়াল চ্যানেলের মাধ্যমে প্রক্রিয়া করা হবে। অনুগ্রহ করে কারো সাথে আপনার পিন "
    "বা ওটিপি শেয়ার করবেন না। রেফারেন্স: {ticket_id}।"
)
PHISHING_SAFE_REPLY_BN: str = (
    "আপনার অভিযোগটি আমরা জরুরি ভিত্তিতে আমাদের ফ্রড ও রিস্ক টিমের কাছে পাঠিয়েছি। অনুগ্রহ "
    "করে কারো সাথে আপনার পিন, ওটিপি বা পাসওয়ার্ড শেয়ার করবেন না — আমাদের কর্মীরা কখনো "
    "এগুলো চাইবে না। রেফারেন্স: {ticket_id}।"
)


def _is_bangla(text: str) -> bool:
    return any("ঀ" <= ch <= "৿" for ch in (text or ""))


# ---------------------------------------------------------------------------
# Department-aware customer reply (deterministic, team-attributed, safe)
# ---------------------------------------------------------------------------
# Actions are attributed to the named handling TEAM (not first-person "we will
# do X"), positioning the service as a copilot that routes to a team — matching
# the official sample replies ("Our dispute team will review the case").

TEAM_EN: Dict[str, str] = {
    "customer_support": "support team",
    "dispute_resolution": "dispute resolution team",
    "payments_ops": "payments operations team",
    "merchant_operations": "merchant operations team",
    "agent_operations": "agent operations team",
    "fraud_risk": "fraud and risk team",
}
TEAM_BN: Dict[str, str] = {
    "customer_support": "সাপোর্ট টিম",
    "dispute_resolution": "ডিসপিউট রেজোলিউশন টিম",
    "payments_ops": "পেমেন্টস অপারেশন্স টিম",
    "merchant_operations": "মার্চেন্ট অপারেশন্স টিম",
    "agent_operations": "এজেন্ট অপারেশন্স টিম",
    "fraud_risk": "ফ্রড ও রিস্ক টিম",
}


def build_customer_reply(case_type: str, department: str, rtid: Optional[str],
                         ticket_id: str, bangla: bool) -> str:
    """Generate a safe, professional, team-attributed customer reply."""
    if bangla:
        team = TEAM_BN.get(department, "সাপোর্ট টিম")
        about = f" (লেনদেন {rtid})" if rtid else ""
        if case_type == "phishing_or_social_engineering":
            return ("কোনো তথ্য শেয়ার করার আগে যোগাযোগ করার জন্য ধন্যবাদ। আমাদের ফ্রড ও রিস্ক টিম "
                    "বিষয়টি পর্যালোচনা করছে। অনুগ্রহ করে আপনার পিন, ওটিপি বা পাসওয়ার্ড কারো সাথে শেয়ার "
                    "করবেন না — আমাদের কোনো কর্মী কখনো এগুলো চাইবে না। শুধুমাত্র অফিসিয়াল সাপোর্ট চ্যানেলের "
                    f"মাধ্যমে যোগাযোগ করা হবে। রেফারেন্স: {ticket_id}।")
        if case_type == "other" or not rtid:
            return ("যোগাযোগ করার জন্য ধন্যবাদ। সঠিক লেনদেনটি শনাক্ত করতে অনুগ্রহ করে লেনদেন আইডি, পরিমাণ "
                    f"এবং আনুমানিক সময় জানান। এরপর আমাদের {team} বিষয়টি অফিসিয়াল চ্যানেলের মাধ্যমে যাচাই করবে। "
                    f"নিরাপত্তার জন্য কারো সাথে আপনার পিন বা ওটিপি শেয়ার করবেন না। রেফারেন্স: {ticket_id}।")
        if case_type == "refund_request":
            return (f"যোগাযোগ করার জন্য ধন্যবাদ। সম্পন্ন পেমেন্টের রিফান্ড প্রযোজ্য নীতিমালার উপর নির্ভর করে। "
                    f"আমাদের {team} আপনার অনুরোধটি যাচাই করবে এবং যেকোনো প্রযোজ্য পরিমাণ অফিসিয়াল চ্যানেলের "
                    f"মাধ্যমে প্রক্রিয়া করা হবে। অনুগ্রহ করে কারো সাথে আপনার পিন বা ওটিপি শেয়ার করবেন না। "
                    f"রেফারেন্স: {ticket_id}।")
        return (f"আপনার অভিযোগটি{about} আমরা পেয়েছি। আমাদের {team} বিষয়টি যাচাই করবে এবং অফিসিয়াল সাপোর্ট "
                "চ্যানেলের মাধ্যমে আপনাকে জানাবে। যেকোনো প্রযোজ্য পরিমাণ অফিসিয়াল চ্যানেলের মাধ্যমে প্রক্রিয়া "
                f"করা হবে। নিরাপত্তার জন্য কারো সাথে আপনার পিন, ওটিপি বা পাসওয়ার্ড শেয়ার করবেন না। "
                f"রেফারেন্স: {ticket_id}।")

    team = TEAM_EN.get(department, "support team")
    about = f" regarding transaction {rtid}" if rtid else ""
    if case_type == "phishing_or_social_engineering":
        return ("Thank you for reaching out before sharing any information. Our fraud and risk team is "
                "reviewing this. Please never share your PIN, OTP, or password with anyone — our staff will "
                "never ask for them. We will only contact you through official support channels. "
                f"Reference: {ticket_id}.")
    if case_type == "other" or not rtid:
        return ("Thank you for reaching out. To identify the right transaction, please share the transaction "
                f"ID, the amount involved, and the approximate time. Our {team} will then review your case "
                "through official support channels. For your security, please do not share your PIN, OTP, or "
                f"password with anyone. Reference: {ticket_id}.")
    if case_type == "refund_request":
        return ("Thank you for reaching out. Refund eligibility for completed payments depends on the "
                f"applicable policy. Our {team} will review your request, and any eligible amount will be "
                "processed through official channels. For your security, please do not share your PIN, OTP, or "
                f"password with anyone. Reference: {ticket_id}.")
    return (f"Thank you for reaching out. We have noted your concern{about}. Our {team} will review the case "
            "and update you through our official support channels. Any eligible amount will be processed "
            "through official channels once the review is complete. For your security, please do not share "
            f"your PIN, OTP, or password with anyone. Reference: {ticket_id}.")


def _default_action(case_type: str, rtid: Optional[str]) -> str:
    """Safe operational next step (used if a model action makes a customer promise)."""
    ref = f" {rtid}" if rtid else ""
    actions = {
        "wrong_transfer": f"Verify transaction{ref} with the customer and route to dispute resolution per the wrong-transfer workflow.",
        "payment_failed": f"Investigate the ledger status of transaction{ref} with payments operations and follow the standard reversal procedure if applicable.",
        "duplicate_payment": f"Verify the duplicate with payments operations and follow the reversal procedure for transaction{ref} if confirmed.",
        "merchant_settlement_delay": f"Route to merchant operations to verify the settlement batch for transaction{ref}.",
        "agent_cash_in_issue": f"Reconcile the agent cash-in transaction{ref} with agent operations.",
        "phishing_or_social_engineering": "Escalate to fraud & risk and advise the customer to secure their account through official support.",
        "refund_request": f"Route the refund request for transaction{ref} per policy without committing to an outcome.",
        "other": "Request the transaction ID, amount, and time from the customer, then route accordingly.",
    }
    return actions.get(case_type, actions["other"])


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


# Credential tokens (English + Bangla). Matched as whole tokens.
_CRED_TOKENS = (
    "pin", "otp", "password", "passcode", "cvv", "credentials", "card number",
    "one time password", "one-time password", "security code", "verification code",
    "পিন", "ওটিপি", "পাসওয়ার্ড",
)
_CRED_TOKEN_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(t) for t in _CRED_TOKENS) + r")(?!\w)"
)
# Verbs that REQUEST a credential (English + Banglish + Bangla).
_SOLICIT_RE = re.compile(
    r"\b(share|provide|enter|confirm|send|give|tell|type|input|reveal|submit|need|"
    r"require|verify with|what(?:'s| is)|whats|forward|dite|diye|den|dao|পাঠান|"
    r"লিখুন|বলুন|জানান|দিন|দিবেন|দাও)\b|দিয়ে"
)
# Negation: marks the credential mention as a SAFE warning, not a request.
# Bangla negators use a Bangla-letter boundary so "না" inside "আপনার" (= your)
# is NOT treated as a negation.
_NEGATION_RE = re.compile(
    r"\b(not|never|do not|don't|dont|does not|doesn't|will not|won't|cannot|"
    r"can not|can't|cant|should not|shouldn't|must not|avoid|without|no need|"
    r"kokhono|kokhonoi)\b"
    r"|(?<![ঀ-৿])(না|নাই|কখনো)(?![ঀ-৿])"
)


def find_credential_violations(reply: str) -> List[str]:
    """
    Return credential SOLICITATIONS only.

    The rubric allows warning users not to share their PIN/OTP (the official
    sample replies do exactly that). It penalises only REQUESTING credentials.
    So a credential token is a violation only when a request verb is nearby AND
    there is no negation in the surrounding window ("do not share your PIN" is
    safe; "please share your PIN" is not).
    """
    text = _norm(reply)
    violations: List[str] = []
    for m in _CRED_TOKEN_RE.finditer(text):
        lo, hi = max(0, m.start() - 45), min(len(text), m.end() + 45)
        window = text[lo:hi]
        if _NEGATION_RE.search(window):
            continue  # it's a "do not share ..." warning → safe
        if _SOLICIT_RE.search(window):
            violations.append(m.group(1))
    return violations


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

def sanitize_reply(reply: str, ticket_id: str, *, phishing: bool = False,
                   bangla: bool = False) -> Tuple[str, List[str]]:
    """
    Return a guaranteed-safe customer reply plus reason codes for any correction.

    We KEEP the generated reply (which may warn users not to share credentials —
    that is allowed and desirable) and only replace it wholesale with a vetted
    template when it is empty or actually SOLICITS credentials / promises a refund.
    Replacement templates are language-aware (English or Bangla).
    """
    reasons: List[str] = []
    safe = (reply or "").strip()

    def fallback() -> str:
        if phishing:
            tpl = PHISHING_SAFE_REPLY_BN if bangla else PHISHING_SAFE_REPLY
        else:
            tpl = SAFE_FALLBACK_REPLY_BN if bangla else SAFE_FALLBACK_REPLY
        return tpl.format(ticket_id=ticket_id)

    if not safe:
        reasons.append("empty_reply_replaced")
        return fallback(), reasons

    cred_hits = find_credential_violations(safe)
    promise_hits = find_promise_violations(safe)
    if cred_hits:
        reasons.append("credential_solicitation_blocked")
    if promise_hits:
        reasons.append("refund_promise_blocked")
    if cred_hits or promise_hits:
        return fallback(), reasons

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

    # --- Customer reply: generated deterministically here (the safety layer is
    #     the single source of truth for the customer-facing reply). It is
    #     team-attributed, language-aware, and provably safe. ---
    bangla = _is_bangla(complaint)
    out["customer_reply"] = build_customer_reply(
        case_type, out["department"], out["relevant_transaction_id"], ticket_id, bangla
    )

    # --- recommended_next_action is an agent instruction; it must not contain a
    #     customer-facing promise (the -10 rule also checks this field). ---
    if find_promise_violations(out.get("recommended_next_action", "")):
        out["recommended_next_action"] = _default_action(case_type, out["relevant_transaction_id"])
        reason_codes.append("action_promise_blocked")

    # --- human_review_required (escalate disputes / suspicious / inconsistent /
    #     critical; clarification-only cases do not need review) ---
    review_cases = {
        "wrong_transfer", "duplicate_payment", "agent_cash_in_issue",
        "phishing_or_social_engineering",
    }
    hrr = (
        out["evidence_verdict"] == "inconsistent"
        or severity == "critical"
        or case_type in review_cases
    )
    # A wrong-transfer we cannot even identify yet is a clarification request,
    # not a dispute — do not flag for review until the transaction is confirmed.
    if case_type == "wrong_transfer" and out["evidence_verdict"] == "insufficient_data":
        hrr = False
    if injection:
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
