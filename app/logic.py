"""
logic.py — Evidence reasoning + classification for QueueStorm Investigator.

Architecture (the rubric recommends: rules for evidence/safety, AI for language):

  * The deterministic RULE ENGINE is authoritative for every scored decision
    field — relevant_transaction_id, evidence_verdict, case_type, severity,
    department, human_review_required. It handles English, Bangla (বাংলা), and
    Banglish, extracts amounts / times / counterparties, disambiguates between
    transactions, and detects ambiguity / duplicates / established recipients.

  * Groq (llama-3.3-70b-versatile) is used ONLY to draft the natural-language
    text fields (agent_summary, recommended_next_action, customer_reply) and to
    reply in the customer's language. It never changes the decision. If Groq is
    absent or fails, deterministic templates (English + Bangla) are used.

  * Prompt-injection text is detected and never reaches the LLM.

`safety.enforce_safety()` always runs last (in main.py) and overrides anything
unsafe and finalizes human_review_required.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("queuestorm.logic")

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "4.0"))
USE_GROQ = os.getenv("USE_GROQ", "1").strip().lower() not in ("0", "false", "no")

_groq_client = None


def _get_groq_client():
    global _groq_client
    if not USE_GROQ:
        return None
    if _groq_client is not None:
        return _groq_client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS)
        return _groq_client
    except Exception as exc:  # pragma: no cover
        logger.warning("Groq client init failed: %s", exc)
        return None


# ===========================================================================
# Normalisation
# ===========================================================================

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.translate(_BN_DIGITS).lower()).strip()


def _is_bangla(text: str, language: Optional[str]) -> bool:
    if language == "bn":
        return True
    return any("ঀ" <= ch <= "৿" for ch in (text or ""))


# ===========================================================================
# Amount extraction
# ===========================================================================

_MULTIPLIERS: Dict[str, float] = {
    "k": 1_000, "hazar": 1_000, "hajar": 1_000, "thousand": 1_000, "হাজার": 1_000,
    "hundred": 100, "shoto": 100, "sho": 100, "শত": 100, "শো": 100,
    "lakh": 100_000, "lac": 100_000, "lakhs": 100_000, "lacs": 100_000,
    "লাখ": 100_000, "লক্ষ": 100_000,
    "crore": 10_000_000, "koti": 10_000_000, "কোটি": 10_000_000,
}
_WORD_NUMBERS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_CURRENCY_MARKERS = ("taka", "tk", "bdt", "৳", "টাকা")

_NUM_RE = re.compile(
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<mult>k|hazar|hajar|thousand|hundred|shoto|sho|lakh|lac|lakhs|lacs|crore|koti|হাজার|শত|শো|লাখ|লক্ষ|কোটি)?",
    re.IGNORECASE,
)
_WORD_NUM_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+"
    r"(thousand|hundred|lakh|lac|hajar|hazar|crore)\b",
    re.IGNORECASE,
)


def extract_amounts(text: str) -> Set[float]:
    amounts: Set[float] = set()
    if not text:
        return amounts
    norm = text.translate(_BN_DIGITS).lower()
    for m in _WORD_NUM_RE.finditer(norm):
        base = _WORD_NUMBERS.get(m.group(1).lower())
        mult = _MULTIPLIERS.get(m.group(2).lower())
        if base and mult:
            amounts.add(float(base * mult))
    for m in _NUM_RE.finditer(norm):
        raw = m.group("num")
        start = m.start("num")
        if start > 0 and (norm[start - 1].isalpha() or norm[start - 1] == "-"):
            continue
        digits_only = raw.replace(",", "")
        mult_token = (m.group("mult") or "").lower()
        if not mult_token and len(digits_only.replace(".", "")) >= 10:
            tail = norm[m.end():m.end() + 8]
            if not any(c in tail for c in _CURRENCY_MARKERS):
                continue
        try:
            value = float(digits_only)
        except ValueError:
            continue
        if mult_token:
            value *= _MULTIPLIERS.get(mult_token, 1)
        if value >= 1:
            amounts.add(value)
    return amounts


# ===========================================================================
# Phone / counterparty
# ===========================================================================

_PHONE_RE = re.compile(r"(?:\+?880|0)?1[3-9]\d{8}")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def extract_phones(text: str) -> Set[str]:
    out: Set[str] = set()
    norm = (text or "").translate(_BN_DIGITS)
    for m in _PHONE_RE.finditer(norm):
        d = _digits(m.group(0))
        if len(d) >= 10:
            out.add(d[-11:] if len(d) >= 11 else d)
    return out


# ===========================================================================
# Time
# ===========================================================================

_TIME_BANDS: List[Tuple[Tuple[str, ...], Tuple[int, int]]] = [
    (("morning", "sokal", "shokal", "সকাল", "bhor", "ভোর"), (5, 11)),
    (("noon", "midday", "dupur", "দুপুর"), (11, 15)),
    (("afternoon", "bikel", "bikal", "বিকেল", "বিকাল"), (14, 18)),
    (("evening", "shondha", "sondha", "সন্ধ্যা"), (17, 20)),
    (("night", "raat", "rat", "রাত", "midnight"), (20, 24)),
]
_RECENT_PHRASES = (
    "just now", "right now", "a moment ago", "moments ago", "akhon", "এখন",
    "এইমাত্র", "একটু আগে", "ektu age", "just transferred", "just sent", "just paid",
)
_EXPLICIT_TIME_RE = re.compile(r"\b(\d{1,2})\s*(?::\s*(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_HHMM_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def extract_time_hours(text: str) -> Set[int]:
    hours: Set[int] = set()
    norm = _normalize(text)
    for m in _EXPLICIT_TIME_RE.finditer(norm):
        hour = int(m.group(1)) % 12
        if m.group(3).lower() == "pm":
            hour += 12
        hours.add(hour)
    for m in _HHMM_RE.finditer(norm):
        hours.add(int(m.group(1)))
    for keywords, (lo, hi) in _TIME_BANDS:
        if any(k in norm for k in keywords):
            for h in range(lo, hi):
                hours.add(h % 24)
    return hours


def has_recent_phrase(text: str) -> bool:
    norm = _normalize(text)
    return any(p in norm for p in _RECENT_PHRASES)


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    cleaned = ts.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(ts.strip(), fmt)
            except ValueError:
                continue
    return None


# ===========================================================================
# Keyword taxonomies
# ===========================================================================

TYPE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "transfer": ("transfer", "sent", "send money", "pathiyechi", "pathai", "pathaisi",
                 "পাঠিয়েছি", "পাঠাইছি", "send korechi", "send korsi", "transfer korsi"),
    "payment": ("payment", "paid", "pay", "bill", "purchase", "merchant", "checkout",
                "recharge", "kinlam", "kinechi", "পেমেন্ট", "বিল", "রিচার্জ"),
    "cash_in": ("cash in", "cash-in", "add money", "deposit", "ক্যাশ ইন", "ঢুকাইছি"),
    "cash_out": ("cash out", "cash-out", "withdraw", "withdrawal", "tola", "তুললাম", "ক্যাশ আউট"),
    "settlement": ("settlement", "settle", "payout", "সেটেলমেন্ট"),
    "refund": ("refund", "reversal", "money back", "ferot", "ফেরত"),
}

WRONG_TRANSFER_KW = ("wrong number", "wrong person", "wrong account", "wrong recipient",
                     "bhul number", "vul number", "ভুল নম্বর", "ভুল মানুষ", "ভুল অ্যাকাউন্ট",
                     "mistakenly sent", "wrongly sent", "sent to wrong", "vul manush", "wrong gateway")
DUPLICATE_KW = ("duplicate", "charged twice", "two times", "double charge", "double charged",
                "deducted twice", "dui bar", "duibar", "দুইবার", "twice deducted", "twice kete",
                "charged 2 times", "deducted 2 times")
PAYMENT_FAILED_KW = ("failed", "fail", "declined", "didn't go through", "did not go through",
                     "unsuccessful", "hoyni", "hoy nai", "fail hoise", "fail hoyeche", "ব্যর্থ", "হয়নি")
SETTLEMENT_KW = ("settlement", "settle", "merchant payout", "not settled", "settle hoy nai",
                 "settle hoyni", "সেটেলমেন্ট", "settlement pai nai")
AGENT_KW = ("cash in", "cash-in", "agent", "এজেন্ট", "agent er kache", "agent point", "ক্যাশ ইন")
REFUND_KW = ("refund", "money back", "return my money", "ferot", "ফেরত", "taka ferot",
             "ferot chai", "refund chai", "want my money back")
PHISHING_KW = ("otp", "scam", "fraud", "hacked", "hack", "unauthorized", "unauthorised", "phishing",
               "প্রতারণা", "প্রতারক", "ভুয়া", "fake call", "fake agent", "someone called", "keu phone",
               "কেউ ফোন", "stole", "stolen", "churi", "চুরি", "click korlam link", "code chaiche",
               "asked for my otp", "asked for my pin", "pin chaiche", "otp chaiche")

# State language → what the customer asserts.
FAILURE_WORDS = ("failed", "fail", "declined", "unsuccessful", "didn't go through",
                 "did not go through", "hoyni", "hoy nai", "ব্যর্থ", "হয়নি", "fail hoise")
DEDUCTED_WORDS = ("deducted", "debited", "charged", "cut", "kete niyeche", "kete nilo",
                  "kete niche", "kete nieche", "taka geche", "taka kete", "কেটে নিয়েছে",
                  "কাটা হয়েছে", "balance kome", "katche", "kete felse", "deduct")
RECEIVED_NEG_WORDS = ("not received", "didn't receive", "did not receive", "haven't received",
                      "hasn't received", "didn't get", "did not get", "didn't get it", "not reflected",
                      "did not reflect", "pai nai", "paini", "পাইনি", "পাই নাই", "আসেনি",
                      "balance e ase nai", "balance e ashe nai", "dekhchi na", "দেখছি না", "pelam na")
PENDING_WORDS = ("pending", "processing", "stuck", "atke", "আটকে", "ঝুলে", "jhule",
                 "delay", "deri", "দেরি", "still waiting", "not settled", "not been settled")
COMPLETED_WORDS = ("completed", "successful", "hoyeche", "hoise", "sofol", "সফল", "হয়েছে", "peyeche")


def _any(text: str, words: Tuple[str, ...]) -> bool:
    return any(w in text for w in words)


# ===========================================================================
# Prompt-injection detection
# ===========================================================================

INJECTION_PATTERNS: Tuple[str, ...] = (
    "ignore previous instructions", "ignore all previous", "ignore the above",
    "disregard previous", "disregard all instructions", "disregard your",
    "you are now", "you're now", "pretend you are", "pretend to be",
    "act as a", "act as an", "act like", "system prompt", "developer mode",
    "jailbreak", "forget your instructions", "forget all previous", "new instructions:",
    "override your", "reveal your prompt", "print your prompt", "you must now",
)


def detect_injection(text: str) -> bool:
    norm = _normalize(text)
    return any(p in norm for p in INJECTION_PATTERNS)


# ===========================================================================
# Transaction matching
# ===========================================================================

def score_transaction(complaint_norm: str, amounts: Set[float], phones: Set[str],
                      hours: Set[int], txn: Dict[str, Any], txn_ts: Optional[datetime]) -> float:
    score = 0.0
    txn_id = str(txn.get("transaction_id") or "")
    if txn_id and txn_id.lower() in complaint_norm:
        score += 6.0
    try:
        txn_amount = float(txn.get("amount"))
    except (TypeError, ValueError):
        txn_amount = None
    if txn_amount is not None and amounts:
        if any(abs(a - txn_amount) < 0.01 for a in amounts):
            score += 3.0
        elif txn_amount > 0 and any(abs(a - txn_amount) / txn_amount <= 0.02 for a in amounts):
            score += 2.0
    counterparty = str(txn.get("counterparty") or "")
    cp_norm = _normalize(counterparty)
    cp_digits = _digits(counterparty)
    if cp_digits and phones and (cp_digits[-11:] in phones or any(p in cp_digits or cp_digits in p for p in phones)):
        score += 2.5
    elif cp_norm and len(cp_norm) >= 3 and cp_norm in complaint_norm:
        score += 2.0
    if hours and txn_ts is not None and txn_ts.hour in hours:
        score += 2.0
    txn_type = str(txn.get("type") or "").lower()
    if _any(complaint_norm, TYPE_KEYWORDS.get(txn_type, ())):
        score += 1.0
    status = str(txn.get("status") or "").lower()
    if status == "failed" and _any(complaint_norm, FAILURE_WORDS):
        score += 1.0
    if status == "pending" and _any(complaint_norm, PENDING_WORDS):
        score += 1.0
    if status == "completed" and _any(complaint_norm, DEDUCTED_WORDS):
        score += 0.5
    if status == "reversed" and _any(complaint_norm, REFUND_KW):
        score += 0.5
    return score


def _referenced_amount_matches(amounts: Set[float], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in history:
        try:
            a = float(t.get("amount"))
        except (TypeError, ValueError):
            continue
        if any(abs(a - amt) < 0.01 for amt in amounts):
            out.append(t)
    return out


def _choose_duplicate(history: List[Dict[str, Any]], amounts: Set[float]) -> Optional[Dict[str, Any]]:
    """For duplicate_payment: return the LATER transaction of the duplicate pair."""
    groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for t in history:
        key = (t.get("amount"), str(t.get("counterparty") or ""))
        groups.setdefault(key, []).append(t)
    candidates = [g for g in groups.values() if len(g) >= 2]
    if not candidates:
        return None
    # Prefer the group whose amount the complaint referenced.
    if amounts:
        for g in candidates:
            try:
                if any(abs(float(g[0].get("amount")) - a) < 0.01 for a in amounts):
                    candidates = [g]
                    break
            except (TypeError, ValueError):
                pass
    group = max(candidates, key=len)
    dated = [(t, _parse_timestamp(t.get("timestamp"))) for t in group]
    dated_known = [(t, ts) for t, ts in dated if ts is not None]
    if dated_known:
        return max(dated_known, key=lambda p: p[1])[0]
    return group[-1]


def _established_recipient(txn: Dict[str, Any], history: List[Dict[str, Any]]) -> bool:
    cp = str(txn.get("counterparty") or "")
    if not cp:
        return False
    return sum(1 for t in history if str(t.get("counterparty") or "") == cp) >= 2


def rank_transactions(complaint: str, history: List[Dict[str, Any]]):
    complaint_norm = _normalize(complaint)
    amounts = extract_amounts(complaint)
    phones = extract_phones(complaint)
    hours = extract_time_hours(complaint)
    recent = has_recent_phrase(complaint)
    parsed = [_parse_timestamp(t.get("timestamp")) for t in history]
    recent_idx = None
    dated = [(i, ts) for i, ts in enumerate(parsed) if ts is not None]
    if dated:
        recent_idx = max(dated, key=lambda p: p[1])[0]
    scored = []
    for i, txn in enumerate(history):
        s = score_transaction(complaint_norm, amounts, phones, hours, txn, parsed[i])
        if recent and i == recent_idx:
            s += 2.0
        scored.append((s, i, txn))
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    return scored, amounts, phones, hours


# ===========================================================================
# Classification
# ===========================================================================

def classify_case_type(complaint: str, txn: Optional[Dict[str, Any]]) -> str:
    text = _normalize(complaint)
    if _any(text, PHISHING_KW):
        return "phishing_or_social_engineering"
    if _any(text, DUPLICATE_KW):
        return "duplicate_payment"
    if _any(text, WRONG_TRANSFER_KW):
        return "wrong_transfer"
    if _any(text, SETTLEMENT_KW):
        return "merchant_settlement_delay"
    if _any(text, AGENT_KW) and _any(text, ("cash in", "cash-in", "deposit", "add money", "ক্যাশ ইন", "agent", "এজেন্ট")):
        if _any(text, RECEIVED_NEG_WORDS) or _any(text, ("cash in", "cash-in", "deposit", "ক্যাশ ইন")):
            return "agent_cash_in_issue"
    if _any(text, PAYMENT_FAILED_KW):
        return "payment_failed"
    if _any(text, REFUND_KW):
        return "refund_request"

    # Transfer that the recipient did not receive → wrong-transfer dispute.
    transfer_words = ("sent", "send", "transfer", "pathai", "pathiye", "পাঠিয়েছি", "পাঠাইছি")
    if _any(text, RECEIVED_NEG_WORDS) and (_any(text, transfer_words) or (txn and str(txn.get("type")) == "transfer")):
        return "wrong_transfer"

    if txn:
        t = str(txn.get("type") or "")
        status = str(txn.get("status") or "")
        if t == "transfer":
            return "wrong_transfer" if _any(text, WRONG_TRANSFER_KW) else "other"
        if t == "payment":
            return "payment_failed" if status in ("failed", "pending") else "other"
        if t == "settlement":
            return "merchant_settlement_delay"
        if t == "cash_in":
            return "agent_cash_in_issue"
        if t == "refund":
            return "refund_request"
    return "other"


def compute_evidence_verdict(complaint: str, txn: Optional[Dict[str, Any]]) -> str:
    if txn is None:
        return "insufficient_data"
    text = _normalize(complaint)
    status = str(txn.get("status") or "").lower()
    claims_failure = _any(text, FAILURE_WORDS)
    claims_deducted = _any(text, DEDUCTED_WORDS)
    claims_not_received = _any(text, RECEIVED_NEG_WORDS)
    claims_pending = _any(text, PENDING_WORDS)
    claims_completed = _any(text, COMPLETED_WORDS)

    if status == "completed":
        if claims_failure and not claims_deducted:
            return "inconsistent"
        if claims_pending and not claims_deducted:
            return "inconsistent"
        return "consistent"
    if status == "failed":
        return "consistent"
    if status == "pending":
        if claims_completed and not claims_not_received:
            return "inconsistent"
        return "consistent"
    if status == "reversed":
        return "consistent"
    return "consistent"


def compute_severity(text: str, case_type: str, txn: Optional[Dict[str, Any]],
                    amounts: Set[float], verdict: str, ambiguous: bool) -> str:
    amount: Optional[float] = None
    if txn is not None:
        try:
            amount = float(txn.get("amount"))
        except (TypeError, ValueError):
            amount = None
    if amount is None and amounts:
        amount = max(amounts)

    if case_type == "phishing_or_social_engineering":
        return "critical"
    if amount is not None and amount > 50_000:
        return "critical"

    if case_type == "wrong_transfer":
        return "medium" if (verdict == "inconsistent" or ambiguous) else "high"
    if case_type == "payment_failed":
        return "high" if _any(text, DEDUCTED_WORDS) else "medium"
    if case_type == "duplicate_payment":
        return "high"
    if case_type == "agent_cash_in_issue":
        return "high"
    if case_type == "merchant_settlement_delay":
        return "medium"
    if case_type == "refund_request":
        return "medium" if (amount is not None and amount > 10_000) else "low"
    if case_type == "other":
        return "low" if txn is None else "medium"

    if amount is not None and amount > 10_000:
        return "high"
    if amount is not None and amount >= 1_000:
        return "medium"
    return "low"


# ===========================================================================
# Deterministic text templates (English + Bangla)
# ===========================================================================

def _about(txn: Optional[Dict[str, Any]]) -> str:
    tid = txn.get("transaction_id") if txn else None
    return f" regarding transaction {tid}" if tid else ""


def _build_summary(case_type: str, verdict: str, txn: Optional[Dict[str, Any]], severity: str, ambiguous: bool) -> str:
    nice = case_type.replace("_", " ")
    if ambiguous:
        return (f"Customer reports a {nice} issue, but multiple transactions plausibly match and "
                f"the complaint lacks a disambiguator; cannot confirm the relevant transaction.")
    if txn is not None:
        return (f"Customer reports a {nice} issue ({severity} severity). Matched transaction "
                f"{txn.get('transaction_id')} ({txn.get('type')}, {txn.get('amount')} BDT to "
                f"{txn.get('counterparty')}, status {txn.get('status')}); evidence is {verdict.replace('_', ' ')}.")
    return (f"Customer reports a {nice} issue ({severity} severity). No matching transaction was found "
            f"in the provided history; evidence is {verdict.replace('_', ' ')}.")


def _build_action(case_type: str, verdict: str, txn: Optional[Dict[str, Any]], ambiguous: bool) -> str:
    if ambiguous:
        return ("Reply to the customer asking for a disambiguating detail (recipient number, exact amount, "
                "or transaction ID) before identifying the transaction. Do not initiate a dispute yet.")
    tid = txn.get("transaction_id") if txn else None
    ref = f" {tid}" if tid else ""
    actions = {
        "wrong_transfer": f"Verify transaction{ref} with the customer and initiate the wrong-transfer dispute workflow per policy.",
        "payment_failed": f"Investigate the ledger status of transaction{ref}; if the balance was deducted on a failed payment, initiate the automatic reversal flow within SLA.",
        "duplicate_payment": f"Verify the duplicate with payments operations; if the biller confirms a single charge, initiate reversal of the duplicate transaction{ref}.",
        "merchant_settlement_delay": f"Route to merchant operations to verify the settlement batch status for transaction{ref} and communicate a revised ETA.",
        "agent_cash_in_issue": f"Investigate the pending cash-in transaction{ref} with agent operations and confirm the settlement state within the cash-in SLA.",
        "phishing_or_social_engineering": "Escalate to the fraud & risk team immediately. Confirm to the customer that the company never asks for OTP/PIN, and log the reported source for fraud analysis.",
        "refund_request": f"Inform the customer that refund eligibility for transaction{ref} depends on policy/the merchant; route accordingly without committing to an outcome.",
        "other": "Reply to the customer requesting specifics: transaction ID, amount, what went wrong, and approximate time.",
    }
    base = actions.get(case_type, actions["other"])
    if verdict == "inconsistent":
        base = "Flag the discrepancy between the complaint and the ledger for human review. " + base
    return base


def _reply_en(case_type: str, ticket_id: str, txn: Optional[Dict[str, Any]], ambiguous: bool) -> str:
    if case_type == "phishing_or_social_engineering":
        return ("Thank you for reaching out before sharing any information. We never ask for your PIN, OTP, "
                "or password under any circumstances, and you should never share them with anyone, even if they "
                f"claim to be from us. Our fraud and risk team has been notified. Reference: {ticket_id}.")
    if ambiguous or case_type == "other":
        return ("Thank you for reaching out. To help you faster, please share the transaction ID, the amount "
                "involved, and the approximate time of the transaction in question. For your security, please do "
                f"not share your PIN, OTP, or password with anyone. Reference: {ticket_id}.")
    if case_type == "refund_request":
        return ("Thank you for reaching out. Refund eligibility for completed payments depends on the applicable "
                "policy, and any eligible amount will be processed through official channels after review. Please "
                f"do not share your PIN, OTP, or password with anyone. Reference: {ticket_id}.")
    return (f"Thank you for reaching out. We have noted your concern{_about(txn)} and our team will investigate "
            "this matter. Any eligible amount will be processed through official channels once the review is "
            "complete. For your security, please do not share your PIN, OTP, or password with anyone. "
            f"Reference: {ticket_id}.")


def _reply_bn(case_type: str, ticket_id: str, txn: Optional[Dict[str, Any]], ambiguous: bool) -> str:
    about = f" (লেনদেন {txn.get('transaction_id')})" if txn and txn.get("transaction_id") else ""
    if case_type == "phishing_or_social_engineering":
        return ("কোনো তথ্য শেয়ার করার আগে যোগাযোগ করার জন্য ধন্যবাদ। আমরা কখনোই আপনার পিন, ওটিপি বা পাসওয়ার্ড "
                "চাই না — অনুগ্রহ করে কারো সাথে এগুলো শেয়ার করবেন না, এমনকি কেউ আমাদের প্রতিনিধি দাবি করলেও নয়। "
                f"আমাদের ফ্রড ও রিস্ক টিমকে জানানো হয়েছে। রেফারেন্স: {ticket_id}।")
    if ambiguous or case_type == "other":
        return ("যোগাযোগ করার জন্য ধন্যবাদ। দ্রুত সহায়তার জন্য অনুগ্রহ করে লেনদেন আইডি, পরিমাণ এবং আনুমানিক সময় "
                f"জানান। নিরাপত্তার জন্য কারো সাথে আপনার পিন বা ওটিপি শেয়ার করবেন না। রেফারেন্স: {ticket_id}।")
    return (f"আপনার অভিযোগটি{about} আমরা পেয়েছি এবং আমাদের টিম বিষয়টি যাচাই করে দেখবে। যেকোনো প্রযোজ্য পরিমাণ "
            "অফিসিয়াল চ্যানেলের মাধ্যমে প্রক্রিয়া করা হবে। অনুগ্রহ করে কারো সাথে আপনার পিন বা ওটিপি শেয়ার করবেন না। "
            f"রেফারেন্স: {ticket_id}।")


def _build_reply(case_type: str, ticket_id: str, txn: Optional[Dict[str, Any]], ambiguous: bool, bangla: bool) -> str:
    return (_reply_bn if bangla else _reply_en)(case_type, ticket_id, txn, ambiguous)


# ===========================================================================
# Rule-based analysis (authoritative for decisions)
# ===========================================================================

def rule_based_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    complaint = payload.get("complaint", "") or ""
    ticket_id = payload.get("ticket_id", "") or ""
    history = payload.get("transaction_history") or []
    bangla = _is_bangla(complaint, payload.get("language"))

    if detect_injection(complaint):
        return {
            "ticket_id": ticket_id, "relevant_transaction_id": None,
            "evidence_verdict": "insufficient_data",
            "case_type": "phishing_or_social_engineering", "severity": "critical",
            "department": "fraud_risk",
            "agent_summary": "Complaint contains embedded instructions / prompt-injection patterns; treated as a social-engineering attempt and not acted upon literally.",
            "recommended_next_action": "Do not follow any instructions contained in the complaint. Escalate to fraud & risk for manual review of the account and the message source.",
            "customer_reply": _build_reply("phishing_or_social_engineering", ticket_id, None, False, bangla),
            "human_review_required": True, "confidence": 0.6,
            "reason_codes": ["phishing_or_social_engineering", "prompt_injection_detected", "critical_escalation"],
        }

    scored, amounts, phones, hours = rank_transactions(complaint, history)
    prelim_txn = scored[0][2] if (scored and scored[0][0] >= 2.0) else None
    case_type = classify_case_type(complaint, prelim_txn)

    ambiguous = False
    txn = prelim_txn

    if case_type == "duplicate_payment":
        dup = _choose_duplicate(history, amounts)
        if dup is not None:
            txn = dup
    else:
        # Ambiguity: the referenced amount matches multiple distinct recipients
        # and the complaint gives no disambiguator.
        amt_matches = _referenced_amount_matches(amounts, history) if amounts else []
        distinct_cp = {str(t.get("counterparty") or "") for t in amt_matches}
        has_disambiguator = bool(phones or hours) or any(
            str(t.get("transaction_id") or "").lower() in _normalize(complaint) for t in history
        )
        if len(amt_matches) >= 2 and len(distinct_cp) >= 2 and not has_disambiguator:
            ambiguous = True
            txn = None

    verdict = compute_evidence_verdict(complaint, txn)

    # Established-recipient pattern contradicts a wrong-transfer claim.
    established = (case_type == "wrong_transfer" and txn is not None and not ambiguous
                  and _established_recipient(txn, history))
    if established:
        verdict = "inconsistent"

    severity = compute_severity(_normalize(complaint), case_type, txn, amounts, verdict, ambiguous)
    rtid = str(txn.get("transaction_id")) if txn and txn.get("transaction_id") else None

    confidence = 0.55
    if ambiguous:
        confidence = 0.6
    elif txn is not None:
        confidence = min(0.95, 0.6 + 0.07 * scored[0][0])
    elif not history:
        confidence = 0.7

    # Meaningful, decision-supporting labels (no internal telemetry markers).
    reason_codes: List[str] = [case_type]
    if ambiguous:
        reason_codes += ["ambiguous_match", "needs_clarification"]
    elif txn is not None:
        reason_codes.append("transaction_match")
    elif history:
        reason_codes.append("no_relevant_transaction")
    else:
        reason_codes.append("empty_history")
    if established:
        reason_codes.append("established_recipient_pattern")
    if verdict == "inconsistent":
        reason_codes.append("evidence_inconsistent")
    if severity == "critical":
        reason_codes.append("critical_escalation")

    return {
        "ticket_id": ticket_id,
        "relevant_transaction_id": rtid,
        "evidence_verdict": verdict,
        "case_type": case_type,
        "severity": severity,
        "department": "customer_support",  # safety re-derives authoritatively
        "agent_summary": _build_summary(case_type, verdict, txn, severity, ambiguous),
        "recommended_next_action": _build_action(case_type, verdict, txn, ambiguous),
        "customer_reply": _build_reply(case_type, ticket_id, txn, ambiguous, bangla),
        "human_review_required": False,  # safety finalizes
        "confidence": round(confidence, 2),
        "reason_codes": reason_codes,
    }


# ===========================================================================
# Groq — TEXT generation only (never changes the decision)
# ===========================================================================

TEXT_SYSTEM_PROMPT = """You are QueueStorm Investigator, a support copilot for a Bangladeshi \
digital-payments company. You will be given a customer complaint, their transaction history, and the \
ALREADY-DECIDED analysis (case_type, evidence_verdict, relevant transaction, severity, department). \
Do NOT change those decisions. Your only job is to write two AGENT-FACING text fields:

- agent_summary: 1-2 sentence factual summary of the case for the support agent.
- recommended_next_action: a specific operational next step for the agent, consistent with the decision.

Rules:
- recommended_next_action is an internal instruction to the agent — it must NEVER promise the customer a \
refund/reversal/unblock and must NEVER ask for credentials. Operational steps like "verify the ledger" or \
"initiate the standard reversal procedure" are fine.
- Be factual and concise. Do not invent transactions or amounts.

Respond with ONLY a JSON object: {"agent_summary": "...", "recommended_next_action": "..."}"""


def _format_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "(no transactions on record)"
    return "\n".join(
        f"{i}. id={t.get('transaction_id')} time={t.get('timestamp')} type={t.get('type')} "
        f"amount={t.get('amount')} BDT counterparty={t.get('counterparty')} status={t.get('status')}"
        for i, t in enumerate(history, 1)
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", text.strip()).strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def groq_text(payload: Dict[str, Any], decision: Dict[str, Any]) -> Optional[Dict[str, str]]:
    client = _get_groq_client()
    if client is None:
        return None
    rtid = decision.get("relevant_transaction_id")
    user = (
        f"Complaint: {payload.get('complaint', '')}\n"
        f"language: {payload.get('language')}  channel: {payload.get('channel')}  user_type: {payload.get('user_type')}\n\n"
        f"Transaction history:\n{_format_history(payload.get('transaction_history') or [])}\n\n"
        f"DECIDED analysis (do not change):\n"
        f"- case_type: {decision.get('case_type')}\n"
        f"- evidence_verdict: {decision.get('evidence_verdict')}\n"
        f"- relevant_transaction_id: {rtid}\n"
        f"- severity: {decision.get('severity')}\n"
        f"- department: {decision.get('department')}\n\n"
        "Write agent_summary and recommended_next_action as JSON."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": TEXT_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        parsed = _extract_json(completion.choices[0].message.content)
        if not parsed:
            return None
        return {
            k: str(parsed[k]).strip()
            for k in ("agent_summary", "recommended_next_action")
            if isinstance(parsed.get(k), str) and parsed.get(k).strip()
        }
    except Exception as exc:
        logger.warning("Groq text generation failed, using templates: %s", exc)
        return None


# ===========================================================================
# Entry point
# ===========================================================================

def analyze_ticket(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rules decide everything. Groq only polishes the AGENT-facing text
    (agent_summary, recommended_next_action). The customer_reply is generated
    deterministically by the safety layer, so it is never left to the LLM.
    """
    complaint = payload.get("complaint", "") or ""
    result = rule_based_analysis(payload)

    if detect_injection(complaint):
        return result  # never send adversarial text to the LLM

    text = groq_text(payload, result)
    if text:
        if text.get("agent_summary"):
            result["agent_summary"] = text["agent_summary"]
        if text.get("recommended_next_action"):
            result["recommended_next_action"] = text["recommended_next_action"]
    return result
