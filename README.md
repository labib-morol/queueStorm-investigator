# QueueStorm Investigator

> bKash presents SUST CSE Carnival 2026 · Codex Community Hackathon · Online Preliminary

A safe, evidence-grounded **support-ticket investigator** for a digital-payments company.
It reads a customer complaint **plus that customer's recent transaction history**, figures
out *what actually happened*, routes the case to the right team, and drafts a customer reply
that **never asks for PIN/OTP/password** and **never promises a refund it cannot authorize**.

It is a **copilot for human support agents**, not an autonomous financial decision-maker.

**Live endpoint:** `https://queuestorm-investigator-ktsw.onrender.com`
- `GET /health` → `{"status":"ok"}`
- `POST /analyze-ticket` → structured analysis
- `GET /docs` → interactive Swagger UI (click *Try it out* → *Execute*)

✅ Matches **10/10** of the official public sample cases on every scored field
(`relevant_transaction_id`, `evidence_verdict`, `case_type`, `department`, `severity`,
`human_review_required`), with **0 safety violations** and **p95 ≈ 0.8s** latency.

---

## 1. What it does (the investigator twist)

The solution is not a complaint *classifier* — it's a complaint *investigator*. Every input
carries both the complaint and 2–5 recent transactions. The complaint says one thing; the data
may say another. The service decides what is true and exposes that reasoning explicitly:

| Field | Meaning |
|-------|---------|
| `relevant_transaction_id` | The transaction the complaint is about, or `null` if none matches |
| `evidence_verdict` | `consistent` (data supports the story) / `inconsistent` (data contradicts it) / `insufficient_data` (can't tell or ambiguous) |

When the evidence is genuinely unclear, it says `insufficient_data` instead of guessing — a
service that confirms a refund without checking the ledger is making the exact mistake real
fintech support must never make.

---

## 2. Architecture

The rubric recommends a **hybrid: rules for evidence/safety, AI for language.** That is exactly
this design — and it's why the decisions are deterministic and reproducible.

```
POST /analyze-ticket
        │
        ▼
  app/main.py ── parse + validate → 200 / 400 / 422 / 500 (no stack traces)
        │
        ▼
  app/logic.analyze_ticket()
        ├─ prompt-injection guard         (adversarial text never reaches the LLM)
        ├─ RULE ENGINE  (authoritative)   → relevant_transaction_id, evidence_verdict,
        │     amount/time/counterparty       case_type, severity, department,
        │     matching · disambiguation ·    human_review_required
        │     duplicate / established-recipient detection · EN/BN/Banglish
        └─ Groq (llama-3.3-70b-versatile) → ONLY drafts the 3 text fields,
              text-only, language-matched     replies in the customer's language
        │
        ▼
  app/safety.enforce_safety()  ── HARD guardrails, override everything, run LAST
        │
        ▼
   structured JSON (200)
```

| File | Responsibility |
|------|----------------|
| [app/main.py](app/main.py) | FastAPI app, request/response schema, HTTP status codes, error handling |
| [app/logic.py](app/logic.py) | Evidence reasoning, multilingual matching, rule engine + Groq text drafting |
| [app/safety.py](app/safety.py) | Hardcoded guardrails that validate, repair, and finalize every response |

**The rule engine decides; the LLM only writes prose.** If Groq is absent or fails, the service
falls back to deterministic English/Bangla templates — it never depends on the network and never
crashes.

---

## 3. Setup & run

```bash
pip install -r requirements.txt
cp .env.example .env        # optional: add GROQ_API_KEY for nicer, language-matched replies
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Windows: if the `uvicorn` shim isn't on PATH, use `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.

The service runs **with or without** a Groq key (no key → deterministic rule-based replies).

### Docker

```bash
docker build -t queuestorm-team .
docker run -p 8000:8000 --env-file .env queuestorm-team
```

Secrets are passed via `--env-file` only — never baked into the image. Image is slim (well under
the 500 MB target), binds `0.0.0.0`, and `/health` is ready within ~1s.

### Runbook (reproduce from scratch)

```bash
git clone https://github.com/labib-morol/queueStorm-investigator.git
cd queueStorm-investigator
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=...        # optional
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health        # -> {"status":"ok"}
```

---

## 4. API

### `GET /health`
Returns exactly `{"status":"ok"}`.

### `POST /analyze-ticket`

**Sample request**
```json
{
  "ticket_id": "TKT-001",
  "complaint": "I sent 5000 taka to a wrong number around 2pm today. Please help me get it back.",
  "language": "en",
  "channel": "in_app_chat",
  "user_type": "customer",
  "campaign_context": "boishakh_bonanza_day_1",
  "transaction_history": [
    { "transaction_id": "TXN-9101", "timestamp": "2026-04-14T14:08:22Z",
      "type": "transfer", "amount": 5000, "counterparty": "+8801719876543", "status": "completed" }
  ]
}
```

**Sample response** (full example saved in [sample_output.json](sample_output.json))
```json
{
  "ticket_id": "TKT-001",
  "relevant_transaction_id": "TXN-9101",
  "evidence_verdict": "consistent",
  "case_type": "wrong_transfer",
  "severity": "high",
  "department": "dispute_resolution",
  "agent_summary": "Customer reports a wrong transfer issue (high severity). Matched transaction TXN-9101 ...",
  "recommended_next_action": "Verify transaction TXN-9101 with the customer and initiate the wrong-transfer dispute workflow ...",
  "customer_reply": "Thank you for reaching out. We have noted your concern regarding transaction TXN-9101 ... Any eligible amount will be processed through official channels ... please do not share your PIN, OTP, or password ... Reference: TKT-001.",
  "human_review_required": true,
  "confidence": 0.9,
  "reason_codes": ["case:wrong_transfer", "verdict:consistent", "severity:high", "transaction_match"]
}
```

### HTTP status codes
| Code | When |
|------|------|
| 200 | Successful analysis (body conforms to the output schema) |
| 400 | Malformed JSON, or a missing required field (`ticket_id` / `complaint`) |
| 422 | Schema valid but complaint is empty/whitespace |
| 500 | Internal error — non-sensitive message only, never a stack trace |

The service never crashes on malformed input; unknown fields and partial transactions are tolerated.

### Enums (exact values)
- `evidence_verdict`: `consistent`, `inconsistent`, `insufficient_data`
- `case_type`: `wrong_transfer`, `payment_failed`, `refund_request`, `duplicate_payment`,
  `merchant_settlement_delay`, `agent_cash_in_issue`, `phishing_or_social_engineering`, `other`
- `severity`: `low`, `medium`, `high`, `critical`
- `department`: `customer_support`, `dispute_resolution`, `payments_ops`, `merchant_operations`,
  `agent_operations`, `fraud_risk`

---

## 5. Evidence reasoning (the 35-point core)

The rule engine cross-examines the complaint against **each** transaction and scores relevance:

- **Amount** — parsed from English (`5,000`, `5k`), Bangla digits (`৫০০০`), Banglish (`5 hajar`,
  `2 lakh`), and word-numbers (`five thousand`). Phone numbers and embedded transaction IDs are
  deliberately excluded so they're never mistaken for money.
- **Time** — explicit (`2pm`, `14:30`) and fuzzy bands (`morning`/`sokal`, `evening`/`bikel`,
  `raat`, `just now`/`akhon`) matched against transaction timestamps.
- **Counterparty** — normalised phone numbers and merchant names.
- **Type & status** — issue keywords aligned to transaction type and status.

On top of matching, it handles the subtle cases the sample pack tests for:

| Situation | Behaviour |
|-----------|-----------|
| **Ambiguous match** (same amount to several recipients, no disambiguator) | `relevant_transaction_id = null`, `insufficient_data`, ask for clarification (no premature dispute) |
| **Duplicate payment** (two identical charges) | points to the **later** transaction, `consistent`, `high` |
| **Established recipient** ("wrong transfer" to a number paid repeatedly) | `inconsistent` — flags the contradiction for human review |
| **Failed payment, balance deducted** | `consistent`, routes to `payments_ops` |
| **Empty / safety-only** complaints | `insufficient_data`, never guesses |

**Multilingual:** English, pure Bangla (বাংলা), and Banglish complaints describing the same event
produce the **same** decision, and the customer reply comes back **in the customer's language**.

---

## 6. Safety logic (the 20-point guardrails — `app/safety.py`)

Runs **last** and overrides any model output. It cannot be talked out of it.

1. **Never requests credentials.** `customer_reply` is scanned (negation-aware, English + Bangla)
   for PIN / OTP / password / CVV / card number. It distinguishes a **warning** ("please *do not*
   share your PIN" — allowed and encouraged) from a **request** ("share your PIN" — blocked and
   replaced with a vetted safe template). Boundary matching means "sho**pping**" is never flagged.
2. **Never promises an unauthorized outcome.** Blocks "we will refund / reverse / unblock /
   guarantee", including Banglish/Bangla forms (`taka ferot dibo`, `টাকা ফেরত দেব`). Approved
   phrasing: *"any eligible amount will be processed through official channels."*
3. **Prompt-injection defense.** Complaints containing "ignore previous instructions", "you are
   now", "pretend you are", "act as a", "jailbreak", etc. are forced to
   `phishing_or_social_engineering` → `fraud_risk` → `human_review_required=true`, and are
   **never sent to the LLM**.
4. **Official channels only** — replies never direct customers to third parties.
5. **Deterministic routing & escalation** — `department` is always re-derived from
   `case_type`/`severity` (contested refunds → `dispute_resolution`); disputes, suspicious,
   inconsistent, and critical cases are escalated for human review.

This directly addresses the rubric's safety penalties (−15 credential request, −10 unauthorized
refund, −10 third-party redirect) and the two-or-more-violations disqualifier.

---

## 7. MODELS

| Model | Where it runs | Why |
|-------|---------------|-----|
| **`llama-3.3-70b-versatile`** | Groq Cloud (external API) | Fast (~0.6–2s) drafting of `agent_summary`, `recommended_next_action`, and `customer_reply`, replying in the customer's language (English/Bangla/Banglish). Runs in strict JSON mode, `temperature=0.2`, 4s timeout. **Optional** — it only writes prose and **never changes a decision**. |
| **Rule engine (no model)** | In-process, deterministic | Authoritative for every scored field: transaction matching, evidence verdict, classification, severity, routing, and all safety guardrails. Also the complete fallback (English + Bangla templates) when Groq is unavailable. |

**Cost & runtime reasoning.** Groq is low-cost and very low-latency, keeping p95 well inside the
≤5s full-credit tier. Because the rule engine is a full fallback, the service has **no hard
dependency** on a paid API, quota, or rate limits during judging. No GPU, no local model weights,
no multi-GB downloads — fits the runtime profile comfortably.

---

## 8. Performance & reliability

- p95 latency **≈ 0.8s** across the 10 official cases (full-credit tier; hard limit is 30s).
- `/health` ready within ~1s (limit 60s).
- 4s Groq timeout → automatic deterministic fallback; valid requests never 5xx.
- Malformed JSON, missing fields, partial transactions, and unknown fields all handled gracefully.

---

## 9. Testing

A dependency-free batch runner compares the live service against a case file and flags
reasoning, schema, and safety issues automatically:

```bash
# official public sample pack (10 cases) — expect 10/10, 0 safety violations
python tests/run_cases.py --url https://queuestorm-investigator-ktsw.onrender.com \
                          --cases tests/official_sample_cases.json

# a broader starter set (multilingual, phishing, injection, edge cases)
python tests/run_cases.py --url http://127.0.0.1:8000 --cases tests/sample_cases.json
```

It prints per-case `case / verdict / rtid / department / severity / latency`, marks `DIFF`
(reasoning), `UNSAFE` (safety), or `SCHEMA` issues, and reports avg/max/p95 latency.

---

## 10. Assumptions & known limitations

- Transaction `timestamp` clock-hour is treated as the customer's local time (matches the
  problem statement's `14:08Z` ↔ "2pm" example); no timezone conversion is applied.
- Amount / time / counterparty extraction is heuristic; very unusual phrasings may not match, in
  which case the verdict conservatively falls back to `insufficient_data` rather than guessing.
- Word-number parsing covers common amounts (e.g. "five thousand"), not arbitrary spelled-out
  numbers.
- The LLM only drafts text; if it is unreachable, replies are deterministic templates (still
  language-aware for Bangla), which are correct and safe but less conversational.
- All data is synthetic; no real payment integration, as required.

## 11. Repository notes

- `.env` / `.env.local` are git-ignored; only `.env.example` (names, no values) is committed —
  **no secrets in the repo at any time.**
- Organizer access: GitHub handle **`bipulhf`** has read access.
- The empty `queuestorm-investigator/` folder is a leftover scaffold and is git-ignored.

## 12. Deliverables map

| Required deliverable | Where |
|----------------------|-------|
| API service (`/health`, `/analyze-ticket`) | [app/](app/) · live URL above |
| README (setup, AI usage, safety, models, limits) | this file |
| Dependency file | [requirements.txt](requirements.txt) |
| Sample output from a public case | [sample_output.json](sample_output.json) |
| `.env.example` | [.env.example](.env.example) |
| Dockerfile (fallback path) | [Dockerfile](Dockerfile) |
| Runbook | §3 above |
