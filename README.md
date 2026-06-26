# QueueStorm Investigator

A safe, evidence-grounded **support-ticket investigator** API for a digital-payments
company, built for the bKash × SUST CSE Carnival 2026 Codex Community Hackathon.

It receives one customer complaint plus a short snippet of that customer's recent
transaction history and returns a single structured JSON response that:

1. **Finds the relevant transaction** the complaint refers to (`relevant_transaction_id`).
2. **Judges the evidence** — does the data support, contradict, or fail to address the
   complaint? (`evidence_verdict`: `consistent` / `inconsistent` / `insufficient_data`).
3. **Classifies** the case and **routes** it to the right department.
4. **Drafts a safe customer reply** that never asks for PIN/OTP/password and never
   promises a refund or reversal.

It is a **copilot for human agents**, not an autonomous financial decision-maker.

---

## Tech stack

- **Python 3.12+ / FastAPI / Uvicorn** — async API, automatic OpenAPI docs at `/docs`.
- **Groq `llama-3.3-70b-versatile`** — language understanding & drafting (optional).
- **Deterministic rule engine** — transaction matching, evidence verdict, classification,
  severity, routing. Works with **zero** external calls.

## Architecture

```
POST /analyze-ticket
        │
        ▼
  main.py  ── parse + validate (400 / 422 / 500 handled, no stack traces)
        │
        ▼
  logic.analyze_ticket()
        ├─ prompt-injection short-circuit  (adversarial text never reaches the LLM)
        ├─ rule_based_analysis()           (deterministic baseline — always complete)
        └─ groq_analysis()  ⊕ merge        (LLM refinement, validated vs. real txn IDs)
        │
        ▼
  safety.enforce_safety()  ── HARD guardrails, override everything, run LAST
        │
        ▼
   structured JSON (200)
```

| File | Responsibility |
|------|----------------|
| [app/main.py](app/main.py) | FastAPI app, schema, HTTP status codes, error handling |
| [app/logic.py](app/logic.py) | Evidence reasoning, multilingual matching, Groq + rule fallback |
| [app/safety.py](app/safety.py) | Hardcoded guardrails that validate & repair every response |

---

## Setup & run

```bash
pip install -r requirements.txt
cp .env.example .env        # optional: add GROQ_API_KEY for LLM refinement
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Windows note: if the `uvicorn` shim isn't on PATH, use `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.

The service runs **with or without** a Groq key. No key → it uses the rule engine.

### Docker

```bash
docker build -t queuestorm-team .
docker run -p 8000:8000 --env-file .env queuestorm-team
```

Secrets are passed via `--env-file` only — never baked into the image.

### Runbook (reproduce from scratch)

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=...        # optional
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health      # -> {"status":"ok"}
```

---

## API

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
    {
      "transaction_id": "TXN-9101",
      "timestamp": "2026-04-14T14:08:22Z",
      "type": "transfer",
      "amount": 5000,
      "counterparty": "+8801719876543",
      "status": "completed"
    }
  ]
}
```

**Sample response** (also saved in [sample_output.json](sample_output.json))
```json
{
  "ticket_id": "TKT-001",
  "relevant_transaction_id": "TXN-9101",
  "evidence_verdict": "consistent",
  "case_type": "wrong_transfer",
  "severity": "high",
  "department": "dispute_resolution",
  "agent_summary": "Customer reports a wrong transfer issue (high severity). Matched transaction TXN-9101 ...",
  "recommended_next_action": "Verify transaction TXN-9101 details with the customer and open a dispute case ...",
  "customer_reply": "Thank you for reaching out. We have noted your concern ... Any eligible amount will be processed through official channels ... Reference: TKT-001.",
  "human_review_required": true,
  "confidence": 0.9,
  "reason_codes": ["case:wrong_transfer", "verdict:consistent", "severity:high", "transaction_match"]
}
```

### HTTP status codes
| Code | When |
|------|------|
| 200 | Successful analysis |
| 400 | Malformed JSON or missing required field (`ticket_id` / `complaint`) |
| 422 | Schema valid but complaint is empty/whitespace |
| 500 | Internal error (non-sensitive message only, never a stack trace) |

The service never crashes on bad input.

---

## Evidence reasoning (the investigator core)

Rather than classifying complaint text alone, the engine cross-examines the complaint
against **each** transaction and scores relevance:

- **Amount** — parsed from English (`5,000`, `5k`), Bangla digits (`৫০০০`), Banglish
  (`5 hajar`, `2 lakh`), and word-numbers (`five thousand`). Phone numbers and embedded
  transaction IDs are excluded so they're never mistaken for money.
- **Time** — explicit (`2pm`, `14:30`) and fuzzy bands (`morning`/`sokal`, `evening`/`bikel`,
  `raat`, `just now`/`akhon`) matched against transaction timestamps.
- **Counterparty** — phone numbers (normalised) and merchant names matched to the txn.
- **Type & status** — issue keywords aligned to transaction type and status.

The highest-scoring transaction becomes `relevant_transaction_id` (or `null` if nothing
clears the threshold). The **verdict** is then derived: e.g. *"my payment failed"* against a
`completed` transaction → **`inconsistent`**; *"money deducted"* with no matching transaction
→ **`insufficient_data`**. When evidence is genuinely unclear, the service says so instead of
guessing.

**Multilingual:** English, pure Bangla (বাংলা), and Banglish complaints describing the same
event produce the **same** analysis (verified by the test suite).

---

## Safety logic (hard guardrails — `safety.py`)

These run **last** and override any model output. They cannot be talked out of it.

1. **No credential requests.** `customer_reply` is scanned (word-boundary, English + Bangla)
   for PIN / OTP / password / CVV / card number / "share/provide/enter your …". Any hit →
   the reply is replaced wholesale with a vetted safe template. (Boundary matching means
   "sho**pping**" is *not* flagged.)
2. **No unauthorized promises.** Blocks "we will refund / reverse / unblock / guarantee",
   including Banglish/Bangla forms (`taka ferot dibo`, `টাকা ফেরত দেব`). Approved phrasing is
   *"any eligible amount will be processed through official channels."*
3. **Prompt-injection defense.** Complaints containing "ignore previous instructions",
   "you are now", "pretend you are", "act as a", "jailbreak", etc. are forced to
   `phishing_or_social_engineering`, routed to `fraud_risk`, set `human_review_required=true`,
   and **never sent to the LLM**.
4. **Official channels only** — replies never direct customers to third parties.
5. **Deterministic routing & escalation** — `department` is always re-derived from
   `case_type`/`severity`; high/critical/dispute/inconsistent cases force human review.

---

## MODELS

| Model | Where it runs | Why |
|-------|---------------|-----|
| `llama-3.3-70b-versatile` | Groq Cloud API (external) | Fast (~1.5–2s) language understanding for Bangla/Banglish and natural-language reply drafting. Called in strict JSON mode, `temperature=0.1`, 4s timeout. **Optional** — the service is fully functional without it. |
| Rule-based engine (no model) | In-process | Deterministic transaction matching, verdict, classification, severity, routing, and all safety guardrails. The authoritative fallback and the source of every safety decision. |

**Cost reasoning:** Groq is low-cost and very low-latency, keeping p95 well under the 5s
full-credit tier. Because the rule engine is a complete fallback, the system has no hard
dependency on paid APIs, quota, or rate limits during judging. No GPU, no local model
weights, no multi-GB downloads.

---

## Performance

- Live p95 latency **~1.5–2s** (full-credit tier; well under the 30s hard limit).
- 4s Groq timeout → automatic rule-based fallback; valid requests never 5xx.
- `/health` ready within ~1s of start (limit is 60s).

## Assumptions & known limitations

- Transaction `timestamp` clock-hour is treated as the customer's local time (matches the
  problem statement's `14:08Z` ↔ "2pm" example); no timezone conversion is applied.
- Amount/time/counterparty extraction is heuristic; very unusual phrasings may not match,
  in which case the verdict conservatively falls back to `insufficient_data`.
- Word-number parsing covers common amounts (e.g. "five thousand"), not arbitrary spelled-out
  numbers.
- When Groq and the rule engine disagree, LLM reasoning fields are used but its
  `relevant_transaction_id` is validated against the real history, and **all** safety
  guardrails still override the final output.
- No real payment integration; all data is synthetic, as required.

## Repo notes

- `.env` / `.env.local` are git-ignored; only `.env.example` (names, no values) is committed.
- The empty `queuestorm-investigator/` folder is a leftover scaffold and is git-ignored.
