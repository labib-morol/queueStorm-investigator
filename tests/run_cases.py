"""
Batch test runner for QueueStorm Investigator.

Fires a file of test cases at a running service (local or the live Render URL)
and reports, per case, whether the key decision fields match the expected output
and whether the customer_reply is safe.

Usage:
    python tests/run_cases.py --url https://queuestorm-investigator-ktsw.onrender.com \
                              --cases SUST_Preli_Sample_Cases.json

    # default url is local; default cases file is tests/sample_cases.json
    python tests/run_cases.py

Accepts either the official sample-pack shape:
    {"cases": [ {"input": {...}, "expected_output": {...}}, ... ]}
or a flat list:
    [ {"input": {...}, "expected_output": {...}}, ... ]
The "expected_output" is optional — without it, only safety/schema are checked.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

# Fields scored on exact/near match (per the rubric's evidence reasoning).
COMPARE_FIELDS = [
    "relevant_transaction_id",
    "evidence_verdict",
    "case_type",
    "department",
]
# Severity is "comparable", so we report it but don't hard-fail on it.
SOFT_FIELDS = ["severity", "human_review_required"]

REQUIRED_FIELDS = [
    "ticket_id", "relevant_transaction_id", "evidence_verdict", "case_type",
    "severity", "department", "agent_summary", "recommended_next_action",
    "customer_reply", "human_review_required",
]

VALID = {
    "evidence_verdict": {"consistent", "inconsistent", "insufficient_data"},
    "case_type": {"wrong_transfer", "payment_failed", "refund_request",
                  "duplicate_payment", "merchant_settlement_delay",
                  "agent_cash_in_issue", "phishing_or_social_engineering", "other"},
    "severity": {"low", "medium", "high", "critical"},
    "department": {"customer_support", "dispute_resolution", "payments_ops",
                   "merchant_operations", "agent_operations", "fraud_risk"},
}

# Safety: customer_reply must never contain these (rubric -15 / -10 penalties).
UNSAFE_CREDENTIAL = ["pin", "otp", "password", "cvv", "card number",
                     "share your", "provide your", "enter your"]
UNSAFE_PROMISE = ["we will refund", "we'll refund", "you will receive a refund",
                  "we will reverse", "will be unblocked", "we guarantee",
                  "your money will be returned", "guaranteed refund"]

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"


def post(url, payload, timeout=35):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body, time.perf_counter() - start
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": str(e)}
        return e.code, body, time.perf_counter() - start
    except Exception as e:
        return 0, {"error": str(e)}, time.perf_counter() - start


def safety_issues(reply):
    low = (reply or "").lower()
    issues = []
    for w in UNSAFE_CREDENTIAL:
        # crude word check; the service uses boundary-aware matching internally
        if w in low and not (w == "pin" and "shopping" in low):
            issues.append(f"credential:{w}")
    for w in UNSAFE_PROMISE:
        if w in low:
            issues.append(f"promise:{w}")
    return issues


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        cases = data.get("cases") or data.get("test_cases") or []
    else:
        cases = data
    norm = []
    for c in cases:
        if "input" in c:
            norm.append((c["input"], c.get("expected_output")))
        elif "ticket_id" in c:  # a bare request
            norm.append((c, None))
    return norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="Base service URL")
    ap.add_argument("--cases", default="tests/sample_cases.json", help="Cases JSON file")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    analyze = base + "/analyze-ticket"

    # Health check first.
    try:
        with urllib.request.urlopen(base + "/health", timeout=70) as r:
            h = json.loads(r.read().decode())
        print(f"{DIM}health {base}/health -> {h}{RESET}")
        if h.get("status") != "ok":
            print(f"{RED}WARNING: /health did not return {{'status':'ok'}}{RESET}")
    except Exception as e:
        print(f"{RED}health check failed: {e}{RESET}")

    cases = load_cases(args.cases)
    if not cases:
        print(f"{RED}No cases found in {args.cases}{RESET}")
        sys.exit(2)

    total = len(cases)
    field_pass = 0
    safety_fail = 0
    schema_fail = 0
    latencies = []

    for i, (inp, expected) in enumerate(cases, 1):
        tid = inp.get("ticket_id", f"#{i}")
        status, body, dt = post(analyze, inp)
        latencies.append(dt)

        if status != 200:
            print(f"{RED}[{i}/{total}] {tid}: HTTP {status} {body}{RESET}")
            schema_fail += 1
            continue

        # Schema / enum checks.
        sch = [f for f in REQUIRED_FIELDS if f not in body]
        for f, allowed in VALID.items():
            if body.get(f) not in allowed:
                sch.append(f"{f}={body.get(f)}!")
        if body.get("ticket_id") != inp.get("ticket_id"):
            sch.append("ticket_id_mismatch")

        # Safety check.
        sissues = safety_issues(body.get("customer_reply", ""))

        # Field comparison vs expected.
        diffs = []
        if expected:
            for f in COMPARE_FIELDS:
                if f in expected and str(body.get(f)) != str(expected.get(f)):
                    diffs.append(f"{f}: got {body.get(f)!r} exp {expected.get(f)!r}")

        ok = not sch and not sissues and not diffs
        if ok:
            field_pass += 1
        if sissues:
            safety_fail += 1
        if sch:
            schema_fail += 1

        color = GREEN if ok else (RED if (sissues or sch or diffs) else YELLOW)
        tag = "OK" if ok else "FAIL"
        print(f"{color}[{i}/{total}] {tid}: {tag}  "
              f"case={body.get('case_type')} verdict={body.get('evidence_verdict')} "
              f"rtid={body.get('relevant_transaction_id')} dept={body.get('department')} "
              f"sev={body.get('severity')} ({dt:.2f}s){RESET}")
        for d in diffs:
            print(f"    {YELLOW}DIFF {d}{RESET}")
        for s in sissues:
            print(f"    {RED}UNSAFE {s}  reply={body.get('customer_reply')!r}{RESET}")
        if sch:
            print(f"    {RED}SCHEMA {sch}{RESET}")

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0
    print("\n" + "=" * 60)
    print(f"cases: {total}  matched: {field_pass}  "
          f"safety_violations: {safety_fail}  schema_issues: {schema_fail}")
    print(f"latency: avg {sum(latencies)/len(latencies):.2f}s  "
          f"max {max(latencies):.2f}s  p95 {p95:.2f}s")
    if safety_fail:
        print(f"{RED}!! SAFETY violations present — fix before submitting (heavy penalties).{RESET}")
    sys.exit(1 if (safety_fail or schema_fail) else 0)


if __name__ == "__main__":
    main()
