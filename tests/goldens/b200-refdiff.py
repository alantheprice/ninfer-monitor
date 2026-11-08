#!/usr/bin/env python3
"""B200 reference-diff runner: replay the golden pack against any engine endpoint.

Usage: python3 b200-refdiff.py http://127.0.0.1:8006 [--tokens N]
Compares greedy completions (reasoning_effort=none) to tests/goldens/b200-refdiff-pack.json.
Scoring: exact-match rate, per-case token-level prefix match, and a soft
first-divergence-distance histogram. Emulation changes accumulation order, so
expect >=99% prefix-12 match rather than byte parity.
Protocol notes (validated on ws6000 2026-11-07):
- `reasoning_effort:"none"` is REQUIRED — the default thinking mode makes content empty at
  max_tokens=48 (reasoning eats the budget) and is itself nondeterministic across runs.
- Cases run SERIALLY with 300ms spacing. Under concurrency or back-to-back replay,
  MTP speculative-decode acceptance races make greedy output nondeterministic
  (observed: 3 distinct outputs for the identical prompt across 6 serial repeats —
  acceptance 57-70% during the window). Compare runs, not absolutes: score a
  replay against the pack for DIVERGENCE vs a replay of the same pack on the same
  engine, and treat the pack as a regression baseline only after confirming the
  engine can reproduce it twice in a row.
"""
import json, sys, time, urllib.request

GOLD = "tests/goldens/b200-refdiff-pack.json"

def gen(base, prompt, max_tokens):
    body = json.dumps({"model": "qwen3.8-flash-next",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0,
                       "reasoning_effort": "none"}).encode()
    r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", body,
                                                     {"Content-Type": "application/json"}), timeout=120)
    j = json.loads(r.read())
    return j["choices"][0]["message"]["content"]

def toks(s):
    return s.split()  # word-level proxy; enough to locate divergence points

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8006"
    pack = json.load(open(GOLD))
    exact = prefix12 = 0
    dists = []
    for case in pack["cases"]:
        got = gen(base, case["prompt"], 48)
        a, b = toks(case["completion"]), toks(got)
        n = 0
        for x, y in zip(a, b):
            if x == y: n += 1
            else: break
        dists.append((case["id"], n, len(a)))
        if got == case["completion"]: exact += 1
        if n >= 12: prefix12 += 1
        print(f"{case['id']}: prefix {n}/{len(a)} {'EXACT' if got == case['completion'] else ''}")
        time.sleep(0.3)
    print(f"\n== {len(pack['cases'])} cases: exact={exact} prefix12>={prefix12} "
          f"({100.0*prefix12/len(pack['cases']):.0f}%) median-prefix={sorted(d for _,d,_ in dists)[len(dists)//2]}")
    print("gate: PASS if prefix12 >= 99% (emulation contract), FAIL below -> stop and diff kernels")

if __name__ == "__main__":
    main()
