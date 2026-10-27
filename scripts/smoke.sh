#!/usr/bin/env bash
# Runs one real task against a running policyserve server and checks the headline numbers.
#   scripts/smoke.sh http://127.0.0.1:8000
set -euo pipefail
base="${1:-http://127.0.0.1:8000}"

for _ in $(seq 1 30); do curl -fs "$base/healthz" >/dev/null && break; sleep 1; done
curl -fs "$base/healthz" >/dev/null || { echo "server did not come up"; exit 1; }
curl -fs "$base/" | grep -q "<title>policyserve</title>" || { echo "viewer page missing"; exit 1; }

id=$(curl -fs -X POST "$base/tasks" -H 'Content-Type: application/json' \
  -d '{"instruction":"pick up the red block and place it on the green platform"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["task_id"])')
echo "task $id"

for _ in $(seq 1 90); do
  status=$(curl -fs "$base/tasks/$id" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')
  [ "$status" != "queued" ] && [ "$status" != "running" ] && break
  sleep 1
done
echo "status: $status"
[ "$status" = "succeeded" ] || { curl -s "$base/tasks/$id"; exit 1; }

curl -fs "$base/metrics" | python3 -c '
import sys, json
m = json.load(sys.stdin)["system1"]
print("System 1: %.1f Hz, tick p50/p99 %.1f/%.1f ms, deadline misses %d" % (m["observed_hz"], m["tick_interval_ms"]["p50"], m["tick_interval_ms"]["p99"], m["deadline_misses_total"]))
assert m["deadline_misses_total"] == 0, "deadline misses in an undisturbed run"
assert m["observed_hz"] >= 95, "System 1 is not holding its rate"
'
echo "smoke test passed"
