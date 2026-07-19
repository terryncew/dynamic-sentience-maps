"""Stdlib-only integrity checks for source transport, receipts, and DOM sinks."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "server main.py", ROOT / "olp_client.py", ROOT / "wire_openline_demo.py"]

for source in SOURCES:
    text = source.read_text(encoding="utf-8")
    ast.parse(text, filename=str(source))
    compile(text, str(source), "exec")

server_text = SOURCES[0].read_text(encoding="utf-8")
for corrupted_token in ("“", "”", "**name**", "**init**", "**main**", "```"):
    if corrupted_token in server_text:
        raise AssertionError(f"rich-text transport token remains in server source: {corrupted_token}")

sys.path.insert(0, str(ROOT))
from olp_client import resolve_receipt_path  # noqa: E402

canonical = ROOT / "docs" / "receipt.latest.json"
if resolve_receipt_path().resolve() != canonical.resolve():
    raise AssertionError("default receipt path is not anchored to the project root")
if (ROOT / "docs" / "docs" / "receipt.latest.json").exists():
    raise AssertionError("nested duplicate receipt exists")

receipt = json.loads(canonical.read_text(encoding="utf-8"))
if "Dynamic Sentience Maps" not in receipt.get("claim", ""):
    raise AssertionError("canonical receipt is not project-specific")
if "SPY" in json.dumps(receipt):
    raise AssertionError("stock-template content remains in the canonical receipt")
if receipt.get("model") != "dynamic-sentience-maps/public-instrument":
    raise AssertionError("canonical receipt model is incorrect")
if receipt.get("attrs", {}).get("candidate") != "0.2.0-rc.2":
    raise AssertionError("canonical receipt candidate is stale")
if not receipt.get("generated_at") or not receipt.get("freshness", {}).get("expires_on_source_change"):
    raise AssertionError("canonical receipt does not expose its freshness boundary")

instrument = (ROOT / "docs" / "instrument.js").read_text(encoding="utf-8")
if ".innerHTML" in instrument:
    raise AssertionError("instrument contains an innerHTML assignment or access")

print("PASS Python AST/compile, transport tokens, canonical receipt, and DOM-sink checks")
