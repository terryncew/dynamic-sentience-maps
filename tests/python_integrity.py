"""Stdlib-only integrity checks for source transport, receipts, and DOM sinks."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "server main.py",
    ROOT / "dsm_security.py",
    ROOT / "olp_client.py",
    ROOT / "wire_openline_demo.py",
]

for source in SOURCES:
    text = source.read_text(encoding="utf-8")
    ast.parse(text, filename=str(source))
    compile(text, str(source), "exec")

server_text = SOURCES[0].read_text(encoding="utf-8")
for corrupted_token in ("“", "”", "**name**", "**init**", "**main**", "```"):
    if corrupted_token in server_text:
        raise AssertionError(f"rich-text transport token remains in server source: {corrupted_token}")

sys.path.insert(0, str(ROOT))
from dsm_security import allowed_origins, mutation_authorization  # noqa: E402
from olp_client import resolve_receipt_path  # noqa: E402

if allowed_origins(None) != ["http://127.0.0.1:8000", "http://localhost:8000"]:
    raise AssertionError("default browser origins are not local and explicit")
try:
    allowed_origins("*")
except ValueError:
    pass
else:
    raise AssertionError("wildcard credential origin was accepted")
for peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
    if mutation_authorization(peer, None, None) != (True, 200, "loopback_peer"):
        raise AssertionError(f"loopback peer was denied: {peer}")
if mutation_authorization("203.0.113.7", None, None) != (False, 403, "remote_mutation_disabled"):
    raise AssertionError("remote mutation is not disabled by default")
if mutation_authorization("203.0.113.7", "Bearer wrong", "secret") != (False, 401, "bearer_token_invalid"):
    raise AssertionError("incorrect bearer token was accepted")
if mutation_authorization("203.0.113.7", "Bearer secret", "secret") != (True, 200, "bearer_token_valid"):
    raise AssertionError("valid configured bearer token was rejected")

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
if receipt.get("attrs", {}).get("candidate") != "0.2.0-rc.3":
    raise AssertionError("canonical receipt candidate is stale")
if not receipt.get("generated_at") or not receipt.get("freshness", {}).get("expires_on_source_change"):
    raise AssertionError("canonical receipt does not expose its freshness boundary")

for script_name in ("instrument.js", "verified-model-swap.js"):
    script = (ROOT / "docs" / script_name).read_text(encoding="utf-8")
    if ".innerHTML" in script:
        raise AssertionError(f"{script_name} contains an innerHTML assignment or access")

projection = json.loads((ROOT / "docs" / "verified_model_swap.latest.json").read_text(encoding="utf-8"))
if projection.get("schema") != "openline.verified-model-swap.dsm-projection.v1":
    raise AssertionError("model-swap display projection schema is unsupported")
if projection.get("display_only") is not True:
    raise AssertionError("model-swap projection is not display-only")
proof = projection.get("proof_card", {})
if proof.get("authority", {}).get("dsm_grading_allowed") is not False:
    raise AssertionError("model-swap projection grants DSM grading authority")
if proof.get("models", {}).get("provider_execution_attested") is not False:
    raise AssertionError("fixture unexpectedly claims live provider execution")
if projection.get("gate_decision", {}).get("decision") != "COMMIT":
    raise AssertionError("display projection is not the declared committed fixture")
proof_bytes = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode("utf-8")
if hashlib.sha256(proof_bytes).hexdigest() != projection.get("integrity", {}).get("proof_card_file_sha256"):
    raise AssertionError("embedded proof card does not match its producer-supplied file hash")
if projection.get("gate_decision", {}).get("payload_hash") != projection.get("integrity", {}).get("decision_receipt_payload_hash"):
    raise AssertionError("displayed Gate decision pointer is inconsistent")

print("PASS source transport, canonical receipt, display projection integrity, and DOM sinks")
