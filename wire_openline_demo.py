from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from olp_client import build_frame, post_frame, build_receipt, write_receipt_file

def main(*, post: bool = False):
    report_path = Path(__file__).resolve().with_name("RELEASE_VERIFICATION.json")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        checks = []
    clear = (
        report.get("candidate") == "0.2.0-rc.3"
        and report.get("disposition") == "clear"
        and isinstance(checks, list)
        and len(checks) == 8
        and all(isinstance(item, dict) and item.get("status") == "pass" for item in checks)
    )
    claim = (
        "Dynamic Sentience Maps v0.2.0rc3 passed bounded release verification"
        if clear
        else "Dynamic Sentience Maps v0.2.0rc3 remains on hold pending bounded release verification"
    )
    delta = 0.0

    ok_post = False
    if post:
        try:
            res = post_frame(build_frame(claim=claim, delta_scale=delta))
            ok_post = bool(res and res.get("ok"))
            print("[post]", res)
        except Exception as e:
            print("[post] failed:", e)

    passed_checks = [
        str(item.get("name"))
        for item in checks
        if item.get("status") == "pass"
    ]
    failed_checks = [
        str(item.get("name"))
        for item in checks
        if item.get("status") != "pass"
    ]
    receipt = build_receipt(
        claim=claim,
        because=passed_checks or ["No current release report is available"],
        but=[
            "Public readings remain disclosed demonstration proxies",
            "No private bridge kernel or universal scientific threshold is certified",
            "The model identifiers are fixture declarations; commercial provider execution is not attested",
            *(["Pending or failed checks: " + ", ".join(failed_checks)] if failed_checks else []),
        ],
        so=(
            "Clear for the rc3 public-instrument candidate within the declared claim boundary"
            if clear
            else "Hold until every declared release check passes in the current source snapshot"
        ),
        delta_scale=delta,
        model="dynamic-sentience-maps/public-instrument",
        attrs={
            "cadence": "release",
            "candidate": "0.2.0-rc.3",
            "disposition": "clear_with_boundary" if clear else "hold",
            "evidence": [
                "RELEASE_VERIFICATION.json",
                "tests/python_integrity.py",
                "tests/server_runtime_smoke.py",
                "tests/instrument.test.mjs",
            ],
        },
    )
    receipt["generated_at"] = datetime.now(timezone.utc).isoformat()
    receipt["freshness"] = {
        "scope": "this release-candidate source snapshot",
        "expires_on_source_change": True,
    }
    path = write_receipt_file(receipt)
    print("[ok] wrote", path, "(posted:", ok_post, ")")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the canonical Dynamic Sentience Maps release receipt.")
    parser.add_argument("--post", action="store_true", help="Also post the frame to the configured OpenLine endpoint.")
    args = parser.parse_args()
    main(post=args.post)
