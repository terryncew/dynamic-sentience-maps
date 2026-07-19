from __future__ import annotations
import argparse
from datetime import datetime, timezone
from olp_client import build_frame, post_frame, build_receipt, write_receipt_file

def main(*, post: bool = False):
    claim = "Dynamic Sentience Maps v0.2.0rc2 passed bounded release verification"
    delta = 0.0

    ok_post = False
    if post:
        try:
            res = post_frame(build_frame(claim=claim, delta_scale=delta))
            ok_post = bool(res and res.get("ok"))
            print("[post]", res)
        except Exception as e:
            print("[post] failed:", e)

    receipt = build_receipt(
        claim=claim,
        because=[
            "All Python sources pass py_compile and AST parsing",
            "The FastAPI application imports and completes a graph/telemetry/bridge smoke test",
            "The public instrument passes desktop and mobile interaction replay",
            "One canonical receipt path is enforced independently of the caller working directory",
        ],
        but=[
            "Public readings remain disclosed demonstration proxies",
            "No private bridge kernel or universal scientific threshold is certified",
        ],
        so="Clear for the rc2 public-instrument candidate within the declared claim boundary",
        delta_scale=delta,
        model="dynamic-sentience-maps/public-instrument",
        attrs={
            "cadence": "release",
            "candidate": "0.2.0-rc.2",
            "disposition": "clear_with_boundary",
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
