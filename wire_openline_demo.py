# wire_openline_demo.py
from __future__ import annotations
from olp_client import build_frame, post_frame, build_receipt, write_receipt_file

def main():
    claim = "SPY likely up tomorrow"
    delta = 0.028

    # Try to POST a Frame to OpenLine (optional)
    ok_post = False
    try:
        res = post_frame(build_frame(claim=claim, delta_scale=delta))
        ok_post = bool(res and res.get("ok"))
        print("[post]", res)
    except Exception as e:
        print("[post] skipped/failed:", e)

    # Always write the receipt (for GitHub Pages)
    receipt = build_receipt(
        claim=claim,
        because=["Curve emergence stayed within coherence band", "30d minute context"],
        but=[f"Scale drift Δ_scale = {delta:.3f} (min↔hour)"],
        so=("Within 3% tolerance — recheck at close" if delta <= 0.03 else
            "Above 3% — needs explanation"),
        delta_scale=delta,
    )
    path = write_receipt_file(receipt)
    print("[ok] wrote", path, "(posted:", ok_post, ")")

if __name__ == "__main__":
    main()
