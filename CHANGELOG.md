# Changelog

## 0.2.0-rc.3

- Added a three-lane Verified Model Swap spread sourced from Receipt Gate's
  bounded, independently graded projection.
- Kept DSM display-only: it performs no lane comparison, signature
  verification, scoring, or receiver authorization.
- Shows the commitments preserved by the causal capsule, the negative-history
  decisions omitted by the ordinary summary, and the records recovered from
  the authenticated archive.
- Changed the bundled API server to bind to loopback by default.
- Denies remote state-changing requests unless the operator explicitly
  configures `DSM_MUTATION_TOKEN` and the peer presents the matching bearer
  token.
- Replaced wildcard credential CORS with explicit local origins and disabled
  forwarded client-header handling in the bundled Uvicorn entry point.
- Added stdlib authorization controls, projection-integrity checks, DOM-sink
  checks, and a real FastAPI middleware smoke test for CI.

## 0.2.0-rc.2

- Restored the FastAPI source after rich-text transport corruption.
- Added Python compile/runtime checks and a canonical project receipt.
- Reworked the public site as an editorial observatory and interactive public
  instrument.
