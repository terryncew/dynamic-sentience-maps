# Dynamic Sentience Maps

An open-core telemetry and graph-processing instrument for visualizing coherence, stress, entropy leak, memory, and boundaries in living knowledge structures.

**Public instrument:** https://terryncew.github.io/dynamic-sentience-maps/

The website is designed as an editorial observatory rather than a conventional dashboard. Its first view is a readable scientific plate; the second exposes state changes and structure; the third exposes relations and evidence.

## What is included

- An interactive, keyboard-accessible coherence map with three bounded observations: baseline, pressure, and drift.
- Calm instrument readings for `κ`, `ε`, `I_c`, and `Φ*`.
- Progressive node and region inspection: meaning, support, contradiction, stress, boundary status, recent change, intervention, and evidence.
- A disclosed demonstration dataset with one stable spine, one pressure field, one drifting claim, and one unresolved contradiction.
- A display-only Verified Model Swap spread showing the full-history reference, an ordinary summary, and a verified causal capsule from Receipt Gate's independently graded fixture.
- A repaired FastAPI prototype with the original graph, telemetry, metric, intervention, import, and Bridge API routes.

The map uses shape, line style, text, and color together. Motion communicates state changes and respects `prefers-reduced-motion`.

## Run the public instrument

No build step or package installation is required.

```bash
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. The site is also ready for GitHub Pages with `docs/` as the publishing source.

## Run the API

The backend has an explicit dependency manifest and uses an isolated SQLite file by default.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python "server main.py"
```

The API and generated OpenAPI documentation are available at `http://localhost:8000` and `http://localhost:8000/docs`.

The server binds to `127.0.0.1` by default. Loopback peers may use state-changing routes without a token. To expose mutations beyond the local machine, set a strong `DSM_MUTATION_TOKEN`; remote callers must send it as `Authorization: Bearer ...`. `DSM_HOST` and comma-separated `DSM_ALLOWED_ORIGINS` are explicit deployment opt-ins. Wildcard credential origins are rejected, and the bundled server ignores forwarded client headers.

## Verified Model Swap surface

`docs/verified_model_swap.latest.json` is produced by OpenLine Receipt Gate's controlled three-lane fixture. It declares which commitments survived, which negative-history decisions an ordinary summary lost, what returned from the authenticated archive, and the receiver Gate disposition.

DSM only parses and renders that bounded projection. It does not compare the lanes, verify the receipt signatures, run Half-Life, or decide whether the swap may continue. The source and target model names are caller-declared fixture identifiers; the artifact does not attest that either commercial provider executed the trial. Verify the proof card, evidence, trust keys, and receiver policy in Receipt Gate before relying on the result.

## Verify the release

Install the Python requirements above, then use Node.js 20 or newer. There are no npm dependencies.

```bash
npm test
npm run check
```

`npm run check` parses and compiles every Python source, imports the FastAPI application, exercises SQLite graph and telemetry round-trips, tests the local/remote mutation boundary, checks the Bridge API, validates the JavaScript and DOM-safety boundary, checks the producer-supplied proof-card hash inside the display projection, enforces the single canonical receipt, and writes a bounded result to `RELEASE_VERIFICATION.json`.

## Architecture boundary

The open layer stores graph structure, telemetry, and observable estimators. A private kernel can return metrics and interventions through:

```text
POST /bridge/analyze
```

The prototype also declares:

```text
GET  /graph
POST /graph/nodes
POST /graph/links
POST /telemetry/events
GET  /telemetry/recent
GET  /metrics/current
GET  /interventions/suggestions
```

The repaired backend retains those route contracts. The release workflow runs `py_compile` and a runtime smoke test; byte equality is recorded as identity evidence, never treated as correctness.

## Evidence boundary

The included readings and graph are deterministic demonstration proxies. They make the interface and method inspectable; they do not establish universal thresholds, scientific validation, consciousness, sentience, or a diagnosis of any real system. No single reading is presented as a verdict.

The canonical project receipt is `docs/receipt.latest.json`. Its path resolves from the project root even when the generator is launched from another working directory. Verify evidence and receiver policy in addition to any signature.

## License

MIT © Terrynce White
