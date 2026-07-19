import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { API_CONTRACTS, LINKS, METRIC_DEFINITIONS, NODES, REGIONS, STATES } from "../docs/demo-data.js";
import { averageRegionStress, edgePath } from "../docs/instrument.js";
import { projectionToView, renderVerifiedModelSwap } from "../docs/verified-model-swap.js";

const rootFile = (path) => new URL(`../${path}`, import.meta.url);
const read = async (path) => readFile(rootFile(path), "utf8");
const sha256 = async (path) => createHash("sha256").update(await readFile(rootFile(path))).digest("hex");

test("the demonstration graph is bounded and internally connected", () => {
  assert.equal(NODES.length, 11);
  assert.equal(REGIONS.length, 4);
  assert.equal(new Set(NODES.map(({ id }) => id)).size, NODES.length);
  assert.equal(new Set(REGIONS.map(({ id }) => id)).size, REGIONS.length);
  const nodeIds = new Set(NODES.map(({ id }) => id));
  const regionIds = new Set(REGIONS.map(({ id }) => id));
  for (const node of NODES) {
    assert.ok(regionIds.has(node.region), `${node.id} has a known region`);
    assert.ok(node.description && node.boundary && node.intervention);
    assert.ok(Array.isArray(node.evidence) && node.evidence.length > 0);
  }
  for (const link of LINKS) {
    assert.ok(nodeIds.has(link.source), `${link.source} exists`);
    assert.ok(nodeIds.has(link.target), `${link.target} exists`);
    assert.ok(link.relation);
  }
  assert.ok(LINKS.some(({ relation }) => relation === "contradicts"));
});

test("every observation supplies complete layouts, stress, and bounded readings", () => {
  const nodeIds = NODES.map(({ id }) => id).sort();
  for (const state of STATES) {
    assert.deepEqual(Object.keys(state.positions).sort(), nodeIds);
    assert.deepEqual(Object.keys(state.mobilePositions).sort(), nodeIds);
    for (const value of Object.values(state.metrics)) {
      assert.ok(value >= 0 && value <= 1, `${state.id} metric is bounded`);
    }
    for (const node of NODES) {
      assert.ok(node.stress[state.id] >= 0 && node.stress[state.id] <= 1);
      assert.ok(node.recent[state.id]);
    }
  }
  assert.equal(METRIC_DEFINITIONS.length, 4);
});

test("geometry helpers are deterministic and region stress stays bounded", () => {
  const support = edgePath([0, 0], [100, 100], "supports");
  const contradiction = edgePath([0, 0], [100, 100], "contradicts");
  assert.match(support, /^M 0 0 C /);
  assert.match(contradiction, /^M 0 0 C /);
  assert.notEqual(support, contradiction);
  for (const region of REGIONS) {
    for (const state of STATES) {
      const stress = averageRegionStress(region.id, state.id);
      assert.ok(stress >= 0 && stress <= 1);
    }
  }
});

test("the editorial surface exposes the required instrument and claim limits", async () => {
  const [html, css, instrument] = await Promise.all([
    read("docs/index.html"), read("docs/styles.css"), read("docs/instrument.js"),
  ]);
  for (const id of ["instrument", "measures", "model-swap", "verified-model-swap", "method", "architecture", "coherence-map", "detail-panel"]) {
    assert.match(html, new RegExp(`id=[\"']${id}[\"']`));
  }
  assert.match(html, /Use Tab to move through regions and nodes/);
  assert.match(html, /demonstration proxies, not universal scientific thresholds/i);
  assert.match(html, /κ/);
  assert.match(html, /ε/);
  assert.match(html, /Φ\*/);
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.doesNotMatch(css, /linear-gradient|radial-gradient|glassmorphism|box-shadow:/i);
  assert.doesNotMatch(instrument, /\.innerHTML\s*=/, "selection content is rendered with DOM-safe text nodes");
});

test("DSM renders the independently produced model-swap projection without grading it", async () => {
  const projection = JSON.parse(await read("docs/verified_model_swap.latest.json"));
  const view = projectionToView(projection);
  assert.equal(projection.display_only, true);
  assert.equal(projection.proof_card.authority.dsm_grading_allowed, false);
  assert.equal(view.sourceModel, "fixture/source-model");
  assert.equal(view.targetModel, "fixture/target-model");
  assert.equal(view.providerExecutionAttested, false);
  assert.equal(view.gateVerdict, "VERIFIED");
  assert.equal(view.gateDecision, "COMMIT");
  assert.equal(view.independentResult, "MATCH");
  assert.deepEqual(view.lanes.map(({ declaredResult }) => declaredResult), ["MATCH", "MISMATCH", "MATCH"]);
  assert.deepEqual(view.lanes.map(({ decisionCount }) => decisionCount), [16, 9, 16]);
  assert.deepEqual(view.lanes.map(({ mismatchCount }) => mismatchCount), [0, 7, 0]);
  assert.equal(view.commitments.length, 7);
  assert.equal(view.summaryLost.length, 7);
  assert.equal(view.archiveReturned.length, 7);
  assert.equal(view.archiveReceiptCount, 65);
});

test("the model-swap surface has no verifier, scoring, or unsafe HTML sink", async () => {
  const source = await read("docs/verified-model-swap.js");
  assert.doesNotMatch(source, /\.innerHTML\b/);
  assert.doesNotMatch(
    source,
    /reference_receipt_gate_projection|receipt_gate_projection|evaluate_request|verify_decision_receipt|crypto\.subtle|createHash/i,
  );
  const projection = JSON.parse(await read("docs/verified_model_swap.latest.json"));
  assert.throws(() => projectionToView({ ...projection, display_only: false }), /display-only/);
  assert.throws(() => projectionToView({
    ...projection,
    proof_card: {
      ...projection.proof_card,
      authority: { ...projection.proof_card.authority, dsm_grading_allowed: true },
    },
  }), /grading authority/);
});

test("the model-swap proof spread renders with DOM-safe text nodes", async () => {
  class FakeElement {
    constructor(tag) {
      this.tag = tag;
      this.children = [];
      this.attributes = {};
      this.dataset = {};
      this.textContent = "";
    }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = [...children]; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
  }
  const originalDocument = globalThis.document;
  globalThis.document = { createElement: (tag) => new FakeElement(tag) };
  try {
    const container = new FakeElement("div");
    const projection = JSON.parse(await read("docs/verified_model_swap.latest.json"));
    renderVerifiedModelSwap(container, projectionToView(projection));
    const collect = (node) => [
      node.textContent,
      ...node.children.flatMap((child) => collect(child)),
    ].join(" ");
    const rendered = collect(container);
    assert.match(rendered, /Change the model without losing the agent/);
    assert.match(rendered, /VERIFIED → COMMIT/);
    assert.match(rendered, /Ordinary summary/);
    assert.match(rendered, /7 negative-history decisions/);
    assert.match(rendered, /DSM did not compare the lanes/);
  } finally {
    if (originalDocument === undefined) delete globalThis.document;
    else globalThis.document = originalDocument;
  }
});

test("the existing API contract remains declared", async () => {
  const backend = await read("server main.py");
  for (const contract of API_CONTRACTS) {
    const route = contract.split(" ")[1];
    assert.ok(backend.includes(route), `${contract} remains in the backend prototype`);
  }
  assert.match(backend, /version="0\.1\.0"/);
  assert.match(backend, /DSM_MUTATION_TOKEN/);
  assert.match(backend, /"127\.0\.0\.1"/);
  assert.match(backend, /proxy_headers=False/);
  assert.doesNotMatch(backend, /allow_origins=\["\*"\]/);
});

test("release automation executes Python rather than treating hashes as correctness", async () => {
  const [releaseCheck, workflow] = await Promise.all([
    read("scripts/release-check.mjs"), read(".github/workflows/test.yml"),
  ]);
  assert.match(releaseCheck, /py_compile/);
  assert.match(releaseCheck, /server_runtime_smoke\.py/);
  assert.match(workflow, /setup-python/);
  assert.match(workflow, /py_compile/);
  assert.match(workflow, /server_runtime_smoke\.py/);
});
