export const SEMANTIC_CASE_KEY = "same-word-different-rules";
export const SEMANTIC_CASE_PATH = "./demos/same-word-different-rules.json";

const SVG_NS = "http://www.w3.org/2000/svg";

export function resolveCaseFromSearch(search = "") {
  return new URLSearchParams(search).get("case");
}

export function stateById(fixture, stateId) {
  return fixture.states.find((state) => state.id === stateId) ?? null;
}

export function visibleNodeIds(fixture, stateId) {
  return [...(stateById(fixture, stateId)?.visible_nodes ?? [])];
}

export function activeEdgeIds(fixture, stateId) {
  return [...(stateById(fixture, stateId)?.active_edges ?? [])];
}

export function nextStateId(fixture, currentId, key) {
  const ids = fixture.states.map(({ id }) => id);
  const index = Math.max(0, ids.indexOf(currentId));
  if (key === "Home") return ids[0];
  if (key === "End") return ids.at(-1);
  if (key === "ArrowRight" || key === "ArrowDown") return ids[(index + 1) % ids.length];
  if (key === "ArrowLeft" || key === "ArrowUp") return ids[(index - 1 + ids.length) % ids.length];
  return currentId;
}

export function validateSemanticFixture(fixture) {
  if (fixture?.case_id !== SEMANTIC_CASE_KEY) throw new Error("Unsupported semantic case fixture");
  if (fixture.final_status !== "Unresolved semantic collision") throw new Error("Final status changed");
  if (fixture.states?.map(({ id }) => id).join(",") !== "baseline,pressure,drift") throw new Error("Case must have exactly three ordered states");
  const nodeIds = new Set(fixture.nodes.map(({ id }) => id));
  const edgeIds = new Set(fixture.edges.map(({ id }) => id));
  if (nodeIds.size !== fixture.nodes.length || edgeIds.size !== fixture.edges.length) throw new Error("Fixture IDs must be unique");
  for (const edge of fixture.edges) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) throw new Error(`Unknown edge endpoint: ${edge.id}`);
  }
  for (const state of fixture.states) {
    if (state.visible_nodes.some((id) => !nodeIds.has(id))) throw new Error(`Unknown node in ${state.id}`);
    if (state.active_edges.some((id) => !edgeIds.has(id))) throw new Error(`Unknown edge in ${state.id}`);
    for (const value of Object.values(state.metrics)) if (value < 0 || value > 1) throw new Error(`Unbounded metric in ${state.id}`);
  }
  return fixture;
}

function element(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

function svgElement(tag, attributes = {}, text) {
  const item = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) item.setAttribute(name, String(value));
  if (text !== undefined) item.textContent = text;
  return item;
}

function splitLabel(label, max = 20) {
  const words = label.replace(/[.?]$/, "").split(/\s+/);
  const lines = [];
  let current = "";
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > max && current) { lines.push(current); current = word; }
    else current = next;
  }
  if (current) lines.push(current);
  return lines.slice(0, 3);
}

function renderMap(svg, fixture, isMobile) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", isMobile ? "0 0 420 980" : "0 0 1000 650");
  svg.append(svgElement("title", {}, "Semantic collision map"));
  svg.append(svgElement("desc", {}, "A deterministic map of visible claims, assumptions, contradiction, unresolved comparison rule, category hinge, and conversational drift."));
  const defs = svgElement("defs");
  for (const [id, className] of [["case-arrow","case-arrow-neutral"],["case-arrow-red","case-arrow-contradiction"],["case-arrow-drift","case-arrow-drift"]]) {
    const marker = svgElement("marker", { id, viewBox:"0 0 10 10", refX:9, refY:5, markerWidth:6, markerHeight:6, orient:"auto-start-reverse" });
    marker.append(svgElement("path", { d:"M 0 0 L 10 5 L 0 10 Z", class:className }));
    defs.append(marker);
  }
  svg.append(defs);
  const edgeLayer = svgElement("g", { class:"case-edge-layer" });
  const nodeLayer = svgElement("g", { class:"case-node-layer" });
  svg.append(edgeLayer, nodeLayer);
  const nodes = new Map(fixture.nodes.map((node) => [node.id, node]));
  const nodeEls = new Map();
  const edgeEls = new Map();
  const pos = (node) => [isMobile ? node.mobile_x : node.x, isMobile ? node.mobile_y : node.y];
  for (const edge of fixture.edges) {
    const source = pos(nodes.get(edge.source));
    const target = pos(nodes.get(edge.target));
    const curve = edge.relation === "contradicts" ? 70 : 24;
    const path = svgElement("path", {
      d:`M ${source[0]} ${source[1]} C ${source[0]} ${source[1] + curve}, ${target[0]} ${target[1] - curve}, ${target[0]} ${target[1]}`,
      class:`case-edge relation-${edge.relation}`,
      "data-case-edge":edge.id,
      "data-relation":edge.relation,
      "marker-end":edge.relation === "contradicts" ? "url(#case-arrow-red)" : edge.relation === "drifts_from" ? "url(#case-arrow-drift)" : "url(#case-arrow)"
    });
    edgeLayer.append(path);
    edgeEls.set(edge.id, path);
  }
  for (const node of fixture.nodes) {
    const [x,y] = pos(node);
    const group = svgElement("g", { class:`case-map-node type-${node.type}`, transform:`translate(${x} ${y})`, tabindex:"0", role:"button", "aria-label":`${node.type}: ${node.label}`, "data-case-node":node.id });
    const width = isMobile ? 176 : 190;
    const height = isMobile ? 76 : 78;
    group.append(svgElement("rect", { x:-width/2, y:-height/2, width, height, rx:node.type === "hinge" ? 0 : 4, class:"case-node-card" }));
    group.append(svgElement("text", { x:0, y:-height/2 + 15, "text-anchor":"middle", class:"case-node-type" }, node.type.replace("_", " ").toUpperCase()));
    const label = svgElement("text", { x:0, y:-5, "text-anchor":"middle", class:"case-node-label" });
    splitLabel(node.short_label, isMobile ? 20 : 22).forEach((line,index) => label.append(svgElement("tspan", { x:0, dy:index === 0 ? 0 : 14 }, line)));
    group.append(label);
    nodeLayer.append(group);
    nodeEls.set(node.id, group);
  }
  return { nodeEls, edgeEls };
}

function renderEvidence(fixture) {
  const nodeList = document.querySelector("#case-node-evidence");
  const relationList = document.querySelector("#case-relation-evidence");
  nodeList.replaceChildren();
  relationList.replaceChildren();
  for (const node of fixture.nodes) {
    const item = element("li", "case-evidence-node");
    item.dataset.nodeEvidence = node.id;
    item.append(element("span", "case-evidence-type", node.type.replace("_", " ")), element("strong", "", node.label), element("p", "", node.meaning));
    nodeList.append(item);
  }
  for (const edge of fixture.edges) {
    const item = element("li", "case-evidence-relation");
    item.dataset.edgeEvidence = edge.id;
    const source = fixture.nodes.find(({ id }) => id === edge.source);
    const target = fixture.nodes.find(({ id }) => id === edge.target);
    item.append(element("span", "case-relation-type", edge.relation), element("strong", "", `${source.short_label} → ${target.short_label}`));
    if (edge.qualification) item.append(element("p", "", edge.qualification));
    relationList.append(item);
  }
}

export function renderSemanticCase(root, rawFixture) {
  const fixture = validateSemanticFixture(rawFixture);
  root.hidden = false;
  document.body.classList.add("case-mode");
  document.title = `${fixture.title} — Dynamic Sentience Maps`;
  const skip = document.querySelector(".skip-link");
  if (skip) { skip.href = "#case-trace"; skip.textContent = "Skip to the case map"; }
  const navInstrument = document.querySelector('.site-header nav a[href="#instrument"]');
  if (navInstrument) { navInstrument.href = "#case-trace"; navInstrument.textContent = "Case map"; }
  document.querySelector("#case-title").textContent = fixture.title;
  document.querySelector("#case-deck").textContent = fixture.deck;
  document.querySelector("#case-final-status").textContent = fixture.final_status;
  document.querySelector("#case-final-explanation").textContent = fixture.final_explanation;
  document.querySelector("#case-primary-hinge").textContent = fixture.primary_hinge;
  document.querySelector("#case-primary-question").textContent = fixture.primary_question;
  document.querySelector("#case-source-disclosure").textContent = fixture.source_disclosure;
  document.querySelector("#case-claim-boundary").textContent = fixture.claim_boundary;
  document.querySelector("#case-prohibited-claim").textContent = fixture.prohibited_claim;
  const criteria = document.querySelector("#case-criteria");
  criteria.replaceChildren(...fixture.criteria_under_dispute.map((text) => element("li", "", text)));
  const exchange = document.querySelector("#case-exchange-list");
  exchange.replaceChildren(...fixture.exchange.map((entry, index) => {
    const item = element("li", "case-exchange-item");
    item.append(element("span", "", `${String(index + 1).padStart(2,"0")} / ${entry.speaker}`), element("p", "", entry.text));
    return item;
  }));
  renderEvidence(fixture);

  const svg = document.querySelector("#semantic-map");
  const layout = window.matchMedia("(max-width: 760px)");
  let graph = renderMap(svg, fixture, layout.matches);
  let activeState = "baseline";
  let selectedNode = "category_hinge";

  function selectNode(nodeId) {
    selectedNode = nodeId;
    const node = fixture.nodes.find(({ id }) => id === nodeId);
    document.querySelector("#case-selection-type").textContent = node.type.replace("_", " ").toUpperCase();
    document.querySelector("#case-selection-title").textContent = node.label;
    document.querySelector("#case-selection-meaning").textContent = node.meaning;
    graph.nodeEls.forEach((item, id) => item.classList.toggle("is-selected", id === nodeId));
  }

  function bindNodes() {
    graph.nodeEls.forEach((item, id) => {
      item.addEventListener("click", () => selectNode(id));
      item.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(id); }
      });
    });
  }

  function setState(stateId, moveFocus = false) {
    const state = stateById(fixture, stateId);
    if (!state) return;
    activeState = stateId;
    const visibleNodes = new Set(state.visible_nodes);
    const activeEdges = new Set(state.active_edges);
    graph.nodeEls.forEach((item, id) => {
      const visible = visibleNodes.has(id);
      item.classList.toggle("is-hidden", !visible);
      item.setAttribute("aria-hidden", String(!visible));
      item.style.setProperty("--case-stress", fixture.nodes.find((node) => node.id === id).stress[stateId]);
    });
    graph.edgeEls.forEach((item, id) => item.classList.toggle("is-hidden", !activeEdges.has(id)));
    document.querySelectorAll("[data-case-state]").forEach((button) => {
      const active = button.dataset.caseState === stateId;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
      button.tabIndex = active ? 0 : -1;
      if (active && moveFocus) button.focus();
    });
    document.querySelector("#case-state-caption").textContent = state.caption;
    const readings = [["kappa",state.metrics.kappa],["epsilon",state.metrics.epsilon],["capacity",state.metrics.i_capacity],["phi",state.metrics.phi_star]];
    for (const [key,value] of readings) {
      document.querySelector(`#case-reading-${key}`).textContent = value.toFixed(2);
      document.querySelector(`#case-line-${key}`).style.setProperty("--reading", `${Math.round(value * 100)}%`);
    }
    document.querySelector("#case-state-label").textContent = `${state.index} / ${state.label}`;
  }

  bindNodes();
  selectNode(selectedNode);
  setState(activeState);
  document.querySelectorAll("[data-case-state]").forEach((button, index, buttons) => {
    button.addEventListener("click", () => setState(button.dataset.caseState));
    button.addEventListener("keydown", (event) => {
      const nextId = nextStateId(fixture, button.dataset.caseState, event.key);
      if (nextId !== button.dataset.caseState) { event.preventDefault(); setState(nextId, true); }
    });
  });
  document.querySelector("#trace-disagreement").addEventListener("click", () => {
    setState("baseline");
    document.querySelector("#case-trace").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block:"start" });
  });
  layout.addEventListener("change", () => {
    graph = renderMap(svg, fixture, layout.matches);
    bindNodes();
    setState(activeState);
    selectNode(selectedNode);
  });
  return fixture;
}

export async function initSemanticCase() {
  if (typeof document === "undefined" || resolveCaseFromSearch(window.location.search) !== SEMANTIC_CASE_KEY) return null;
  const response = await fetch(SEMANTIC_CASE_PATH, { cache:"no-store" });
  if (!response.ok) throw new Error(`Could not load semantic case fixture: ${response.status}`);
  return renderSemanticCase(document.querySelector("#semantic-collision-case"), await response.json());
}

if (typeof document !== "undefined") {
  initSemanticCase().catch((error) => {
    const root = document.querySelector("#semantic-collision-case");
    if (root) { root.hidden = false; document.body.classList.add("case-mode"); root.querySelector(".case-load-error").textContent = `Case study unavailable: ${error.message}`; }
  });
}
