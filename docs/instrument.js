import { LINKS, NODES, REGIONS, STATES } from "./demo-data.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const REDUCED_MOTION = typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function edgePath(source, target, relation = "supports") {
  const [sx, sy] = source;
  const [tx, ty] = target;
  const dx = tx - sx;
  const dy = ty - sy;
  if (relation === "contradicts") {
    const bend = Math.max(52, Math.abs(dx) * 0.34);
    return `M ${sx} ${sy} C ${sx + bend} ${sy - 78}, ${tx - bend} ${ty - 78}, ${tx} ${ty}`;
  }
  if (relation === "pressures" || relation === "blocks") {
    return `M ${sx} ${sy} C ${sx + dx * 0.22} ${sy + dy * 0.08}, ${tx - dx * 0.28} ${ty - dy * 0.06}, ${tx} ${ty}`;
  }
  return `M ${sx} ${sy} C ${sx + dx * 0.42} ${sy}, ${tx - dx * 0.42} ${ty}, ${tx} ${ty}`;
}

export function averageRegionStress(regionId, stateId, nodes = NODES) {
  const values = nodes
    .filter((node) => node.region === regionId)
    .map((node) => Number(node.stress?.[stateId] ?? 0));
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function svgElement(tag, attributes = {}, text = null) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined) element.setAttribute(name, String(value));
  }
  if (text !== null) element.textContent = text;
  return element;
}

function nodeShape(node, compact = false) {
  const size = compact ? 9 : 14;
  const className = `node-core shape-${node.shape}`;
  if (node.shape === "square") {
    return svgElement("rect", { x: -size, y: -size, width: size * 2, height: size * 2, class: className });
  }
  if (node.shape === "diamond") {
    return svgElement("rect", {
      x: -size * 0.82,
      y: -size * 0.82,
      width: size * 1.64,
      height: size * 1.64,
      transform: "rotate(45)",
      class: className,
    });
  }
  if (node.shape === "triangle") {
    return svgElement("path", {
      d: `M 0 ${-size - 2} L ${size + 2} ${size} L ${-size - 2} ${size} Z`,
      class: className,
    });
  }
  if (node.shape === "hexagon") {
    return svgElement("path", {
      d: `M ${-size} 0 L ${-size / 2} ${-size * 0.88} L ${size / 2} ${-size * 0.88} L ${size} 0 L ${size / 2} ${size * 0.88} L ${-size / 2} ${size * 0.88} Z`,
      class: className,
    });
  }
  return svgElement("circle", { r: size, class: className });
}

function addDefs(svg) {
  const defs = svgElement("defs");
  const pattern = svgElement("pattern", {
    id: "halftone",
    width: 8,
    height: 8,
    patternUnits: "userSpaceOnUse",
  });
  pattern.append(svgElement("circle", { cx: 1, cy: 1, r: 0.8, class: "halftone-dot" }));
  defs.append(pattern);
  for (const [id, className] of [["arrow", "arrow-neutral"], ["arrow-warn", "arrow-warning"], ["arrow-red", "arrow-contradiction"]]) {
    const marker = svgElement("marker", {
      id,
      viewBox: "0 0 10 10",
      refX: 8,
      refY: 5,
      markerWidth: 5,
      markerHeight: 5,
      orient: "auto-start-reverse",
    });
    marker.append(svgElement("path", { d: "M 0 0 L 10 5 L 0 10 Z", class: className }));
    defs.append(marker);
  }
  svg.append(defs);
}

function drawRegistration(svg, x, y) {
  const mark = svgElement("g", { class: "registration-mark", transform: `translate(${x} ${y})` });
  mark.append(
    svgElement("line", { x1: -10, y1: 0, x2: 10, y2: 0 }),
    svgElement("line", { x1: 0, y1: -10, x2: 0, y2: 10 }),
    svgElement("circle", { r: 4 }),
  );
  svg.append(mark);
}

function renderHeroPlate() {
  const svg = document.querySelector("#hero-plate");
  if (!svg) return;
  svg.replaceChildren();
  svg.append(
    svgElement("title", {}, "A coherence map showing a stable spine, pressure field, drifting claim, and unresolved contradiction."),
    svgElement("desc", {}, "Eleven intentional nodes connect claims, evidence, constraints, memory, plans, and outcomes. An oxblood contradiction interrupts the stable path."),
  );
  addDefs(svg);
  const plate = svgElement("g", { class: "hero-plate-field" });
  plate.append(
    svgElement("path", { d: "M 70 294 C 185 170, 350 188, 548 292 C 620 330, 676 344, 734 328", class: "hero-spine" }),
    svgElement("ellipse", { cx: 570, cy: 160, rx: 150, ry: 86, class: "hero-region hero-pressure" }),
    svgElement("ellipse", { cx: 625, cy: 365, rx: 135, ry: 94, class: "hero-region hero-drift" }),
    svgElement("path", { d: "M 515 367 C 470 310, 420 286, 355 278", class: "hero-contradiction" }),
  );
  const heroNodes = [
    ["continuity", 92, 291], ["source-chain", 228, 224], ["region-current", 355, 278],
    ["human-review", 253, 365], ["migration-plan", 470, 330], ["tests-pass", 650, 323],
    ["context-load", 535, 158], ["retry-loop", 655, 120], ["stale-memory", 662, 402],
    ["region-drift", 515, 367], ["receiver-question", 610, 455],
  ];
  for (const [id, x, y] of heroNodes) {
    const node = NODES.find((item) => item.id === id);
    const group = svgElement("g", { class: `hero-node region-${node.region}`, transform: `translate(${x} ${y})` });
    group.append(nodeShape(node, true));
    if (["continuity", "region-current", "migration-plan", "region-drift"].includes(id)) {
      group.append(svgElement("text", { x: 0, y: 27, "text-anchor": "middle" }, node.shortLabel));
    }
    plate.append(group);
  }
  plate.append(
    svgElement("text", { x: 72, y: 72, class: "plate-label" }, "FIG. 01 — COHERENCE UNDER PRESSURE"),
    svgElement("text", { x: 548, y: 57, class: "plate-reading" }, "κ  0.72"),
    svgElement("text", { x: 662, y: 57, class: "plate-reading" }, "Φ* 0.64"),
    svgElement("text", { x: 515, y: 255, class: "region-caption" }, "PRESSURE FIELD"),
    svgElement("text", { x: 627, y: 487, class: "region-caption" }, "DRIFT BASIN"),
    svgElement("text", { x: 92, y: 475, class: "plate-note" }, "Stable spine remains intact."),
    svgElement("text", { x: 520, y: 505, class: "plate-note plate-note-alert" }, "One contradiction remains unresolved."),
  );
  svg.append(plate);
  drawRegistration(svg, 36, 36);
  drawRegistration(svg, 744, 484);
}

function initInstrument() {
  const svg = document.querySelector("#coherence-map");
  if (!svg) return;
  const nodesById = new Map(NODES.map((node) => [node.id, node]));
  const nodeElements = new Map();
  const edgeElements = new Map();
  const edgeLabelElements = new Map();
  const regionLabelElements = new Map();
  const layoutQuery = window.matchMedia("(max-width: 760px)");
  let activeState = STATES[2];
  let currentPositions = Object.fromEntries(
    Object.entries(layoutQuery.matches ? activeState.mobilePositions : activeState.positions)
      .map(([id, value]) => [id, [...value]]),
  );
  let selected = { kind: "node", id: "migration-plan" };
  let activeTab = "overview";

  svg.replaceChildren();
  svg.append(
    svgElement("title", {}, "Interactive Dynamic Sentience Map"),
    svgElement("desc", {}, "Select a node or named region to inspect its support, contradictions, stress, boundary status, recent change, and evidence."),
  );
  addDefs(svg);

  const regionLayer = svgElement("g", { class: "region-layer" });
  const edgeLayer = svgElement("g", { class: "edge-layer" });
  const nodeLayer = svgElement("g", { class: "node-layer" });
  const annotationLayer = svgElement("g", { class: "annotation-layer" });
  svg.append(regionLayer, edgeLayer, nodeLayer, annotationLayer);

  const regionShapes = {
    spine: svgElement("path", {
      d: "M 100 360 C 210 150, 515 180, 880 400 C 650 590, 280 620, 100 360 Z",
      class: "region-shape region-spine",
    }),
    pressure: svgElement("ellipse", { cx: 795, cy: 205, rx: 242, ry: 145, class: "region-shape region-pressure" }),
    drift: svgElement("ellipse", { cx: 865, cy: 520, rx: 250, ry: 166, class: "region-shape region-drift" }),
    boundary: svgElement("path", { d: "M 610 105 L 610 660", class: "region-shape region-boundary" }),
  };
  Object.entries(regionShapes).forEach(([id, shape]) => {
    shape.dataset.region = id;
    shape.setAttribute("role", "button");
    shape.setAttribute("tabindex", "0");
    shape.setAttribute("aria-label", `Inspect ${REGIONS.find((region) => region.id === id).label}`);
    shape.addEventListener("click", () => select({ kind: "region", id }));
    shape.addEventListener("keydown", (event) => activateOnKeyboard(event, () => select({ kind: "region", id })));
    regionLayer.append(shape);
  });

  const regionLabels = [
    ["spine", 116, 112], ["pressure", 858, 76], ["drift", 990, 683], ["boundary", 625, 128],
  ];
  for (const [id, x, y] of regionLabels) {
    const region = REGIONS.find((item) => item.id === id);
    const group = svgElement("g", {
      class: "region-label-group",
      transform: `translate(${x} ${y})`,
      role: "button",
      tabindex: 0,
      "aria-label": `Inspect ${region.label}`,
    });
    group.dataset.region = id;
    group.append(
      svgElement("text", { class: "region-number" }, region.number),
      svgElement("text", { x: 28, class: "region-name" }, region.label.toLocaleUpperCase()),
    );
    group.addEventListener("click", () => select({ kind: "region", id }));
    group.addEventListener("keydown", (event) => activateOnKeyboard(event, () => select({ kind: "region", id })));
    regionLabelElements.set(id, group);
    annotationLayer.append(group);
  }

  LINKS.forEach((link, index) => {
    const path = svgElement("path", {
      class: `edge relation-${link.relation}`,
      id: `map-edge-${index}`,
      "data-source": link.source,
      "data-target": link.target,
      "data-relation": link.relation,
      "marker-end": link.relation === "contradicts"
        ? "url(#arrow-red)"
        : ["pressures", "blocks"].includes(link.relation)
          ? "url(#arrow-warn)"
          : "url(#arrow)",
    });
    edgeElements.set(index, path);
    edgeLayer.append(path);
    if (["supports", "contradicts", "pressures"].includes(link.relation) && index !== 7) {
      const label = svgElement("text", { class: `edge-label relation-${link.relation}` }, link.relation.toLocaleUpperCase());
      edgeLabelElements.set(index, label);
      annotationLayer.append(label);
    }
  });

  NODES.forEach((node) => {
    const group = svgElement("g", {
      class: `map-node region-${node.region} type-${node.type.toLocaleLowerCase()}`,
      role: "button",
      tabindex: 0,
      "aria-label": `${node.type}: ${node.label}. Select for details.`,
    });
    group.dataset.node = node.id;
    const title = svgElement("title", {}, `${node.type}: ${node.label}`);
    const stressRing = svgElement("circle", { r: 24, class: "stress-ring" });
    const hitTarget = svgElement("circle", { r: 45, class: "node-hit-target" });
    const shape = nodeShape(node);
    const label = svgElement("text", { x: 0, y: 37, "text-anchor": "middle", class: "node-label" });
    const words = node.shortLabel.split(" ");
    const splitAt = words.length > 2 ? Math.ceil(words.length / 2) : words.length;
    label.append(svgElement("tspan", { x: 0 }, words.slice(0, splitAt).join(" ")));
    if (splitAt < words.length) label.append(svgElement("tspan", { x: 0, dy: 14 }, words.slice(splitAt).join(" ")));
    const type = svgElement("text", { x: 0, y: -31, "text-anchor": "middle", class: "node-type" }, node.type.toLocaleUpperCase());
    group.append(title, hitTarget, stressRing, shape, type, label);
    group.addEventListener("click", () => select({ kind: "node", id: node.id }));
    group.addEventListener("keydown", (event) => activateOnKeyboard(event, () => select({ kind: "node", id: node.id })));
    nodeElements.set(node.id, group);
    nodeLayer.append(group);
  });

  drawRegistration(svg, 28, 28);
  drawRegistration(svg, 1170, 710);

  function activateOnKeyboard(event, callback) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      callback();
    }
  }

  function activityFor(link, stateId) {
    const driftLink = [link.source, link.target].some((id) => nodesById.get(id)?.region === "drift");
    const pressureLink = link.relation === "pressures";
    if (driftLink) return stateId === "drift" ? "active" : stateId === "pressure" ? "faint" : "dormant";
    if (pressureLink) return stateId === "baseline" ? "faint" : "active";
    if (link.relation === "blocks") return stateId === "drift" ? "active" : "dormant";
    return "active";
  }

  function updateGeometry(positions) {
    nodeElements.forEach((element, id) => {
      const [x, y] = positions[id];
      element.setAttribute("transform", `translate(${x} ${y})`);
    });
    LINKS.forEach((link, index) => {
      const source = positions[link.source];
      const target = positions[link.target];
      edgeElements.get(index).setAttribute("d", edgePath(source, target, link.relation));
      const label = edgeLabelElements.get(index);
      if (label) {
        const x = (source[0] + target[0]) / 2;
        const y = (source[1] + target[1]) / 2 - (link.relation === "contradicts" ? 45 : 8);
        label.setAttribute("x", x);
        label.setAttribute("y", y);
        label.setAttribute("text-anchor", "middle");
      }
    });
  }

  function animateTo(target) {
    const start = structuredClone(currentPositions);
    const duration = REDUCED_MOTION ? 0 : 880;
    const startTime = performance.now();
    const ease = (value) => 1 - Math.pow(1 - value, 3);
    const frame = (now) => {
      const progress = duration === 0 ? 1 : Math.min(1, (now - startTime) / duration);
      const eased = ease(progress);
      const next = {};
      Object.keys(target).forEach((id) => {
        next[id] = [
          start[id][0] + (target[id][0] - start[id][0]) * eased,
          start[id][1] + (target[id][1] - start[id][1]) * eased,
        ];
      });
      updateGeometry(next);
      if (progress < 1) requestAnimationFrame(frame);
      else currentPositions = structuredClone(target);
    };
    requestAnimationFrame(frame);
  }

  function positionsFor(state) {
    return layoutQuery.matches ? state.mobilePositions : state.positions;
  }

  function applyLayoutMode() {
    if (layoutQuery.matches) {
      svg.setAttribute("viewBox", "0 0 480 820");
      regionShapes.spine.setAttribute("d", "M 28 142 C 110 40, 325 95, 445 485 C 305 570, 85 485, 28 142 Z");
      regionShapes.pressure.setAttribute("cx", "350");
      regionShapes.pressure.setAttribute("cy", "130");
      regionShapes.pressure.setAttribute("rx", "126");
      regionShapes.pressure.setAttribute("ry", "98");
      regionShapes.drift.setAttribute("cx", "350");
      regionShapes.drift.setAttribute("cy", "650");
      regionShapes.drift.setAttribute("rx", "128");
      regionShapes.drift.setAttribute("ry", "142");
      regionShapes.boundary.setAttribute("d", "M 215 260 L 215 770");
      const mobileLabels = {
        spine: [35, 55], pressure: [325, 34], drift: [332, 805], boundary: [226, 286],
      };
      Object.entries(mobileLabels).forEach(([id, [x, y]]) => {
        regionLabelElements.get(id)?.setAttribute("transform", `translate(${x} ${y})`);
      });
    } else {
      svg.setAttribute("viewBox", "0 0 1200 740");
      regionShapes.spine.setAttribute("d", "M 100 360 C 210 150, 515 180, 880 400 C 650 590, 280 620, 100 360 Z");
      regionShapes.pressure.setAttribute("cx", "795");
      regionShapes.pressure.setAttribute("cy", "205");
      regionShapes.pressure.setAttribute("rx", "242");
      regionShapes.pressure.setAttribute("ry", "145");
      regionShapes.drift.setAttribute("cx", "865");
      regionShapes.drift.setAttribute("cy", "520");
      regionShapes.drift.setAttribute("rx", "250");
      regionShapes.drift.setAttribute("ry", "166");
      regionShapes.boundary.setAttribute("d", "M 610 105 L 610 660");
      const desktopLabels = {
        spine: [116, 112], pressure: [858, 76], drift: [990, 683], boundary: [625, 128],
      };
      Object.entries(desktopLabels).forEach(([id, [x, y]]) => {
        regionLabelElements.get(id)?.setAttribute("transform", `translate(${x} ${y})`);
      });
    }
    currentPositions = structuredClone(positionsFor(activeState));
    updateGeometry(currentPositions);
  }

  function updateMetrics(state) {
    const readings = [
      ["kappa", state.metrics.kappa],
      ["epsilon", state.metrics.epsilon],
      ["capacity", state.metrics.iCapacity],
      ["phi", state.metrics.phiStar],
    ];
    readings.forEach(([id, value]) => {
      const element = document.querySelector(`#reading-${id}`);
      const line = document.querySelector(`#line-${id}`);
      if (element) element.textContent = value.toFixed(2);
      if (line) line.style.setProperty("--reading", `${Math.round(value * 100)}%`);
    });
    const observation = document.querySelector("#state-observation");
    const event = document.querySelector("#state-event");
    const time = document.querySelector("#state-time");
    if (observation) observation.textContent = state.observation;
    if (event) event.textContent = state.event;
    if (time) time.textContent = state.time;
  }

  function setState(id) {
    const next = STATES.find((state) => state.id === id);
    if (!next || next.id === activeState.id) return;
    activeState = next;
    svg.dataset.state = next.id;
    document.querySelectorAll("[data-state]").forEach((button) => {
      const pressed = button.dataset.state === id;
      button.setAttribute("aria-pressed", String(pressed));
      button.classList.toggle("is-active", pressed);
    });
    LINKS.forEach((link, index) => {
      const activity = activityFor(link, id);
      edgeElements.get(index).dataset.activity = activity;
      edgeLabelElements.get(index)?.setAttribute("data-activity", activity);
    });
    NODES.forEach((node) => {
      const element = nodeElements.get(node.id);
      const dormant = node.region === "drift" && id === "baseline";
      element.classList.toggle("is-dormant", dormant);
      element.style.setProperty("--stress", node.stress[id]);
    });
    regionShapes.pressure.classList.toggle("is-active", id !== "baseline");
    regionShapes.drift.classList.toggle("is-active", id === "drift");
    animateTo(positionsFor(next));
    updateMetrics(next);
    updateDetail();
  }

  function select(next) {
    selected = next;
    nodeElements.forEach((element, id) => element.classList.toggle("is-selected", next.kind === "node" && next.id === id));
    regionLayer.querySelectorAll("[data-region]").forEach((element) => {
      element.classList.toggle("is-selected", next.kind === "region" && next.id === element.dataset.region);
    });
    annotationLayer.querySelectorAll("[data-region]").forEach((element) => {
      element.classList.toggle("is-selected", next.kind === "region" && next.id === element.dataset.region);
    });
    updateDetail();
  }

  function relationsFor(nodeId) {
    return LINKS.filter((link) => link.source === nodeId || link.target === nodeId).map((link) => {
      const outbound = link.source === nodeId;
      const other = nodesById.get(outbound ? link.target : link.source);
      return {
        direction: outbound ? "out" : "in",
        relation: link.relation,
        label: other?.label ?? "Unknown node",
      };
    });
  }

  function updateDetail() {
    const region = selected.kind === "region" ? REGIONS.find((item) => item.id === selected.id) : null;
    const node = selected.kind === "node" ? nodesById.get(selected.id) : null;
    const name = node?.label ?? region?.label ?? "Selection";
    const type = node?.type ?? "Region";
    const description = node?.description ?? region?.description ?? "";
    const boundary = node?.boundary ?? region?.boundary ?? "unassessed";
    const stress = node ? node.stress[activeState.id] : averageRegionStress(region.id, activeState.id);
    const recent = node?.recent?.[activeState.id] ?? activeState.event;
    const intervention = node?.intervention ?? region?.intervention ?? "No intervention proposed.";
    const evidence = node?.evidence ?? region?.evidence.map((label) => ({ label, status: "observed" })) ?? [];
    const relations = node ? relationsFor(node.id) : LINKS.filter((link) => {
      const source = nodesById.get(link.source);
      const target = nodesById.get(link.target);
      return source?.region === region.id || target?.region === region.id;
    }).map((link) => ({
      direction: "field",
      relation: link.relation,
      label: `${nodesById.get(link.source)?.shortLabel} → ${nodesById.get(link.target)?.shortLabel}`,
    }));

    document.querySelector("#selection-kind").textContent = type.toLocaleUpperCase();
    document.querySelector("#selection-title").textContent = name;
    document.querySelector("#selection-description").textContent = description;
    const panel = document.querySelector("#detail-panel");
    panel.replaceChildren();

    if (activeTab === "overview") {
      const readings = document.createElement("dl");
      readings.className = "selection-readings";
      [
        ["Local stress", `κ ${stress.toFixed(2)}`],
        ["Boundary", boundary],
        ["Changed recently", recent],
      ].forEach(([term, value]) => {
        const row = document.createElement("div");
        const label = document.createElement("dt");
        const reading = document.createElement("dd");
        label.textContent = term;
        reading.textContent = value;
        row.append(label, reading);
        readings.append(row);
      });
      const action = document.createElement("div");
      action.className = "intervention-note";
      const actionLabel = document.createElement("span");
      const actionText = document.createElement("p");
      actionLabel.textContent = "Suggested intervention";
      actionText.textContent = intervention;
      action.append(actionLabel, actionText);
      panel.append(readings, action);
    } else if (activeTab === "relations") {
      const list = document.createElement("ol");
      list.className = "relation-list";
      relations.forEach((relation) => {
        const item = document.createElement("li");
        const verb = document.createElement("span");
        verb.textContent = relation.relation;
        const label = document.createElement("strong");
        label.textContent = relation.label;
        item.append(verb, label);
        list.append(item);
      });
      if (!relations.length) {
        const item = document.createElement("li");
        const label = document.createElement("strong");
        label.textContent = "No active relations.";
        item.append(label);
        list.append(item);
      }
      panel.append(list);
    } else {
      const list = document.createElement("ol");
      list.className = "evidence-list";
      evidence.forEach((item, index) => {
        const row = document.createElement("li");
        const number = document.createElement("span");
        const label = document.createElement("strong");
        const status = document.createElement("em");
        number.textContent = String(index + 1).padStart(2, "0");
        label.textContent = item.label;
        status.textContent = item.status;
        row.append(number, label, status);
        list.append(row);
      });
      panel.append(list);
    }
  }

  document.querySelectorAll("[data-state]").forEach((button) => {
    button.addEventListener("click", () => setState(button.dataset.state));
  });
  document.querySelectorAll("[data-detail-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      activeTab = button.dataset.detailTab;
      document.querySelectorAll("[data-detail-tab]").forEach((peer) => {
        const selectedTab = peer === button;
        peer.classList.toggle("is-active", selectedTab);
        peer.setAttribute("aria-selected", String(selectedTab));
      });
      updateDetail();
    });
  });
  layoutQuery.addEventListener("change", applyLayoutMode);

  svg.dataset.state = activeState.id;
  applyLayoutMode();
  LINKS.forEach((link, index) => {
    const activity = activityFor(link, activeState.id);
    edgeElements.get(index).dataset.activity = activity;
    edgeLabelElements.get(index)?.setAttribute("data-activity", activity);
  });
  NODES.forEach((node) => nodeElements.get(node.id).style.setProperty("--stress", node.stress[activeState.id]));
  regionShapes.pressure.classList.add("is-active");
  regionShapes.drift.classList.add("is-active");
  updateMetrics(activeState);
  select(selected);
}

if (typeof document !== "undefined") {
  renderHeroPlate();
  initInstrument();
}
