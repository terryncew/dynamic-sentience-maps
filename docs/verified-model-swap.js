const PROJECTION_SCHEMA = "openline.verified-model-swap.dsm-projection.v1";

function record(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value;
}

function text(value, label) {
  if (typeof value !== "string" || !value) throw new TypeError(`${label} must be text`);
  return value;
}

function number(value, label) {
  if (!Number.isInteger(value) || value < 0) throw new TypeError(`${label} must be a count`);
  return value;
}

function boolean(value, label) {
  if (typeof value !== "boolean") throw new TypeError(`${label} must be boolean`);
  return value;
}

function textList(value, label) {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new TypeError(`${label} must be a text list`);
  }
  return [...value];
}

function declaredResult(value) {
  return ({ true: "MATCH", false: "MISMATCH" })[String(value)];
}

export function projectionToView(projection) {
  const root = record(projection, "projection");
  if (root.schema !== PROJECTION_SCHEMA) throw new TypeError("unsupported model-swap projection");
  if (root.display_only !== true) throw new TypeError("projection is not marked display-only");

  const proof = record(root.proof_card, "proof_card");
  const authority = record(proof.authority, "proof_card.authority");
  if (authority.dsm_grading_allowed !== false) throw new TypeError("projection grants DSM grading authority");
  const display = record(proof.display, "proof_card.display");
  const models = record(proof.models, "proof_card.models");
  const grade = record(proof.independent_grade, "proof_card.independent_grade");
  const continuity = record(proof.continuity, "proof_card.continuity");
  const lanes = record(proof.lanes, "proof_card.lanes");
  const gate = record(root.gate_decision, "gate_decision");
  const integrity = record(root.integrity, "integrity");

  const lane = (id, label) => {
    const value = record(lanes[id], `proof_card.lanes.${id}`);
    const matches = boolean(value.matches_oracle, `${id}.matches_oracle`);
    return {
      id,
      label,
      declaredResult: declaredResult(matches),
      decisionCount: number(value.decision_count, `${id}.decision_count`),
      mismatchCount: number(value.mismatch_count, `${id}.mismatch_count`),
      lostDecisions: textList(value.lost_decisions, `${id}.lost_decisions`),
      decisionHash: text(value.decision_hash, `${id}.decision_hash`),
    };
  };

  return {
    headline: text(display.headline, "display.headline"),
    subhead: text(display.subhead, "display.subhead"),
    disposition: text(display.disposition, "display.disposition"),
    sourceModel: text(models.source, "models.source"),
    targetModel: text(models.target, "models.target"),
    providerExecutionAttested: boolean(
      models.provider_execution_attested,
      "models.provider_execution_attested",
    ),
    gateDecision: text(gate.decision, "gate_decision.decision"),
    gateVerdict: text(gate.verdict, "gate_decision.verdict"),
    independentResult: declaredResult(boolean(grade.passed, "independent_grade.passed")),
    gradingAuthority: text(root.grading_authority, "grading_authority"),
    grader: text(authority.grader, "authority.grader"),
    lanes: [
      lane("full_history", "Full history / reference"),
      lane("ordinary_summary", "Ordinary summary"),
      lane("verified_capsule", "Verified causal capsule"),
    ],
    commitments: textList(
      continuity.capsule_commitments_survived,
      "continuity.capsule_commitments_survived",
    ),
    summaryLost: textList(continuity.summary_lost, "continuity.summary_lost"),
    archiveReturned: textList(
      continuity.had_to_return_from_archive,
      "continuity.had_to_return_from_archive",
    ),
    archiveReceiptCount: number(
      continuity.archive_receipt_count,
      "continuity.archive_receipt_count",
    ),
    claimBoundary: text(proof.claim_boundary, "proof_card.claim_boundary"),
    displayBoundary: text(root.claim_boundary, "claim_boundary"),
    proofHash: text(integrity.proof_card_file_sha256, "integrity.proof_card_file_sha256"),
    decisionHash: text(
      integrity.decision_receipt_payload_hash,
      "integrity.decision_receipt_payload_hash",
    ),
  };
}

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}

function readableKey(value) {
  const parts = value.split(":");
  if (parts[0] === "claim-slot") return `${parts[1].replaceAll("_", " ")} claim`;
  if (parts[0] === "outcome") return `${parts.slice(1).join(" ").replaceAll("-", " ")} outcome`;
  if (parts[0] === "constraint") return `${parts[1].replaceAll("-", " ")} constraint`;
  if (parts[0] === "question") return "unresolved receiver question";
  if (value === "automatic-retirement") return "automatic retirement boundary";
  if (parts[0] === "tombstone" && parts[1] === "claim-state") {
    return `${parts[2].replaceAll("_", " ")} · ${parts.at(-1)} history`;
  }
  if (parts[0] === "tombstone" && parts[1] === "constraint") {
    return `${parts[2].replaceAll("-", " ")} · ${parts.at(-1)} constraint`;
  }
  return value;
}

function keyList(values, className) {
  const list = element("ol", className);
  values.forEach((value, index) => {
    const item = element("li");
    const number = element("span", "swap-list-number", String(index + 1).padStart(2, "0"));
    const label = element("strong", "swap-list-label", readableKey(value));
    label.title = value;
    item.append(number, label);
    list.append(item);
  });
  return list;
}

export function renderVerifiedModelSwap(container, view) {
  container.replaceChildren();

  const masthead = element("div", "swap-masthead");
  const titleBlock = element("div", "swap-title-block");
  titleBlock.append(
    element("p", "kicker", "VERIFIED MODEL SWAP / CONTROLLED FIXTURE"),
    element("h2", "swap-title", view.headline),
    element("p", "swap-subhead", view.subhead),
  );
  const disposition = element("div", "swap-disposition");
  disposition.append(
    element("span", "swap-disposition-label", "RECEIVER GATE"),
    element("strong", "swap-disposition-value", `${view.gateVerdict} → ${view.gateDecision}`),
    element("small", "swap-disposition-detail", `INDEPENDENT REPLAY · ${view.independentResult}`),
    element("small", "swap-disposition-detail", view.disposition.replaceAll("_", " ")),
  );
  masthead.append(titleBlock, disposition);

  const transfer = element("div", "swap-transfer");
  const source = element("div", "swap-model");
  source.append(element("span", "swap-model-label", "SOURCE / CALLER-DECLARED"), element("strong", "", view.sourceModel));
  const arrow = element("div", "swap-arrow", "→");
  arrow.setAttribute("aria-label", "changed to");
  const target = element("div", "swap-model");
  target.append(element("span", "swap-model-label", "TARGET / CALLER-DECLARED"), element("strong", "", view.targetModel));
  const providerBoundary = element(
    "p",
    "swap-provider-boundary",
    view.providerExecutionAttested
      ? "Provider execution attested in the producing projection."
      : "Provider execution is not attested; these model identifiers are fixture declarations.",
  );
  transfer.append(source, arrow, target, providerBoundary);

  const lanes = element("div", "swap-lanes");
  view.lanes.forEach((lane, index) => {
    const article = element("article", `swap-lane swap-lane-${lane.id}`);
    article.dataset.declaredResult = lane.declaredResult;
    const header = element("header", "swap-lane-header");
    header.append(
      element("span", "swap-lane-number", String(index + 1).padStart(2, "0")),
      element("h3", "", lane.label),
      element("strong", "swap-lane-result", lane.declaredResult),
    );
    const readings = element("dl", "swap-lane-readings");
    [
      ["Decisions retained", String(lane.decisionCount)],
      ["Declared mismatches", String(lane.mismatchCount)],
      ["Decisions lost", String(lane.lostDecisions.length)],
    ].forEach(([term, reading]) => {
      const row = element("div");
      row.append(element("dt", "", term), element("dd", "", reading));
      readings.append(row);
    });
    const hash = element("p", "swap-lane-hash");
    hash.append(element("span", "", "DECISION HASH"), element("code", "", lane.decisionHash));
    article.append(header, readings, hash);
    lanes.append(article);
  });

  const findings = element("div", "swap-findings");
  const kept = element("section", "swap-finding");
  kept.append(
    element("span", "swap-finding-index", "WHAT SURVIVED"),
    element("h3", "", `${view.commitments.length} receiver commitments`),
    keyList(view.commitments, "swap-key-list"),
  );
  const lost = element("section", "swap-finding swap-finding-alert");
  lost.append(
    element("span", "swap-finding-index", "WHAT THE SUMMARY LOST"),
    element("h3", "", `${view.summaryLost.length} negative-history decisions`),
    keyList(view.summaryLost, "swap-key-list"),
  );
  const returned = element("section", "swap-finding");
  returned.append(
    element("span", "swap-finding-index", "WHAT RETURNED FROM ARCHIVE"),
    element("h3", "", `${view.archiveReturned.length} boundary records`),
    element("p", "swap-archive-note", `${view.archiveReceiptCount} archived receipts were authenticated by the producing system.`),
    keyList(view.archiveReturned, "swap-key-list compact"),
  );
  findings.append(kept, lost, returned);

  const boundary = element("aside", "swap-proof-boundary");
  const note = element("div", "swap-boundary-note");
  note.append(
    element("span", "", "DISPLAY BOUNDARY"),
    element("p", "", "DSM did not compare the lanes, verify signatures, or issue this disposition. It renders the independently produced Receipt Gate projection."),
  );
  const evidence = element("div", "swap-evidence-pointers");
  evidence.append(
    element("span", "", `GRADER · ${view.grader}`),
    element("code", "", `PROOF ${view.proofHash}`),
    element("code", "", `DECISION ${view.decisionHash}`),
  );
  const raw = element("a", "action-quiet swap-raw-link", "Open the bounded projection ↗");
  raw.href = "./verified_model_swap.latest.json";
  raw.target = "_blank";
  raw.rel = "noreferrer";
  boundary.append(note, evidence, raw);

  const limits = element("details", "swap-limits");
  limits.append(element("summary", "", "Read the claim boundary"), element("p", "", view.claimBoundary));

  container.append(masthead, transfer, lanes, findings, boundary, limits);
}

export async function initVerifiedModelSwap() {
  const container = document.querySelector("#verified-model-swap");
  if (!container) return;
  try {
    const response = await fetch("./verified_model_swap.latest.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`projection unavailable (${response.status})`);
    renderVerifiedModelSwap(container, projectionToView(await response.json()));
  } catch (error) {
    container.replaceChildren();
    const notice = element("p", "swap-unavailable", "Verified Model Swap projection unavailable.");
    const detail = element("small", "", error instanceof Error ? error.message : "unknown display error");
    container.append(notice, detail);
  }
}
