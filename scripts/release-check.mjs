import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const run = (command, args) => execFileSync(command, args, { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
const digest = (path) => createHash("sha256").update(readFileSync(resolve(root, path))).digest("hex");
const python = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
const pythonSources = ["server main.py", "dsm_security.py", "olp_client.py", "wire_openline_demo.py"];

const checks = [];
function check(name, action) {
  try {
    const detail = action();
    checks.push({ name, status: "pass", detail: String(detail || "verified").trim() });
  } catch (error) {
    checks.push({ name, status: "fail", detail: error.stderr?.trim() || error.message });
  }
}

check("JavaScript syntax: demo data", () => run(process.execPath, ["--check", "docs/demo-data.js"]));
check("JavaScript syntax: instrument", () => run(process.execPath, ["--check", "docs/instrument.js"]));
check("JavaScript syntax: model-swap display", () => run(process.execPath, ["--check", "docs/verified-model-swap.js"]));
check("JavaScript syntax: semantic case", () => run(process.execPath, ["--check", "docs/semantic-case.js"]));
check("Semantic case fixture JSON", () => JSON.parse(readFileSync(resolve(root, "docs/demos/same-word-different-rules.json"), "utf8")) && "parsed");
check("Python syntax: all sources", () => run(python, ["-m", "py_compile", ...pythonSources]));
check("Python integrity and canonical receipt", () => run(python, ["tests/python_integrity.py"]));
check("FastAPI import and runtime smoke", () => run(python, ["tests/server_runtime_smoke.py"]));
check("Deterministic test suite", () => run(process.execPath, ["--test", "tests/instrument.test.mjs"]));
check("No remote runtime dependencies", () => {
  const html = readFileSync(resolve(root, "docs/index.html"), "utf8");
  if (/<(script|link)[^>]+(?:src|href)=["']https?:/i.test(html)) throw new Error("remote runtime asset found");
  return "all CSS and JavaScript are local";
});

const pythonFiles = ["server main.py", "dsm_security.py", "olp_client.py", "wire_openline_demo.py"];
const report = {
  schema: "dynamic-sentience-maps/release-verification/v2",
  candidate: "0.2.0-rc.4",
  generated_at: new Date().toISOString(),
  disposition: checks.every(({ status }) => status === "pass") ? "clear" : "hold",
  checks,
  python_source_sha256: Object.fromEntries(pythonFiles.map((path) => [path, digest(path)])),
  canonical_receipt_sha256: digest("docs/receipt.latest.json"),
  verified_model_swap_projection_sha256: digest("docs/verified_model_swap.latest.json"),
  public_surface_sha256: Object.fromEntries([
    "docs/index.html", "docs/styles.css", "docs/demo-data.js", "docs/instrument.js",
    "docs/verified-model-swap.js", "docs/semantic-case.js", "docs/demos/same-word-different-rules.json",
  ].map((path) => [path, digest(path)])),
  verification_source_sha256: Object.fromEntries([
    "scripts/release-check.mjs", "tests/instrument.test.mjs", "tests/python_integrity.py",
    "tests/server_runtime_smoke.py", ".github/workflows/test.yml", "requirements.txt",
  ].map((path) => [path, digest(path)])),
  claim_boundary: [
    "The graph and readings are deterministic demonstration proxies.",
    "DSM renders a producer-supplied model-swap projection and does not grade the lanes.",
    "DSM deterministically renders an anonymized semantic-collision fixture across baseline, pressure, and drift states.",
    "DSM does not determine whether a person, institution, or statement is racist.",
    "Fixture model identifiers do not attest execution by a commercial provider.",
    "This verification does not establish scientific validity or a private kernel implementation.",
    "Evidence and receiver policy must be checked in addition to signatures or hashes."
  ],
};

writeFileSync(resolve(root, "RELEASE_VERIFICATION.json"), `${JSON.stringify(report, null, 2)}\n`);
for (const item of checks) console.log(`${item.status.toUpperCase()}  ${item.name}`);
console.log(`${report.disposition.toUpperCase()}  ${report.candidate}`);
if (report.disposition !== "clear") process.exitCode = 1;
