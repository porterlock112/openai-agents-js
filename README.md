# Δ135 v135.7-RKR — auto-repin + Rekor-seal: patch + sealed run (minimal console)
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, textwrap

ROOT = Path.cwd()
PROJ = ROOT / "truthlock"
SCRIPTS = PROJ / "scripts"
GUI = PROJ / "gui"
OUT = PROJ / "out"
SCHEMAS = PROJ / "schemas"
for d in (SCRIPTS, GUI, OUT, SCHEMAS): d.mkdir(parents=True, exist_ok=True)

# --- (1) Runner patch: auto-repin missing/invalid CIDs, write-back scroll, Rekor JSON proof ---
trigger = textwrap.dedent(r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Δ135_TRIGGER — Initiate → Expand → Seal

- Scans truthlock/out/ΔLEDGER for sealed objects
- Validates ledger files (built-in + JSON Schema at truthlock/schemas/ledger.schema.json if jsonschema is installed)
- Guardrails for resolver: --max-bytes (env RESOLVER_MAX_BYTES), --allow (env RESOLVER_ALLOW or RESOLVER_ALLOW_GLOB),
  --deny (env RESOLVER_DENY or RESOLVER_DENY_GLOB)
- Auto-repin: missing or invalid CIDs get pinned (ipfs add -Q → fallback Pinata) and written back into the scroll JSON
- Emits ΔMESH_EVENT_135.json on --execute
- Optional: Pin Δ135 artifacts and Rekor-seal report
- Rekor: uploads report hash with --format json (if rekor-cli available), stores rekor_proof_<REPORT_SHA>.json
- Emits QR for best CID (report → trigger → any scanned)
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, fnmatch, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path.cwd()
OUTDIR = ROOT / "truthlock" / "out"
LEDGER_DIR = OUTDIR / "ΔLEDGER"
GLYPH_PATH = OUTDIR / "Δ135_GLYPH.json"
REPORT_PATH = OUTDIR / "Δ135_REPORT.json"
TRIGGER_PATH = OUTDIR / "Δ135_TRIGGER.json"
MESH_EVENT_PATH = OUTDIR / "ΔMESH_EVENT_135.json"
VALIDATION_PATH = OUTDIR / "ΔLEDGER_VALIDATION.json"
SCHEMA_PATH = ROOT / "truthlock" / "schemas" / "ledger.schema.json"

CID_PATTERN = re.compile(r'^(Qm[1-9A-HJ-NP-Za-km-z]{44,}|baf[1-9A-HJ-NP-Za-km-z]{20,})$')

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def which(bin_name: str) -> Optional[str]:
    from shutil import which as _which
    return _which(bin_name)

def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def find_ledger_objects() -> List[Path]:
    if not LEDGER_DIR.exists(): return []
    return sorted([p for p in LEDGER_DIR.glob("**/*.json") if p.is_file()])

# ---------- Guardrails ----------
def split_globs(s: str) -> List[str]:
    return [g.strip() for g in (s or "").split(",") if g.strip()]

def allowed_by_globs(rel_path: str, allow_globs: List[str], deny_globs: List[str]) -> Tuple[bool, str]:
    for g in deny_globs:
        if fnmatch.fnmatch(rel_path, g): return (False, f"denied by pattern: {g}")
    if allow_globs:
        for g in allow_globs:
            if fnmatch.fnmatch(rel_path, g): return (True, f"allowed by pattern: {g}")
        return (False, "no allowlist pattern matched")
    return (True, "no allowlist; allowed")

# ---------- Pin helpers ----------
def ipfs_add_cli(path: Path) -> Optional[str]:
    ipfs_bin = which("ipfs")
    if not ipfs_bin: return None
    try:
        return subprocess.check_output([ipfs_bin, "add", "-Q", str(path)], text=True).strip() or None
    except Exception:
        return None

def pinata_pin_json(obj: Dict[str, Any], name: str) -> Optional[str]:
    jwt = os.getenv("PINATA_JWT")
    if not jwt: return None
    token = jwt if jwt.startswith("Bearer ") else f"Bearer {jwt}"
    try:
        import urllib.request
        payload = {"pinataOptions": {"cidVersion": 1}, "pinataMetadata": {"name": name}, "pinataContent": obj}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request("https://api.pinata.cloud/pinning/pinJSONToIPFS", data=data,
                                     headers={"Authorization": token, "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            info = json.loads(resp.read().decode("utf-8") or "{}")
            return info.get("IpfsHash") or info.get("ipfsHash")
    except Exception:
        return None

def maybe_pin_file_or_json(path: Path, obj: Optional[Dict[str, Any]], label: str) -> Tuple[str, str]:
    cid = None
    if path.exists():
        cid = ipfs_add_cli(path)
        if cid: return ("ipfs", cid)
    if obj is not None:
        cid = pinata_pin_json(obj, label)
        if cid: return ("pinata", cid)
    return ("pending", "")

# ---------- Rekor ----------
def rekor_upload_json(path: Path) -> Tuple[bool, Dict[str, Any]]:
    binp = which("rekor-cli")
    rep_sha = sha256_path(path)
    proof_path = OUTDIR / f"rekor_proof_{rep_sha}.json"
    if not binp:
        return (False, {"message": "rekor-cli not found", "proof_path": None})
    try:
        out = subprocess.check_output([binp, "upload", "--artifact", str(path), "--format", "json"],
                                      text=True, stderr=subprocess.STDOUT)
        try:
            data = json.loads(out)
        except Exception:
            data = {"raw": out}
        proof_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        info = {
            "ok": True,
            "uuid": data.get("UUID") or data.get("uuid"),
            "logIndex": data.get("LogIndex") or data.get("logIndex"),
            "proof_path": str(proof_path.relative_to(ROOT)),
            "raw": data
        }
        return (True, info)
    except subprocess.CalledProcessError as e:
        return (False, {"message": (e.output or "").strip(), "proof_path": None})
    except Exception as e:
        return (False, {"message": str(e), "proof_path": None})

# ---------- Validation ----------
def validate_builtin(obj: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(obj, dict): return ["not a JSON object"]
    if not isinstance(obj.get("scroll_name"), str) or not obj.get("scroll_name"):
        errors.append("missing/invalid scroll_name")
    if "status" in obj and not isinstance(obj["status"], str):
        errors.append("status must be string if present")
    cid = obj.get("cid") or obj.get("ipfs_pin")
    if cid and not CID_PATTERN.match(str(cid)):
        errors.append("cid/ipfs_pin does not look like IPFS CID")
    return errors

def validate_with_schema(obj: Dict[str, Any]) -> List[str]:
    if not SCHEMA_PATH.exists(): return []
    try:
        import jsonschema
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = getattr(jsonschema, "Draft202012Validator", jsonschema.Draft7Validator)(schema)
        return [f"{'/'.join([str(p) for p in e.path]) or '<root>'}: {e.message}" for e in validator.iter_errors(obj)]
    except Exception:
        return []

def write_validation_report(results: List[Dict[str, Any]]) -> Path:
    write_json(VALIDATION_PATH, {"timestamp": now_iso(), "results": results})
    return VALIDATION_PATH

# ---------- QR ----------
def emit_cid_qr(cid: Optional[str]) -> Dict[str, Optional[str]]:
    out = {"cid": cid, "png": None, "txt": None}
    if not cid: return out
    txt_path = OUTDIR / f"cid_{cid}.txt"
    txt_path.write_text(f"ipfs://{cid}\nhttps://ipfs.io/ipfs/{cid}\n", encoding="utf-8")
    out["txt"] = str(txt_path.relative_to(ROOT))
    try:
        import qrcode
        img = qrcode.make(f"ipfs://{cid}")
        png_path = OUTDIR / f"cid_{cid}.png"
        img.save(png_path)
        out["png"] = str(png_path.relative_to(ROOT))
    except Exception:
        pass
    return out

# ---------- Glyph ----------
def update_glyph(plan: Dict[str, Any], mode: str, pins: Dict[str, Dict[str, str]], extra: Dict[str, Any]) -> Dict[str, Any]:
    glyph = {
        "scroll_name": "Δ135_TRIGGER",
        "timestamp": now_iso(),
        "initiator": plan.get("initiator", "Matthew Dewayne Porter"),
        "meaning": "Initiate → Expand → Seal",
        "phases": plan.get("phases", ["ΔSCAN_LAUNCH","ΔMESH_BROADCAST_ENGINE","ΔSEAL_ALL"]),
        "summary": {
            "ledger_files": plan.get("summary", {}).get("ledger_files", 0),
            "unresolved_cids": plan.get("summary", {}).get("unresolved_cids", 0)
        },
        "inputs": plan.get("inputs", [])[:50],
        "last_run": {"mode": mode, **extra, "pins": pins}
    }
    write_json(GLYPH_PATH, glyph); return glyph

# ---------- Main ----------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Δ135 auto-executing trigger")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--resolve-missing", action="store_true")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--rekor", action="store_true")
    ap.add_argument("--max-bytes", type=int, default=int(os.getenv("RESOLVER_MAX_BYTES", "10485760")))
    # env harmonization
    allow_env = os.getenv("RESOLVER_ALLOW", os.getenv("RESOLVER_ALLOW_GLOB", ""))
    deny_env  = os.getenv("RESOLVER_DENY",  os.getenv("RESOLVER_DENY_GLOB",  ""))
    ap.add_argument("--allow", action="append", default=[g for g in allow_env.split(",") if g.strip()])
    ap.add_argument("--deny",  action="append", default=[g for g in deny_env.split(",")  if g.strip()])
    args = ap.parse_args(argv)

    OUTDIR.mkdir(parents=True, exist_ok=True); LEDGER_DIR.mkdir(parents=True, exist_ok=True)

    # Scan ledger
    scanned: List[Dict[str, Any]] = []
    for p in find_ledger_objects():
        meta = {"path": str(p.relative_to(ROOT)), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}
        j = load_json(p)
        if j:
            meta["scroll_name"] = j.get("scroll_name"); meta["status"] = j.get("status")
            meta["cid"] = j.get("cid") or j.get("ipfs_pin") or ""
        scanned.append(meta)

    # Validate
    validation_results: List[Dict[str, Any]] = []
    for item in scanned:
        j = load_json(ROOT / item["path"]) or {}
        errs = validate_with_schema(j) or validate_builtin(j)
        if errs: validation_results.append({"path": item["path"], "errors": errs})
    validation_report_path = write_validation_report(validation_results)

    # unresolved = missing OR invalid CID
    def is_invalid_or_missing(x): 
        c = x.get("cid", "")
        return (not c) or (not CID_PATTERN.match(str(c)))
    unresolved = [s for s in scanned if is_invalid_or_missing(s)]

    plan = {
        "scroll_name": "Δ135_TRIGGER", "timestamp": now_iso(),
        "initiator": os.getenv("GODKEY_IDENTITY", "Matthew Dewayne Porter"),
        "phases": ["ΔSCAN_LAUNCH", "ΔMESH_BROADCAST_ENGINE", "ΔSEAL_ALL"],
        "summary": {"ledger_files": len(scanned), "unresolved_cids": len(unresolved)},
        "inputs": scanned
    }
    write_json(TRIGGER_PATH, plan)

    if args.dry_run or (not args.execute):
        write_json(REPORT_PATH, {
            "timestamp": now_iso(), "mode": "plan",
            "plan_path": str(TRIGGER_PATH.relative_to(ROOT)),
            "plan_sha256": sha256_path(TRIGGER_PATH),
            "validation_report": str(validation_report_path.relative_to(ROOT)),
            "result": {"message": "Δ135 planning only (no actions executed)"}
        })
        update_glyph(plan, mode="plan", pins={}, extra={
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
            "report_sha256": sha256_path(REPORT_PATH),
            "mesh_event_path": None,
            "qr": {"cid": None}
        })
        print(f"[Δ135] Planned. Ledger files={len(scanned)} unresolved_cids={len(unresolved)}")
        return 0

    # Resolve (auto-repin) with guardrails; write-back scroll JSON on success
    cid_resolution: List[Dict[str, Any]] = []
    if args.resolve_missing and unresolved:
        allow_globs = [g for sub in (args.allow or []) for g in (split_globs(sub) or [""]) if g]
        deny_globs  = [g for sub in (args.deny  or []) for g in (split_globs(sub) or [""]) if g]
        for item in list(unresolved):
            rel = item["path"]; ledger_path = ROOT / rel
            # guardrails
            ok, reason = allowed_by_globs(rel, allow_globs, deny_globs)
            if not ok:
                cid_resolution.append({"path": rel, "action": "skip", "reason": reason}); continue
            if (not ledger_path.exists()) or (ledger_path.stat().st_size > args.max_bytes):
                cid_resolution.append({"path": rel, "action": "skip", "reason": f"exceeds max-bytes ({args.max_bytes}) or missing"}); continue
            # pin flow
            j = load_json(ledger_path) or {}
            prev = j.get("cid")
            mode, cid = maybe_pin_file_or_json(ledger_path, j, f"ΔLEDGER::{ledger_path.name}")
            if cid:
                j["cid"] = cid  # write back
                try: ledger_path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception: pass
                item["cid"] = cid
                cid_resolution.append({"path": rel, "action": "repinned", "mode": mode, "prev": prev, "cid": cid})
        # recompute unresolved
        unresolved = [s for s in scanned if (not s.get("cid")) or (not CID_PATTERN.match(str(s.get("cid",""))))]
        plan["summary"]["unresolved_cids"] = len(unresolved)
        write_json(TRIGGER_PATH, plan)

    # Mesh event
    affected = [{"path": i["path"], "cid": i.get("cid", ""), "scroll_name": i.get("scroll_name")} for i in scanned]
    event = {"event_name": "ΔMESH_EVENT_135", "timestamp": now_iso(), "trigger": "Δ135",
             "affected": affected, "actions": ["ΔSCAN_LAUNCH","ΔMESH_BROADCAST_ENGINE","ΔSEAL_ALL"]}
    write_json(MESH_EVENT_PATH, event)

    pins: Dict[str, Dict[str, str]] = {}
    if args.pin:
        mode, ident = maybe_pin_file_or_json(TRIGGER_PATH, plan, "Δ135_TRIGGER")
        pins["Δ135_TRIGGER"] = {"mode": mode, "id": ident}

    # Best CID + QR
    best_cid = pins.get("Δ135_REPORT", {}).get("id") if pins else None
    if not best_cid: best_cid = pins.get("Δ135_TRIGGER", {}).get("id") if pins else None
    if not best_cid:
        for s in scanned:
            if s.get("cid"): best_cid = s["cid"]; break
    qr = emit_cid_qr(best_cid)

    # Report
    result = {"timestamp": now_iso(), "mode": "execute",
              "mesh_event_path": str(MESH_EVENT_PATH.relative_to(ROOT)),
              "mesh_event_hash": sha256_path(MESH_EVENT_PATH)}
    report = {"timestamp": now_iso(), "plan": plan, "event": event, "result": result,
              "pins": pins, "cid_resolution": cid_resolution,
              "validation_report": str(validation_report_path.relative_to(ROOT)), "qr": qr}
    write_json(REPORT_PATH, report)

    # Rekor sealing (optional)
    if args.rekor:
        ok, info = rekor_upload_json(REPORT_PATH)
        report["rekor"] = {"ok": ok, **info}
        write_json(REPORT_PATH, report)

    # Pin the report (optional, after Rekor for stable hash capture)
    if args.pin:
        rep_obj = load_json(REPORT_PATH)
        mode, ident = maybe_pin_file_or_json(REPORT_PATH, rep_obj, "Δ135_REPORT")
        pins["Δ135_REPORT"] = {"mode": mode, "id": ident}
        report["pins"] = pins; write_json(REPORT_PATH, report)

    # Glyph
    extra = {"report_path": str(REPORT_PATH.relative_to(ROOT)),
             "report_sha256": sha256_path(REPORT_PATH),
             "mesh_event_path": str(MESH_EVENT_PATH.relative_to(ROOT)),
             "qr": qr}
    if report.get("rekor", {}).get("proof_path"):
        extra["rekor_proof"] = report["rekor"]["proof_path"]
        extra["rekor_uuid"] = report["rekor"].get("uuid")
        extra["rekor_logIndex"] = report["rekor"].get("logIndex")
    update_glyph(plan, mode="execute", pins=pins, extra=extra)

    print(f"[Δ135] Executed. Mesh event → {MESH_EVENT_PATH.name}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
''').strip("\n")

(SCRIPTS / "Δ135_TRIGGER.py").write_text(trigger, encoding="utf-8")

# --- (2) Dashboard patch: Rekor panel + pinning matrix ---
tile = textwrap.dedent(r'''
import json, os, subprocess
from pathlib import Path
import streamlit as st

ROOT = Path.cwd()
OUTDIR = ROOT / "truthlock" / "out"
GLYPH = OUTDIR / "Δ135_GLYPH.json"
REPORT = OUTDIR / "Δ135_REPORT.json"
TRIGGER = OUTDIR / "Δ135_TRIGGER.json"
EVENT = OUTDIR / "ΔMESH_EVENT_135.json"
VALID = OUTDIR / "ΔLEDGER_VALIDATION.json"

def load_json(p: Path):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}

st.title("Δ135 — Auto-Repin + Rekor")
st.caption("Initiate → Expand → Seal  •  ΔSCAN_LAUNCH → ΔMESH_BROADCAST_ENGINE → ΔSEAL_ALL")

glyph = load_json(GLYPH)
report = load_json(REPORT)
plan = load_json(TRIGGER)
validation = load_json(VALID)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ledger files", plan.get("summary", {}).get("ledger_files", 0))
c2.metric("Unresolved CIDs", plan.get("summary", {}).get("unresolved_cids", 0))
c3.metric("Last run", (glyph.get("last_run", {}) or {}).get("mode", (report or {}).get("mode", "—")))
c4.metric("Timestamp", glyph.get("timestamp", "—"))

issues = validation.get("results", [])
if isinstance(issues, list) and len(issues) == 0:
    st.success("Ledger validation: clean ✅")
else:
    st.error(f"Ledger validation: {len(issues)} issue(s) ❗")
    with st.expander("Validation details"): st.json(issues)

with st.expander("Guardrails (env)"):
    st.write("**Max bytes:**", os.getenv("RESOLVER_MAX_BYTES", "10485760"))
    st.write("**Allow globs:**", os.getenv("RESOLVER_ALLOW", os.getenv("RESOLVER_ALLOW_GLOB", "")) or "—")
    st.write("**Deny globs:**",  os.getenv("RESOLVER_DENY",  os.getenv("RESOLVER_DENY_GLOB",  "")) or "—")

st.write("---")
st.subheader("Rekor Transparency")
rk = (report or {}).get("rekor", {})
if rk.get("ok"):
    st.success("Rekor sealed ✅")
    st.write("UUID:", rk.get("uuid") or "—")
    st.write("Log index:", rk.get("logIndex") or "—")
    if rk.get("proof_path"):
        proof = ROOT / rk["proof_path"]
        if proof.exists():
            st.download_button("Download Rekor proof", proof.read_bytes(), file_name=proof.name)
else:
    st.info(rk.get("message") or "Not sealed (run with --rekor)")

st.write("---")
st.subheader("Pinning Matrix")
rows = []
for r in (report.get("cid_resolution") or []):
    rows.append({"path": r.get("path"), "action": r.get("action"), "mode": r.get("mode"),
                 "cid": r.get("cid"), "reason": r.get("reason")})
if rows:
    st.dataframe(rows, hide_index=True)
else:
    st.caption("No CID resolution activity in last run.")

st.write("---")
st.subheader("Run Controls")
with st.form("run135"):
    a,b,c,d = st.columns(4)
    execute = a.checkbox("Execute", True)
    resolve = b.checkbox("Resolve missing", True)
    pin     = c.checkbox("Pin artifacts", True)
    rekor   = d.checkbox("Rekor upload", True)
    max_bytes = st.number_input("Max bytes", value=int(os.getenv("RESOLVER_MAX_BYTES","10485760")), min_value=0, step=1_048_576)
    allow = st.text_input("Allow globs (comma-separated)", value=os.getenv("RESOLVER_ALLOW", os.getenv("RESOLVER_ALLOW_GLOB","")))
    deny  = st.text_input("Deny globs (comma-separated)",  value=os.getenv("RESOLVER_DENY",  os.getenv("RESOLVER_DENY_GLOB","")))
    go = st.form_submit_button("Run Δ135")
    if go:
        args = []
        if execute: args += ["--execute"]
        else: args += ["--dry-run"]
        if resolve: args += ["--resolve-missing"]
        if pin: args += ["--pin"]
        if rekor: args += ["--rekor"]
        args += ["--max-bytes", str(int(max_bytes))]
        if allow.strip():
            for a1 in allow.split(","):
                a1=a1.strip()
                if a1: args += ["--allow", a1]
        if deny.strip():
            for d1 in deny.split(","):
                d1=d1.strip()
                if d1: args += ["--deny", d1]
        subprocess.call(["python", "truthlock/scripts/Δ135_TRIGGER.py", *args])
        st.experimental_rerun()

st.write("---")
st.subheader("Latest CID & QR")
qr = (glyph.get("last_run", {}) or {}).get("qr") or (report or {}).get("qr") or {}
if qr.get("cid"):
    st.write(f"CID: `{qr['cid']}`")
    png = OUTDIR / f"cid_{qr['cid']}.png"
    txt = OUTDIR / f"cid_{qr['cid']}.txt"
    if png.exists():
        st.image(str(png), caption=f"QR for ipfs://{qr['cid']}")
        st.download_button("Download QR PNG", png.read_bytes(), file_name=png.name)
    if txt.exists():
        st.download_button("Download QR TXT", txt.read_bytes(), file_name=txt.name)
else:
    st.caption("No CID yet.")

st.write("---")
st.subheader("Artifacts")
cols = st.columns(4)
if TRIGGER.exists(): cols[0].download_button("Δ135_TRIGGER.json", TRIGGER.read_bytes(), file_name="Δ135_TRIGGER.json")
if REPORT.exists():  cols[1].download_button("Δ135_REPORT.json",  REPORT.read_bytes(),  file_name="Δ135_REPORT.json")
if EVENT.exists():   cols[2].download_button("ΔMESH_EVENT_135.json", EVENT.read_bytes(), file_name="ΔMESH_EVENT_135.json")
if VALID.exists():   cols[3].download_button("ΔLEDGER_VALIDATION.json", VALID.read_bytes(), file_name="ΔLEDGER_VALIDATION.json")
''').strip("\n")

(GUI / "Δ135_tile.py").write_text(tile, encoding="utf-8")

# --- (3) Execute sealed run (uses env if present) ---
def run(cmd): 
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

rc, out, err = run([
    "python", str(SCRIPTS / "Δ135_TRIGGER.py"),
    "--execute", "--resolve-missing", "--pin", "--rekor",
    "--max-bytes", "10485760", "--allow", "truthlock/out/ΔLEDGER/*.json"
])

# Write a tiny summary for quick inspection
summary = {
    "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "rc": rc, "stdout": out, "stderr": err,
    "artifacts": sorted(p.name for p in OUT.iterdir())
}
(OUT / "Δ135_RKR_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))# OpenAI Agents SDK (JavaScript/TypeScript)

[![npm version](https://badge.fury.io/js/@openai%2Fagents.svg)](https://badge.fury.io/js/@openai%2Fagents)
[![CI](https://github.com/openai/openai-agents-js/actions/workflows/test.yml/badge.svg)](https://github.com/openai/openai-agents-js/actions/workflows/test.yml)

The OpenAI Agents SDK is a lightweight yet powerful framework for building multi-agent workflows in JavaScript/TypeScript. It is provider-agnostic, supporting OpenAI APIs and more.

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

> [!NOTE]
> Looking for the Python version? Check out [Agents SDK Python](https://github.com/openai/openai-agents-python).

## Core concepts

1. **Agents**: LLMs configured with instructions, tools, guardrails, and handoffs.
2. **Handoffs**: Specialized tool calls for transferring control between agents.
3. **Guardrails**: Configurable safety checks for input and output validation.
4. **Tracing**: Built-in tracking of agent runs, allowing you to view, debug, and optimize your workflows.

Explore the [`examples/`](examples/) directory to see the SDK in action.

## Supported Features

- [x] **Multi-Agent Workflows**: Compose and orchestrate multiple agents in a single workflow.
- [x] **Tool Integration**: Seamlessly call tools/functions from within agent responses.
- [x] **Handoffs**: Transfer control between agents dynamically during a run.
- [x] **Structured Outputs**: Support for both plain text and schema-validated structured outputs.
- [x] **Streaming Responses**: Stream agent outputs and events in real time.
- [x] **Tracing & Debugging**: Built-in tracing for visualizing and debugging agent runs.
- [x] **Guardrails**: Input and output validation for safety and reliability.
- [x] **Parallelization**: Run agents or tool calls in parallel and aggregate results.
- [x] **Human-in-the-Loop**: Integrate human approval or intervention into workflows.
- [x] **Realtime Voice Agents**: Build realtime voice agents using WebRTC or Websockets
- [x] **Local MCP Server Support**: Give an Agent access to a locally running MCP server to provide tools
- [x] **Separate optimized browser package**: Dedicated package meant to run in the browser for Realtime agents.
- [x] **Broader model support**: Use non-OpenAI models through the Vercel AI SDK adapter
- [ ] **Long running functions**: Suspend an agent loop to execute a long-running function and revive it later <img src="https://img.shields.io/badge/Future-lightgrey" alt="Future" style="width: auto; height: 1em; vertical-align: middle;">
- [ ] **Voice pipeline**: Chain text-based agents using speech-to-text and text-to-speech into a voice agent <img src="https://img.shields.io/badge/Future-lightgrey" alt="Future" style="width: auto; height: 1em; vertical-align: middle;">

## Get started

### Supported environments

- Node.js 22 or later
- Deno
- Bun

Experimental support:

- Cloudflare Workers with `nodejs_compat` enabled

[Check out the documentation](https://openai.github.io/openai-agents-js/guides/troubleshooting/) for more detailed information.

### Installation

This SDK currently does not work with `zod@3.25.68` and above. Please install `zod@3.25.67` (or any older version) explicitly. We will resolve this dependency issue soon. Please check [this issue](https://github.com/openai/openai-agents-js/issues/187) for updates.

```bash
npm install @openai/agents 'zod@<=3.25.67'
```

## Hello world example

```js
import { Agent, run } from '@openai/agents';

const agent = new Agent({
  name: 'Assistant',
  instructions: 'You are a helpful assistant',
});

const result = await run(
  agent,
  'Write a haiku about recursion in programming.',
);
console.log(result.finalOutput);
// Code within the code,
// Functions calling themselves,
// Infinite loop's dance.
```

(_If running this, ensure you set the `OPENAI_API_KEY` environment variable_)

## Functions example

```js
import { z } from 'zod';
import { Agent, run, tool } from '@openai/agents';

const getWeatherTool = tool({
  name: 'get_weather',
  description: 'Get the weather for a given city',
  parameters: z.object({ city: z.string() }),
  execute: async (input) => {
    return `The weather in ${input.city} is sunny`;
  },
});

const agent = new Agent({
  name: 'Data agent',
  instructions: 'You are a data agent',
  tools: [getWeatherTool],
});

async function main() {
  const result = await run(agent, 'What is the weather in Tokyo?');
  console.log(result.finalOutput);
}

main().catch(console.error);
```

## Handoffs example

```js
import { z } from 'zod';
import { Agent, run, tool } from '@openai/agents';

const getWeatherTool = tool({
  name: 'get_weather',
  description: 'Get the weather for a given city',
  parameters: z.object({ city: z.string() }),
  execute: async (input) => {
    return `The weather in ${input.city} is sunny`;
  },
});

const dataAgent = new Agent({
  name: 'Data agent',
  instructions: 'You are a data agent',
  handoffDescription: 'You know everything about the weather',
  tools: [getWeatherTool],
});

// Use Agent.create method to ensure the finalOutput type considers handoffs
const agent = Agent.create({
  name: 'Basic test agent',
  instructions: 'You are a basic agent',
  handoffs: [dataAgent],
});

async function main() {
  const result = await run(agent, 'What is the weather in San Francisco?');
  console.log(result.finalOutput);
}

main().catch(console.error);
```

## Voice Agent

```js
import { z } from 'zod';
import { RealtimeAgent, RealtimeSession, tool } from '@openai/agents-realtime';

const getWeatherTool = tool({
  name: 'get_weather',
  description: 'Get the weather for a given city',
  parameters: z.object({ city: z.string() }),
  execute: async (input) => {
    return `The weather in ${input.city} is sunny`;
  },
});

const agent = new RealtimeAgent({
  name: 'Data agent',
  instructions: 'You are a data agent',
  tools: [getWeatherTool],
});

// Intended to be run the browser
const { apiKey } = await fetch('/path/to/ephemerial/key/generation').then(
  (resp) => resp.json(),
);
// automatically configures audio input/output so start talking
const session = new RealtimeSession(agent);
await session.connect({ apiKey });
```

## Running Complete Examples

The [`examples/`](examples/) directory contains a series of examples to get started:

- `pnpm examples:basic` - Basic example with handoffs and tool calling
- `pnpm examples:agents-as-tools` - Using agents as tools for translation
- `pnpm examples:web-search` - Using the web search tool
- `pnpm examples:file-search` - Using the file search tool
- `pnpm examples:deterministic` - Deterministic multi-agent workflow
- `pnpm examples:parallelization` - Running agents in parallel and picking the best result
- `pnpm examples:human-in-the-loop` - Human approval for certain tool calls
- `pnpm examples:streamed` - Streaming agent output and events in real time
- `pnpm examples:streamed:human-in-the-loop` - Streaming output with human-in-the-loop approval
- `pnpm examples:routing` - Routing between agents based on language or context
- `pnpm examples:realtime-demo` - Framework agnostic Voice Agent example
- `pnpm examples:realtime-next` - Next.js Voice Agent example application

## The agent loop

When you call `Runner.run()`, the SDK executes a loop until a final output is produced.

1. The agent is invoked with the given input, using the model and settings configured on the agent (or globally).
2. The LLM returns a response, which may include tool calls or handoff requests.
3. If the response contains a final output (see below), the loop ends and the result is returned.
4. If the response contains a handoff, the agent is switched to the new agent and the loop continues.
5. If there are tool calls, the tools are executed, their results are appended to the message history, and the loop continues.

You can control the maximum number of iterations with the `maxTurns` parameter.

### Final output

The final output is the last thing the agent produces in the loop.

1. If the agent has an `outputType` (structured output), the loop ends when the LLM returns a response matching that type.
2. If there is no `outputType` (plain text), the first LLM response without tool calls or handoffs is considered the final output.

**Summary of the agent loop:**

- If the current agent has an `outputType`, the loop runs until structured output of that type is produced.
- If not, the loop runs until a message is produced with no tool calls or handoffs.

### Error handling

- If the maximum number of turns is exceeded, a `MaxTurnsExceededError` is thrown.
- If a guardrail is triggered, a `GuardrailTripwireTriggered` exception is raised.

## Documentation

To view the documentation locally:

```bash
pnpm docs:dev
```

Then visit [http://localhost:4321](http://localhost:4321) in your browser.

## Development

If you want to contribute or edit the SDK/examples:

1. Install dependencies

   ```bash
   pnpm install
   ```

2. Build the project

   ```bash
   pnpm build
   ```

3. Run tests, linter, etc. (add commands as appropriate for your project)

## Acknowledgements

We'd like to acknowledge the excellent work of the open-source community, especially:

- [zod](https://github.com/colinhacks/zod) (schema validation)
- [Starlight](https://github.com/withastro/starlight)
- [vite](https://github.com/vitejs/vite) and [vitest](https://github.com/vitest-dev/vitest)
- [pnpm](https://pnpm.io/)
- [Next.js](https://github.com/vercel/next.js)

We're committed to building the Agents SDK as an open source framework so others in the community can expand on our approach.

For more details, see the [documentation](https://openai.github.io/openai-agents-js) or explore the [`examples/`](examples/) directory.
