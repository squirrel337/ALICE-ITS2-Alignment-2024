#!/usr/bin/env python3
"""Local web console for configuring, launching and reviewing ITS2 alignment runs.

    python3 tools/run_console.py [--port 8770] [--module .] [--roots /data,/scratch]

Pick the data file, geometry, starting alignment and seed parameters, choose where the
output goes, and launch. The console composes a self-contained run directory and never
writes into the module checkout, which is what lets it drive a tree that is meant to stay
untouched.

Version independence is by discovery, not by a table: the knobs are parsed out of the
module's own DetectorConstant.h and YMLPParallel.h, and the geometry backend is detected
from YDetectorGeometry. The 2024 and 2025 trees share an entry point
(batch_train -> run_train_circle(step), all paths cwd-relative), so the same composition
works for both.

Python standard library only, except uproot for validating inputs and reducing outputs.
It stays up even when ROOT does not, which matters because a broken cling install is
exactly when you want to see what went wrong.
"""
import argparse, html, json, os, re, shutil, signal, subprocess, sys, tarfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PAGE = os.path.join(REPO, "docs", "run-console.html")

# Knobs the console exposes. Anything else in those headers is left exactly as the module
# ships it -- the console is a launcher, not an editor.
EDITABLE = {
    "nDATA":        ("YMLPParallel.h",            "events read from the source tree"),
    "nEPOCH":       ("YMLPParallel.h",            "0 evaluates only; above 0 trains"),
    "nCORE":        ("YMLPParallel.h",            ""),
    "jparallel":    ("YMLPParallel.h",            ""),
    "nTrackMax":    ("Ymlp/inc/DetectorConstant.h", "prongs per event; ndf = 12n+1"),
    "DET_MAG":      ("Ymlp/inc/DetectorConstant.h", "signed field in T; the sign reaches the impact parameter"),
    "Update_pTmin": ("Ymlp/inc/DetectorConstant.h", ""),
    "Update_pTmax": ("Ymlp/inc/DetectorConstant.h", ""),
    "VERTEXFIT":    ("Ymlp/inc/DetectorConstant.h", ""),
}

# Fitted on this session's completed runs; predicts run_negB (nDATA 4000, nEPOCH 5) at
# 304 min against 304.6 measured. Shown before launch so a five-hour job is a choice.
COST_FIXED, COST_EVAL, COST_EPOCH = 4.8, 0.00337, 0.01431

JOBS, JOBS_LOCK = {}, threading.Lock()


# ---------------------------------------------------------------- module inspection
def read_defines(path):
    out = {}
    try:
        for ln in open(path, errors="ignore"):
            m = re.match(r"\s*#define\s+(\w+)\s+(\S+)", ln)
            if m:
                out.setdefault(m.group(1), m.group(2))
    except OSError:
        pass
    return out


def inspect_module(root):
    """What this module version offers, discovered rather than assumed."""
    par = os.path.join(root, "YMLPParallel.h")
    det = os.path.join(root, "Ymlp", "inc", "DetectorConstant.h")
    geo_h = os.path.join(root, "Ymlp", "inc", "YDetectorGeometry.h")
    geo_c = os.path.join(root, "Ymlp", "src", "YDetectorGeometry.cxx")
    ok = all(os.path.exists(p) for p in (par, det, geo_h, os.path.join(root, "run_train_circle.C")))
    defines = {}
    defines.update(read_defines(par))
    defines.update(read_defines(det))
    src = ""
    for p in (geo_h, geo_c):
        try:
            src += open(p, errors="ignore").read()
        except OSError:
            pass
    # A cache-backed geometry can run without O2; an O2-only one cannot, and saying so up
    # front is better than a compile error twenty minutes in.
    cache_capable = "LoadCache" in src or "YGEOM_CACHE" in src
    o2_required = ("ITSBase/GeometryTGeo.h" in src) and not cache_capable
    knobs = []
    for name, (f, note) in EDITABLE.items():
        if name in defines:
            knobs.append({"name": name, "value": defines[name], "file": f, "note": note})
    return {"root": os.path.abspath(root), "valid": ok, "knobs": knobs,
            "cacheCapable": cache_capable, "o2Required": o2_required,
            "nDefines": len(defines)}


def estimate_minutes(ndata, nepoch):
    """Wall-clock estimate, fitted on this session's completed runs.

    At nEPOCH = 0 Train() returns before the test pass (YMultiLayerPerceptron.cxx:1279),
    so the epoch -1 cost is the training pass alone -- about 75% of the work. Ignoring
    that over-predicted a 50k evaluation by 55%.
    """
    try:
        n, e = float(ndata), float(nepoch)
    except (TypeError, ValueError):
        return None
    eval_share = 0.75 if e <= 0 else 1.0
    return COST_FIXED + COST_EVAL * eval_share * n + max(e, 0) * COST_EPOCH * n


# ---------------------------------------------------------------- file browsing
ROLE_MATCH = {
    "data":   lambda n: n.endswith(".root"),
    "geom":   lambda n: n.endswith(".root"),
    "align":  lambda n: n.endswith(".root"),
    "params": lambda n: n.endswith((".tgz", ".tar.gz")) or True,   # also plain directories
    "out":    lambda n: False,
}


def browse(dirpath, role, allowed):
    dirpath = os.path.abspath(dirpath or allowed[0])
    if not any(dirpath == a or dirpath.startswith(a.rstrip("/") + "/") for a in allowed):
        raise PermissionError(f"{dirpath} is outside the allowed roots")
    entries = []
    try:
        for name in sorted(os.listdir(dirpath)):
            full = os.path.join(dirpath, name)
            try:
                isdir = os.path.isdir(full)
                size = 0 if isdir else os.path.getsize(full)
            except OSError:
                continue
            if not isdir and not ROLE_MATCH.get(role, lambda n: True)(name):
                continue
            entries.append({"name": name, "dir": isdir, "size": size,
                            "path": full, "link": os.path.islink(full)})
    except OSError as e:
        raise PermissionError(str(e))
    parent = os.path.dirname(dirpath.rstrip("/")) or "/"
    return {"dir": dirpath, "parent": parent, "entries": entries}


# ---------------------------------------------------------------- preflight
def _uproot():
    try:
        import uproot
        return uproot
    except ImportError:
        return None


def preflight(cfg, module):
    """Check what can be checked before spending hours on a job that cannot work.

    Every item here corresponds to a failure that actually happened while developing
    this module: a missing weightsDU.txt that produced -nan, a step 0 whose previous-USL
    name is empty, two concurrent jobs that OOM-killed each other.
    """
    out, up = [], _uproot()

    def add(level, what, detail):
        out.append({"level": level, "what": what, "detail": detail})

    mi = inspect_module(module)
    if not mi["valid"]:
        add("error", "module", f"{module} does not look like an alignment checkout")
    if mi["o2Required"]:
        add("warn", "geometry backend", "this tree needs O2 at runtime; it cannot run where O2 is absent")

    # --- source data
    d = cfg.get("data", "")
    if not d or not os.path.exists(d):
        add("error", "data file", "not set or missing")
    elif up is None:
        add("warn", "data file", "uproot unavailable, contents not checked")
    else:
        try:
            f = up.open(d)
            tname = cfg.get("tree", "DataInput")
            if tname not in [k.split(";")[0] for k in f.keys()]:
                add("error", "data file", f"no '{tname}' tree; found {f.keys()[:4]}")
            else:
                t = f[tname]
                n = t.num_entries
                need = int(cfg.get("nDATA") or 0)
                add("ok", "data file", f"{tname}: {n:,} entries")
                if need > n:
                    add("warn", "nDATA", f"{need:,} requested but the tree holds {n:,}")
        except Exception as e:
            add("error", "data file", f"unreadable: {e}")

    # --- geometry and alignment
    for role, label in (("geom", "geometry"), ("align", "alignment")):
        p = cfg.get(role, "")
        if not p:
            add("warn", label, "not set")
        elif not os.path.exists(p):
            add("error", label, f"missing: {p}")
        elif up is not None:
            try:
                up.open(p).keys()
                add("ok", label, os.path.basename(p))
            except Exception as e:
                add("error", label, f"unreadable: {e}")

    # --- seed parameters. The missing weightsDU.txt cost a full diagnostic cycle.
    p = cfg.get("params", "")
    step = int(cfg.get("step") or 901)
    if not p or not os.path.exists(p):
        add("error", "seed parameters", "not set or missing")
    else:
        names = []
        if os.path.isdir(p):
            for base, _, fs in os.walk(p):
                names += [os.path.relpath(os.path.join(base, x), p) for x in fs]
        else:
            try:
                with tarfile.open(p) as tf:
                    names = tf.getnames()
            except Exception as e:
                add("error", "seed parameters", f"cannot read archive: {e}")
        flat = "\n".join(names)
        for need, why in (("UpdateSensorsList.txt", "read by SetPrevUSL"),
                          ("weights.txt", "read by SetPrevWeight"),
                          ("weightsDU.txt", "read by SetPrevWeightDU; absent, the detector-unit "
                                            "normalisations stay uninitialised and the cost comes out -nan")):
            if need in flat:
                add("ok", f"seed: {need}", why)
            else:
                add("error", f"seed: {need}", f"not in the archive -- {why}")
    if step <= 0:
        add("error", "step", "step must be >= 1; at step 0 LoadUpdateSensorList gets an empty name and errors out")

    # --- output location
    o = cfg.get("outdir", "")
    if not o:
        add("error", "output directory", "not set")
    else:
        try:
            os.makedirs(o, exist_ok=True)
            free = shutil.disk_usage(o).free / 1e9
            add("ok" if free > 2 else "warn", "output directory", f"{o} — {free:.1f} GB free")
        except OSError as e:
            add("error", "output directory", f"not writable: {e}")

    # --- machine headroom. 8 GB resident per job, independent of nDATA.
    try:
        mem = {}
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":")
            mem[k] = int(v.split()[0]) * 1024
        avail = mem.get("MemAvailable", 0) / 1e9
        running = sum(1 for j in JOBS.values() if j["state"] == "running")
        if running:
            add("error", "concurrency", f"{running} job already running — a second needs 8 GB more and would OOM one of them")
        add("ok" if avail > 9 else "warn", "memory", f"{avail:.1f} GB available; a job holds ~8 GB regardless of nDATA")
    except OSError:
        pass

    est = estimate_minutes(cfg.get("nDATA"), cfg.get("nEPOCH"))
    if est:
        add("ok", "estimated runtime", f"{est/60:.1f} h  ({est:.0f} min)")
    return {"findings": out,
            "blocked": any(f["level"] == "error" for f in out),
            "module": mi}


# ---------------------------------------------------------------- run composition
COPY_TREE = ["Ymlp", "monitor", "geometry", "NetworkParameters", "tools"]
COPY_FILE = ["run_train_circle.C", "run_profile_beam.C", "batch_train", "YMLPParallel.h",
             "YMLPBeamProfile.h", "TrendingNetwork.tgz", "OffsetSlopeCorrectionParams.txt",
             "UpdateSensorsList.txt", "NTracksBySensor.txt"]
LINK_AS = {"data": "XXXXinput.root", "geom": "o2sim_geometry.root", "align": "ITSAlignment.root"}


def patch_defines(path, updates):
    """Rewrite #define values in place, leaving every other line untouched."""
    if not updates or not os.path.exists(path):
        return
    lines = open(path, errors="ignore").read().split("\n")
    for i, ln in enumerate(lines):
        m = re.match(r"(\s*#define\s+)(\w+)(\s+)(\S+)(.*)$", ln)
        if m and m.group(2) in updates:
            lines[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{updates[m.group(2)]}{m.group(5)}"
    open(path, "w").write("\n".join(lines))


def compose(cfg, module, job_dir):
    """Build a self-contained run directory. The module checkout is only ever read."""
    os.makedirs(job_dir, exist_ok=True)
    notes = []
    for d in COPY_TREE:
        s = os.path.join(module, d)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(job_dir, d), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "AlignLib"))
    for f in COPY_FILE:
        s = os.path.join(module, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(job_dir, f))

    # Big read-only inputs are linked, not copied: the source file alone is ~800 MB and
    # copying it per job would fill the disk for no benefit.
    for role, dest in LINK_AS.items():
        src = cfg.get(role)
        if src and os.path.exists(src):
            dst = os.path.join(job_dir, dest)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            notes.append(f"{dest} -> {src}")
    for extra in ("ITSClusterDictionary_20220903.root", "o2simtopology_13294.json"):
        s = os.path.join(module, extra)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(job_dir, extra))

    # Seed parameters land in MLPTrain_Step<step-1>, which is where run_train_circle looks.
    step = int(cfg.get("step") or 901)
    seed_dir = os.path.join(job_dir, f"MLPTrain_Step{step-1}")
    os.makedirs(seed_dir, exist_ok=True)
    p = cfg.get("params", "")
    if os.path.isdir(p):
        shutil.copytree(p, seed_dir, dirs_exist_ok=True)
    elif p:
        with tarfile.open(p) as tf:
            members = tf.getnames()
            top = os.path.commonpath([m for m in members if "/" in m]) if any("/" in m for m in members) else ""
            tf.extractall(seed_dir if not top else job_dir)
            if top:
                got = os.path.join(job_dir, top)
                if os.path.abspath(got) != os.path.abspath(seed_dir):
                    shutil.copytree(got, seed_dir, dirs_exist_ok=True)
                    shutil.rmtree(got, ignore_errors=True)
    notes.append(f"seed parameters -> MLPTrain_Step{step-1}/")

    # Physics knobs are #defines, so they are patched into this run's own copies.
    by_file = {}
    for name, val in (cfg.get("defines") or {}).items():
        if name in EDITABLE:
            by_file.setdefault(EDITABLE[name][0], {})[name] = val
    for rel, ups in by_file.items():
        patch_defines(os.path.join(job_dir, rel), ups)
        notes.append(f"{rel}: " + ", ".join(f"{k}={v}" for k, v in ups.items()))

    bt = os.path.join(job_dir, "batch_train")
    if os.path.isfile(bt):
        txt = open(bt).read()
        new = re.sub(r"run_train_circle\.C\(\s*\d+\s*\)", f"run_train_circle.C({step})", txt)
        if new != txt:
            open(bt, "w").write(new)
            notes.append(f"batch_train: step {step}")

    manifest = {"job_dir": job_dir, "module": os.path.abspath(module), "step": step,
                "config": cfg, "composed": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": notes}
    json.dump(manifest, open(os.path.join(job_dir, "run_console_manifest.json"), "w"), indent=2)
    return manifest


def launch(job_id, job_dir, rootsys):
    env = dict(os.environ)
    if rootsys:
        env["ROOTSYS"] = rootsys
        env["PATH"] = os.path.join(rootsys, "bin") + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = os.path.join(rootsys, "lib")
    log_path = os.path.join(job_dir, "run.log")
    log = open(log_path, "wb", buffering=0)
    log.write(f"START {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
    p = subprocess.Popen(["root", "-l", "-b", "-q", "batch_train"], cwd=job_dir,
                         stdout=log, stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "dir": job_dir, "pid": p.pid, "proc": p,
                        "log": log_path, "state": "running", "started": time.time()}

    def wait():
        rc = p.wait()
        try:
            log.write(f"\nEXIT {rc} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
            log.close()
        except Exception:
            pass
        with JOBS_LOCK:
            JOBS[job_id]["state"] = "done" if rc == 0 else f"failed ({rc})"
            JOBS[job_id]["rc"] = rc
            JOBS[job_id]["ended"] = time.time()
    threading.Thread(target=wait, daemon=True).start()
    return JOBS[job_id]


# ---------------------------------------------------------------- outputs
def outputs_for(job_dir, step):
    """What the job produced, in the order a reader wants it."""
    out = {"stepDir": None, "monitors": [], "residuals": [], "weights": [], "trending": [], "log": None}
    lp = os.path.join(job_dir, "run.log")
    if os.path.exists(lp):
        out["log"] = {"path": lp, "size": os.path.getsize(lp)}
    sd = os.path.join(job_dir, f"MLPTrain_Step{step}")
    if not os.path.isdir(sd):
        return out
    out["stepDir"] = sd
    res = os.path.join(sd, "Residual")
    if os.path.isdir(res):
        for n in sorted(os.listdir(res)):
            rec = {"name": n, "path": os.path.join(res, n),
                   "size": os.path.getsize(os.path.join(res, n))}
            m = re.search(r"_At_(-?\d+)\.root$", n)
            if m:
                rec["epoch"] = int(m.group(1))
            (out["monitors"] if n.startswith("Residual_Monitor") else out["residuals"]).append(rec)
    for key, sub in (("weights", "weights"), ("trending", "TrendingNetwork")):
        d = os.path.join(sd, sub)
        if os.path.isdir(d):
            for n in sorted(os.listdir(d))[:40]:
                out[key].append({"name": n, "path": os.path.join(d, n)})
    return out


def reduce_outputs(job_dir, step, out_json, all_epochs=True):
    """Hand the run to tools/build_explorer_data.py so it opens in the explorer."""
    script = os.path.join(HERE, "build_explorer_data.py")
    cmd = [sys.executable, script,
           "--step-dir", os.path.join(job_dir, f"MLPTrain_Step{step}"),
           "--epoch", "-1",
           "--log", os.path.join(job_dir, "run.log"),
           "--geometry", os.path.join(job_dir, "geometry", "its2_geom.root"),
           "--weights", os.path.join(job_dir, "NetworkParameters", "weights.txt"),
           "--seed-weights", os.path.join(job_dir, f"MLPTrain_Step{step-1}", "weights", "weights.txt"),
           "--usl", os.path.join(job_dir, "UpdateSensorsList.txt"),
           "--out", out_json]
    if all_epochs:
        cmd.append("--all-epochs")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    return {"ok": r.returncode == 0, "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:],
            "out": out_json if r.returncode == 0 else None,
            "size": os.path.getsize(out_json) if r.returncode == 0 and os.path.exists(out_json) else 0}


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "ITS2RunConsole/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                if not os.path.exists(PAGE):
                    return self._send(500, f"missing {PAGE}", "text/plain")
                return self._send(200, open(PAGE, encoding="utf-8").read(), "text/html; charset=utf-8")
            if u.path == "/api/module":
                return self._send(200, inspect_module(q.get("root") or self.server.module))
            if u.path == "/api/roots":
                return self._send(200, {"roots": self.server.roots,
                                        "module": os.path.abspath(self.server.module),
                                        "rootsys": self.server.rootsys or ""})
            if u.path == "/api/browse":
                return self._send(200, browse(q.get("dir"), q.get("role", ""), self.server.roots))
            if u.path == "/api/jobs":
                with JOBS_LOCK:
                    js = [{k: v for k, v in j.items() if k != "proc"} for j in JOBS.values()]
                for j in js:
                    j["elapsed"] = int((j.get("ended") or time.time()) - j["started"])
                return self._send(200, {"jobs": sorted(js, key=lambda x: -x["started"])})
            if u.path == "/api/log":
                j = JOBS.get(q.get("job"))
                if not j:
                    return self._send(404, {"error": "no such job"})
                frm = int(q.get("from") or 0)
                data, size = "", 0
                if os.path.exists(j["log"]):
                    size = os.path.getsize(j["log"])
                    with open(j["log"], "rb") as fh:
                        fh.seek(min(frm, size))
                        data = fh.read(200000).decode("utf-8", "replace")
                return self._send(200, {"from": frm, "next": min(frm + len(data.encode()), size),
                                        "size": size, "text": data, "state": j["state"]})
            if u.path == "/api/outputs":
                j = JOBS.get(q.get("job"))
                if not j:
                    return self._send(404, {"error": "no such job"})
                step = json.load(open(os.path.join(j["dir"], "run_console_manifest.json")))["step"]
                return self._send(200, outputs_for(j["dir"], step))
            return self._send(404, {"error": "not found"})
        except PermissionError as e:
            return self._send(403, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self._body()
            if u.path == "/api/preflight":
                return self._send(200, preflight(b.get("config", {}), b.get("module") or self.server.module))
            if u.path == "/api/prepare":
                cfg = b.get("config", {})
                pf = preflight(cfg, b.get("module") or self.server.module)
                if pf["blocked"] and not b.get("force"):
                    return self._send(400, {"error": "preflight blocked", "preflight": pf})
                jid = time.strftime("job%Y%m%d-%H%M%S")
                jd = os.path.join(cfg["outdir"], jid)
                man = compose(cfg, b.get("module") or self.server.module, jd)
                return self._send(200, {"job": jid, "manifest": man, "preflight": pf})
            if u.path == "/api/run":
                jid, jd = b["job"], b["job_dir"]
                with JOBS_LOCK:
                    if any(j["state"] == "running" for j in JOBS.values()):
                        return self._send(409, {"error": "a job is already running; each holds ~8 GB"})
                j = launch(jid, jd, self.server.rootsys)
                return self._send(200, {"job": jid, "pid": j["pid"], "state": j["state"]})
            if u.path == "/api/stop":
                j = JOBS.get(b.get("job"))
                if not j:
                    return self._send(404, {"error": "no such job"})
                try:
                    os.killpg(os.getpgid(j["pid"]), signal.SIGTERM)
                except Exception as e:
                    return self._send(500, {"error": str(e)})
                return self._send(200, {"stopped": b["job"]})
            if u.path == "/api/reduce":
                j = JOBS.get(b.get("job"))
                if not j:
                    return self._send(404, {"error": "no such job"})
                step = json.load(open(os.path.join(j["dir"], "run_console_manifest.json")))["step"]
                dest = os.path.join(j["dir"], "explorer-data.json")
                return self._send(200, reduce_outputs(j["dir"], step, dest, b.get("allEpochs", True)))
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--module", default=REPO, help="alignment checkout to launch (2024 or 2025)")
    ap.add_argument("--roots", default="", help="comma-separated directories the file picker may enter")
    ap.add_argument("--rootsys", default=os.environ.get("ROOTSYS", ""))
    a = ap.parse_args()
    roots = [os.path.abspath(x) for x in a.roots.split(",") if x] or \
            [os.path.abspath(a.module), os.path.expanduser("~"), "/tmp"]
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.module, srv.roots, srv.rootsys = a.module, roots, a.rootsys
    mi = inspect_module(a.module)
    print(f"ITS2 run console  http://127.0.0.1:{a.port}")
    print(f"  module   {mi['root']}  ({'valid' if mi['valid'] else 'NOT a checkout'}, "
          f"{len(mi['knobs'])} knobs, {'O2 required' if mi['o2Required'] else 'runs without O2'})")
    print(f"  roots    {', '.join(roots)}")
    print(f"  ROOTSYS  {a.rootsys or '(inherited)'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
