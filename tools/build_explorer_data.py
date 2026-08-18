#!/usr/bin/env python3
"""Reduce the alignment outputs to one JSON payload for docs/alignment-explorer.html.

    python3 tools/build_explorer_data.py

Reads the geometry cache, both weight sets, the occupancy list, the per-track residual
ntuple and the run log; writes docs/explorer-data.json. Needs uproot and numpy, not ROOT.

The residual ntuple is ~200 MB and 434k tracks, so nothing raw is shipped: residuals are
reduced here to profiles and histograms, binned once at the pT thresholds the module's
own monitor macros use. The same reduction is applied to the impact parameter, which the
monitor tree carries per track as vdca* and vtxevt*.
"""
import glob, json, re, sys, os
import numpy as np
import uproot

GEOM   = "geometry/its2_geom.root"
WDEPL  = "NetworkParameters/weights.txt"
WSTEP  = "MLPTrain_Step900/weights/weights.txt"
USL    = "UpdateSensorsList.txt"
RESMON = "MLPTrain_Step901/Residual/Residual_Monitor_Epoch_At_-1.root"
LOG    = "full.log"
OUT    = "docs/explorer-data.json"

PT_CUTS  = [0.0, 0.3, 0.5, 1.0, 2.0]      # same thresholds as monitor/batch_Residual
NPROF    = 40                              # bins per profile abscissa
NHIST    = 60                              # bins per residual histogram
NLAYER   = 7
CHIPBND  = [0, 108, 252, 432, 3120, 6480, 14712, 24120]


def load_weights(path):
    """ChipID -> [aR0, aR1, aR2, aT0, aT1, aT2] from an 18-column dump."""
    out = {}
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) == 18 and f[0].lstrip("-").isdigit():
                out[int(f[0])] = [float(x) for x in f[12:18]]
    return out


def load_occupancy(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) >= 2:
                out[int(f[0])] = int(f[1])
    return out


def profile(x, y, lo, hi, nbin):
    """Binned mean / RMS / count, with empty bins reported as null rather than zero."""
    edges = np.linspace(lo, hi, nbin + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbin - 1)
    keep = (x >= lo) & (x < hi)
    idx, y = idx[keep], y[keep]
    cnt = np.bincount(idx, minlength=nbin).astype(float)
    s1 = np.bincount(idx, weights=y, minlength=nbin)
    s2 = np.bincount(idx, weights=y * y, minlength=nbin)
    MINCNT = 50   # a bin below this is noise, not a measurement; reported as empty
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt >= MINCNT, s1 / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt >= MINCNT, s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
    rms = np.sqrt(np.maximum(var, 0))
    r = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in a]
    return {"edges": [round(float(e), 4) for e in edges],
            "mean": r(mean), "rms": r(rms), "n": [int(c) for c in cnt]}


def parse_args(argv):
    """Point the extractor at a different run without editing it.

        python3 tools/build_explorer_data.py --step-dir MLPTrain_Step902 --epoch 3

    A training run writes one Residual_Monitor_Epoch_At_N.root per epoch, so the epoch is
    part of choosing a dataset, not a detail. Unknown flags are refused rather than
    ignored: silently building the default payload under a name the caller did not ask
    for is worse than stopping.
    """
    global GEOM, WDEPL, WSTEP, USL, RESMON, LOG, OUT
    step, epoch, all_epochs = None, None, False
    i = 0
    while i < len(argv):
        a = argv[i]
        def val():
            if i + 1 >= len(argv):
                sys.exit(f"{a} needs a value")
            return argv[i + 1]
        if   a == "--step-dir": step = val(); i += 2
        elif a == "--epoch":    epoch = val(); i += 2
        elif a == "--resmon":   RESMON = val(); i += 2
        elif a == "--log":      LOG = val(); i += 2
        elif a == "--weights":  WDEPL = val(); i += 2
        elif a == "--seed-weights": WSTEP = val(); i += 2
        elif a == "--geometry": GEOM = val(); i += 2
        elif a == "--usl":      USL = val(); i += 2
        elif a == "--out":      OUT = val(); i += 2
        elif a == "--all-epochs": all_epochs = True; i += 1
        elif a in ("-h", "--help"):
            print(__doc__); print(parse_args.__doc__); sys.exit(0)
        else:
            sys.exit(f"unknown argument: {a}")
    if step is not None or epoch is not None:
        st = step if step is not None else os.path.dirname(os.path.dirname(RESMON))
        ep = epoch if epoch is not None else "-1"
        RESMON = os.path.join(st, "Residual", f"Residual_Monitor_Epoch_At_{ep}.root")
    return {"step": step, "epoch": epoch, "all_epochs": all_epochs}


def reduce_monitor(path):
    """Reduce one Residual_Monitor_Epoch_At_N.root to residual and DCA summaries.

    Split out of main() so the same reduction runs over every epoch a run produced,
    which is what makes epoch-to-epoch monitoring possible without a second code path
    that could drift from this one.
    """
    if True:
        # ---- residual profiles and histograms --------------------------------
        t = uproot.open(path)["ResMonitor"]
        br = lambda b: t["monitor/" + b].array(library="np")
        ds1, ds2, cid = br("fds1[8]"), br("fds2[8]"), br("fchipID[8]")
        gphi, gz = br("fgPhi[8]"), br("fgZ[8]")
        eta, pT = br("eta"), br("pT")

        AX = {"phi": (-np.pi, np.pi), "z": (-45.0, 45.0), "eta": (-1.5, 1.5), "pT": (0.4, 2.0)}
        res = {"ptCuts": PT_CUTS, "layers": {}}
        for l in range(NLAYER):
            hit = (cid[:, l] >= 0) & np.isfinite(ds1[:, l]) & np.isfinite(ds2[:, l])
            d1, d2 = ds1[hit, l] * 1e4, ds2[hit, l] * 1e4          # cm -> um
            ax = {"phi": gphi[hit, l], "z": gz[hit, l], "eta": eta[hit], "pT": pT[hit]}
            pt_hit = pT[hit]
            entry = {"profiles": {}, "hists": {}, "n": int(hit.sum())}
            for ci, cut in enumerate(PT_CUTS):
                sel = pt_hit >= cut
                for cname, dv in (("ds1", d1), ("ds2", d2)):
                    for aname, av in ax.items():
                        entry["profiles"].setdefault(f"{cname}|{aname}", []).append(
                            profile(av[sel], dv[sel], *AX[aname], NPROF))
                    rng = 300.0 if l < 3 else 600.0
                    vv = dv[sel]
                    inr = np.abs(vv) <= rng
                    # Out-of-range entries are DROPPED, not clipped. Clipping piles the
                    # overflow into the end bins and draws a spike that is not in the data.
                    h, e = np.histogram(vv[inr], bins=NHIST, range=(-rng, rng))
                    entry["hists"].setdefault(cname, []).append(
                        {"lo": -rng, "hi": rng, "counts": h.astype(int).tolist(),
                         "outside": round(float(np.mean(~inr) * 100), 2) if vv.size else None,
                         "median": round(float(np.median(vv)), 3) if vv.size else None,
                         "iqr": round(float(np.subtract(*np.percentile(vv, [75, 25]))), 3) if vv.size else None,
                         "tail300": round(float(np.mean(np.abs(vv) > 300) * 100), 2) if vv.size else None})
            res["layers"][str(l)] = entry

        # ---- impact parameter (DCA) ------------------------------------------
        # fip[0] and fip[1] are the impact parameters proper: YImpactParameter propagates the
        # helix to the beam in getImpactParams, and the same two numbers feed the selection
        # cut at YMultiLayerPerceptron.cxx:4656. The vdca* branches are a separate,
        # circle-geometry estimate and are not what the module calls the DCA.
        ip = t["monitor/fip[2]"].array(library="np")
        d0xy = ip[:, 0] * 1e4                                            # cm -> um
        d0z = ip[:, 1] * 1e4
        trkphi = br("phi")
        charge = np.sign(br("cuvR"))     # curvature sign stands in for the track charge
        vz = br("vtxevtZ")

        # The track is parametrized from layer 0 (YMultiLayerPerceptron.cxx:4623,
        # `vxyz = {proj_GXc[0], proj_GYc[0], proj_GZc[0]}`), but proj_G*c is an
        # uninitialized local that the fill loop at :4465 skips for layers with no hit. When
        # layer 0 is missing -- 51.6% of tracks in this sample -- vxyz reads slots that were
        # never written, and ip_z goes from a 29 um median to 16.5 mm. The split is carried
        # in the payload so the page can separate the two populations instead of averaging a
        # real measurement together with unwritten memory.
        hasL0 = cid[:, 0] >= 0

        DAX = {"phi": (-np.pi, np.pi, trkphi), "eta": (-1.5, 1.5, eta),
               "pT": (0.4, 2.0, pT), "vz": (-15.0, 15.0, vz)}
        DRNG = {"d0xy": 400.0, "d0z": 400.0}
        dca = {"ptCuts": PT_CUTS, "profiles": {}, "hists": {}, "charge": {},
               "n": int(d0xy.size), "nL0": int(hasL0.sum()),
               "fracL0": round(float(hasL0.mean()), 4), "l0Only": True}
        for ci, cut in enumerate(PT_CUTS):
            # Layer-0 tracks only. Including the rest would plot uninitialized memory.
            sel = (pT >= cut) & hasL0
            for cname, dv in (("d0xy", d0xy), ("d0z", d0z)):
                for aname, (lo, hi, av) in DAX.items():
                    dca["profiles"].setdefault(f"{cname}|{aname}", []).append(
                        profile(av[sel], dv[sel], lo, hi, NPROF))
                rng = DRNG[cname]
                vv = dv[sel]
                inr = np.abs(vv) <= rng
                h, e = np.histogram(vv[inr], bins=NHIST, range=(-rng, rng))
                dca["hists"].setdefault(cname, []).append(
                    {"lo": -rng, "hi": rng, "counts": h.astype(int).tolist(),
                     "outside": round(float(np.mean(~inr) * 100), 2) if vv.size else None,
                     "median": round(float(np.median(vv)), 3) if vv.size else None,
                     "iqr": round(float(np.subtract(*np.percentile(vv, [75, 25]))), 3) if vv.size else None})
            # Positive against negative, profiled in phi. A coherent split here is the
            # sagitta signature; the module already penalises it through fCostChargeSym.
            for q, tag in ((+1, "pos"), (-1, "neg")):
                m = sel & (charge == q)
                dca["charge"].setdefault(tag, []).append({
                    "n": int(m.sum()),
                    "meanD0xy": round(float(d0xy[m].mean()), 4) if m.any() else None,
                    "rmsD0xy": round(float(d0xy[m].std()), 4) if m.any() else None,
                    "profile": profile(trkphi[m], d0xy[m], -np.pi, np.pi, NPROF)})
        return res, dca, int(len(pT))


def cost_series(log):
    """Per-epoch fit / charge-symmetry / total for both samples, from the run log."""
    out = {}
    if not os.path.exists(log):
        return out
    pat = re.compile(r"COSTMONITOR\[(\w+)\] EPOCH(-?\d+) Fit \+ CHSYM = "
                     r"([\d.eE+-]+) \+ ([\d.eE+-]+) = ([\d.eE+-]+)")
    for ln in open(log, errors="ignore"):
        m = pat.search(ln)
        if not m:
            continue
        e = out.setdefault(m.group(2), {})
        e[m.group(1).lower()] = {"fit": float(m.group(3)), "chsym": float(m.group(4)),
                                 "total": float(m.group(5))}
    return out


def main():
    argsel = parse_args(sys.argv[1:])
    for p in (GEOM, WDEPL, USL, RESMON):
        if not os.path.exists(p):
            sys.exit(f"missing input: {p}")
    print(f"[extract] residuals from {RESMON}")

    # ---- geometry + parameters -------------------------------------------
    g = uproot.open(GEOM)["geom"].arrays(library="np")
    n = len(g["chipID"])
    wd, ws = load_weights(WDEPL), (load_weights(WSTEP) if os.path.exists(WSTEP) else {})
    occ = load_occupancy(USL)

    T = g["T"]
    chips = {
        "chipID": g["chipID"].astype(int).tolist(),
        "layer":  g["layer"].astype(int).tolist(),
        "stave":  g["stave"].astype(int).tolist(),
        "halfBarrel": g["halfBarrel"].astype(int).tolist(),
        "x": [round(float(v), 4) for v in T[:, 0]],
        "y": [round(float(v), 4) for v in T[:, 1]],
        "z": [round(float(v), 4) for v in T[:, 2]],
        "phi": [round(float(v), 5) for v in np.arctan2(T[:, 1], T[:, 0])],
        "r":   [round(float(v), 4) for v in np.hypot(T[:, 0], T[:, 1])],
        "ntracks": [occ.get(int(c), 0) for c in g["chipID"]],
    }
    # six DoF, rotations in urad and translations in um so the page never rescales
    for tag, table in (("depl", wd), ("step900", ws)):
        if not table:
            continue
        arr = np.array([table.get(int(c), [0] * 6) for c in g["chipID"]])
        for k in range(3):
            chips[f"{tag}_R{k}"] = [round(float(v) * 1e6, 2) for v in arr[:, k]]
            chips[f"{tag}_T{k}"] = [round(float(v) * 1e4, 2) for v in arr[:, 3 + k]]

    res, dca, n_tracks = reduce_monitor(RESMON)

    # Every epoch a run produced, reduced the same way, so the page can step through them.
    # Only the pT >= 0 slice is kept per epoch: the full five-cut detail is ~2 MB and six
    # epochs of it would not fit in a page, while the trend is what epoch monitoring is
    # for. The selected epoch still carries the full detail above.
    epochs = {}
    if argsel["all_epochs"]:
        d = os.path.dirname(RESMON)
        found = sorted(glob.glob(os.path.join(d, "Residual_Monitor_Epoch_At_*.root")),
                       key=lambda f: int(re.search(r"_At_(-?\d+)\.root$", f).group(1)))
        for f in found:
            ep = re.search(r"_At_(-?\d+)\.root$", f).group(1)
            r_e, d_e, n_e = reduce_monitor(f)
            slim_layers = {}
            for l, v in r_e["layers"].items():
                slim_layers[l] = {
                    "n": v["n"],
                    "profiles": {k: [vv[0]] for k, vv in v["profiles"].items()},
                    "hists": {k: [vv[0]] for k, vv in v["hists"].items()}}
            epochs[ep] = {
                "n": n_e,
                "residuals": {"ptCuts": [PT_CUTS[0]], "layers": slim_layers},
                "dca": {"ptCuts": [PT_CUTS[0]],
                        "profiles": {k: [v[0]] for k, v in d_e["profiles"].items()},
                        "hists": {k: [v[0]] for k, v in d_e["hists"].items()},
                        "charge": {k: [v[0]] for k, v in d_e["charge"].items()},
                        "n": d_e["n"], "nL0": d_e["nL0"], "fracL0": d_e["fracL0"],
                        "l0Only": True}}
            print(f"[extract]   epoch {ep:>3}: {n_e} tracks")

    # ---- run summary from the log ----------------------------------------
    run = {"tracksInMonitor": n_tracks}
    if os.path.exists(LOG):
        txt = open(LOG, errors="ignore").read()
        m = re.search(r"COSTMONITOR\[TRAINING\] EPOCH-1 Fit \+ CHSYM = ([\d.eE+-]+) \+ ([\d.eE+-]+) = ([\d.eE+-]+)", txt)
        if m:
            run |= {"costFit": float(m.group(1)), "costChSym": float(m.group(2)), "costTotal": float(m.group(3))}
        m = re.search(r"Using (\d+) train and (\d+) test events", txt)
        if m:
            run |= {"trainEvents": int(m.group(1)), "testEvents": int(m.group(2))}
        m = re.search(r"Using (\d+) train and (\d+) test tracks", txt)
        if m:
            run |= {"trainTracks": int(m.group(1)), "testTracks": int(m.group(2))}

    # Provenance travels with the payload. The page reads its configuration table from
    # here rather than hard-coding it, so a swapped dataset cannot be described by the
    # previous run's settings.
    def grep_define(path, name, default="?"):
        try:
            for ln in open(path):
                m = re.match(r"\s*#define\s+" + name + r"\s+(\S+)", ln)
                if m:
                    return m.group(1)
        except OSError:
            pass
        return default

    def monitor_set(nepoch):
        try:
            n = int(nepoch)
        except (TypeError, ValueError):
            return "unknown (could not read nEPOCH)"
        return ("training; at nEPOCH = 0 Train() returns before the test pass runs"
                if n == 0 else
                "test (holdout); the training pass is overwritten in the same file")

    # Read the configuration from the run that produced the monitor file, not from the
    # checkout this script happens to be run in. RESMON is <run>/<step>/Residual/<file>,
    # so the run root is three levels up. Getting this wrong labels a payload with
    # whatever the local tree currently says, which is exactly what prov exists to stop.
    run_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(RESMON))))
    if not os.path.exists(os.path.join(run_root, "YMLPParallel.h")):
        run_root = "."
    par = os.path.join(run_root, "YMLPParallel.h")
    det = os.path.join(run_root, "Ymlp", "inc", "DetectorConstant.h")
    src = os.path.join(run_root, "XXXXinput.root")

    prov = {
        "input":     os.path.basename(os.path.realpath(src)) if os.path.exists(src) else "?",
        "seed":      os.path.dirname(os.path.dirname(WSTEP)) or "?",
        "runRoot":   os.path.basename(os.path.abspath(run_root)) if run_root != "." else ".",
        "weights":   WDEPL,
        "geometry":  GEOM,
        "nTrackMax": grep_define(det, "nTrackMax"),
        "DET_MAG":   grep_define(det, "DET_MAG"),
        "nEPOCH":    grep_define(par, "nEPOCH"),
        "nDATA":     grep_define(par, "nDATA"),
        "epoch":     argsel["epoch"] if argsel["epoch"] is not None else "-1",
        "monitor":   RESMON,
        # Which pass survives in the monitor file depends on nEPOCH, so it is derived
        # rather than assumed. GetCost(EDataSet) opens the file "recreate", and Train()
        # runs the training pass, then hits `if(nEpoch==0) return;`
        # (YMultiLayerPerceptron.cxx:1279), then the test pass. At nEPOCH = 0 the
        # function returns before the test pass ever runs, so the training pass is what
        # is on disk. Above zero the test pass reopens the file and truncates it.
        # Both readings check out against the pT-window acceptance, which is ~36% in
        # every run: 434569/1183248 training at nEPOCH 0, and 11497/31488 and 1459/4080
        # test at nEPOCH 5.
        "monitorSet": monitor_set(grep_define(par, "nEPOCH")),
        "built":     __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    os.makedirs("docs", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"chips": chips, "residuals": res, "dca": dca, "run": run, "prov": prov,
                   "epochs": epochs, "costSeries": cost_series(LOG),
                   "chipBoundary": CHIPBND, "nChips": n}, fh, separators=(",", ":"))
    print(f"wrote {OUT}  {os.path.getsize(OUT)/1e6:.2f} MB   chips={n}  tracks={n_tracks}")


if __name__ == "__main__":
    main()
