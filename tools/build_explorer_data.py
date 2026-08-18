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
import json, re, sys, os
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
    step, epoch = None, None
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
        elif a in ("-h", "--help"):
            print(__doc__); print(parse_args.__doc__); sys.exit(0)
        else:
            sys.exit(f"unknown argument: {a}")
    if step is not None or epoch is not None:
        st = step if step is not None else os.path.dirname(os.path.dirname(RESMON))
        ep = epoch if epoch is not None else "-1"
        RESMON = os.path.join(st, "Residual", f"Residual_Monitor_Epoch_At_{ep}.root")
    return {"step": step, "epoch": epoch}


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

    # ---- residual profiles and histograms --------------------------------
    t = uproot.open(RESMON)["ResMonitor"]
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
    # vdcaX/vdcaY are already expressed relative to the beam center: GetCost builds the
    # circle in the beam-shifted frame (CircleXf = CircleXc - BeamCenter(0)) before
    # solving for the closest approach, so no further subtraction belongs here. vdcaZ is
    # absolute, hence the explicit vtxevtZ subtraction below.
    dx, dy = br("vdcaX"), br("vdcaY")
    trkphi = br("phi")
    # Signed transverse impact parameter, the component perpendicular to the track
    # direction. The sign is what makes a charge-dependent bias visible; |d0| folds it
    # away, which is exactly the asymmetry the cost's charge-symmetry term targets.
    d0xy = (-dx * np.sin(trkphi) + dy * np.cos(trkphi)) * 1e4        # cm -> um
    d0z = (br("vdcaZ") - br("vtxevtZ")) * 1e4
    charge = np.sign(br("cuvR"))     # curvature sign stands in for the track charge
    vz = br("vtxevtZ")

    DAX = {"phi": (-np.pi, np.pi, trkphi), "eta": (-1.5, 1.5, eta),
           "pT": (0.4, 2.0, pT), "vz": (-15.0, 15.0, vz)}
    # d0z is bimodal: a sharp core plus a nearly flat pedestal reaching several cm, so no
    # single axis shows both. The axis is set to the core and the pedestal is reported as
    # the dropped fraction, which states its size in one number instead of hiding it.
    DRNG = {"d0xy": 300.0, "d0z": 500.0}
    dca = {"ptCuts": PT_CUTS, "profiles": {}, "hists": {}, "charge": {},
           "n": int(d0xy.size)}
    for ci, cut in enumerate(PT_CUTS):
        sel = pT >= cut
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

    # ---- run summary from the log ----------------------------------------
    run = {"tracksInMonitor": int(len(pT))}
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

    prov = {
        "input":     os.path.basename(os.path.realpath("XXXXinput.root")) if os.path.exists("XXXXinput.root") else "?",
        "seed":      os.path.dirname(os.path.dirname(WSTEP)) or "?",
        "weights":   WDEPL,
        "geometry":  GEOM,
        "nTrackMax": grep_define("Ymlp/inc/DetectorConstant.h", "nTrackMax"),
        "nEPOCH":    grep_define("YMLPParallel.h", "nEPOCH"),
        "nDATA":     grep_define("YMLPParallel.h", "nDATA"),
        "epoch":     argsel["epoch"] if argsel["epoch"] is not None else "-1",
        "monitor":   RESMON,
        # GetCost(EDataSet) opens the monitor file "recreate" and Train() calls it once
        # for the training set and then once for the test set, so the second call
        # truncates the first. Verified on a live job: the file reached 16.9 MB during
        # the training pass and dropped to 0 when the test pass reopened it. What
        # survives on disk is the test pass, and every residual and DCA number in this
        # payload comes from it.
        "monitorSet": "test (holdout); the training pass is overwritten in the same file",
        "built":     __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    os.makedirs("docs", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"chips": chips, "residuals": res, "dca": dca, "run": run, "prov": prov,
                   "chipBoundary": CHIPBND, "nChips": n}, fh, separators=(",", ":"))
    print(f"wrote {OUT}  {os.path.getsize(OUT)/1e6:.2f} MB   chips={n}  tracks={len(pT)}")


if __name__ == "__main__":
    main()
