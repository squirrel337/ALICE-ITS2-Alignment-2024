# ALICE ITS2 — ML-based Alignment

Self-supervised neural alignment of the ALICE Inner Tracking System 2.

Each of the **24,120 ALPIDE sensors** carries its own small correction network. There is
no external alignment reference: the training signal is the track residual itself — a
circle fit in *r*–φ together with a *z*–β line fit — so the detector is aligned against
its own tracks.

Per sensor the module fits **11 network parameters** (5 neuron biases and 6 synapse
weights), from which **6 rigid-body degrees of freedom** are derived: three rotations
`Ralpha, Rbeta, Rgamma` and three translations `T1, T2, T3`.

| | |
|---|---|
| Layers | 7 — chip boundaries `0, 108, 252, 432, 3120, 6480, 14712, 24120` |
| Staves per layer | `12, 16, 20, 24, 30, 42, 48` |
| Sensors | 24,120 |
| Runtime dependency | **ROOT only** — see below |

---

## It runs on ROOT alone

This module used to require a full O<sup>2</sup> environment *at runtime*, not just to
build: `batch_train` loads `YDetectorGeometry.cxx` through cling, so O<sup>2</sup>
headers and libraries had to resolve while the job ran.

That is no longer true. The entire O<sup>2</sup> surface turned out to be about 3.2 MB
of static per-chip data — transforms and addressing — which is now exported once into a
cache file. `YDetectorGeometry` has two interchangeable backends behind one unchanged
public interface:

| Backend | When |
|---|---|
| geometry cache *(default)* | ROOT only; no O<sup>2</sup>, no CVMFS, no `alienv` |
| `YGEOM_USE_O2` | the original path, kept as the reference the cache is validated against |

Every `o2::` reference in the tree is inside that guard or a comment. Build the cache
once from the two committed inputs:

```sh
root -l -b -q tools/make_alignlib.C           # once, rebuilds AlignParam from the file's own StreamerInfo
root -l -b -q tools/export_geometry_cache.C   # writes geometry/its2_geom.root
```

---

## Running a job

Everything one run needs lives in `config/runconsole.conf`. Drive it from a terminal or
from a ROOT window — the window shells out to the same script, so anything it can do
works over ssh with no display.

```sh
eval `alienv load -w $O2_DIR/sw O2/latest`   # only if this tree needs O2

./config/runctl.sh ui         # set everything in one window
./config/runctl.sh doctor     # check this machine has what the run needs
./config/runctl.sh compose    # build the job directory
./config/runctl.sh run        # launch it
./config/runctl.sh log -f     # watch
```

`compose` builds a self-contained job directory and patches the knobs into **that job's**
copies of the headers and sources. The module checkout is only ever read.

Full reference: **[`config/README.md`](config/README.md)** and
**[`docs/run-console.html`](docs/run-console.html)**.

### Current configuration

| Knob | Value | File |
|---|---|---|
| `nDATA` | 4000 | `YMLPParallel.h` |
| `nEPOCH` | 5 | `YMLPParallel.h` |
| `nTrackMax` | 8 | `Ymlp/inc/DetectorConstant.h` |
| `DET_MAG` | −0.5 T | `Ymlp/inc/DetectorConstant.h` |
| `FITMODEL` | 2 (circle) | `Ymlp/inc/DetectorConstant.h` |

---

## Layout

| Path | What |
|---|---|
| `Ymlp/` | The module. `YMultiLayerPerceptron.cxx` is the core (~13.8k lines), `YAlignment.cxx` the driver |
| `run_train_circle.C` | Entry point — `batch_train` loads the geometry, then runs this for one step |
| `config/` | Run console: one configuration file, a library, a dispatcher |
| `tools/ConfigUI/` | The ROOT window |
| `tools/` | Geometry cache export, and a tree-entry helper |
| `monitor/` | Post-training monitoring macros |
| `docs/` | [Workflow](docs/workflow.html) · [Run console](docs/run-console.html) |
| `NetworkParameters/` | Deployed weights |

**There is no compiled build.** `Ymlp/CMakeLists.txt` is 534 null bytes; the module is
cling-JIT only, loaded from source on every run. That is also what lets the console
patch a job's `.cxx` without rebuilding anything.

---

## Known limitations

These are measured, not hypothetical, and any number produced from this tree carries
them.

**The layer-0 defect.** When a track has no layer-0 hit, `vxyz` is read from three stack
slots that were never written, so its impact parameter is meaningless. That is not a
corner case: in run 539884, **51.6 % of tracks have no L0 hit**, and **90.85 % of those**
are pushed outside `RANGE_IMPACTPARAMS_*` and dropped from *both* the cost and the weight
update — **47.33 % of the whole sample**, selected by leftover stack contents rather than
by physics.

> Fixed in `ALICE-ITS2-Alignment-2025`, which projects the track to the vertex β and
> searches for the innermost layer that actually has a hit. **Deliberately not backported
> here:** this tree is legacy, and patching it would invalidate every run already recorded
> against it.

**The geometry cache's alignment-delta convention is unvalidated.** `DeltaMatrix()` in
`tools/export_geometry_cache.C` is the one thing there not read from a file — it assumes
an extrinsic `Rz(phi)Ry(theta)Rx(psi)` composition. Cache integrity and round-trip checks
pass; the comparison against an O<sup>2</sup> reference has never been run.

**`Angle2Alpha` and `kB2C` are reconstructed.** `Ymlp/inc/YO2Compat.h` stands in for a
handful of O<sup>2</sup> header-only helpers, and these two were rebuilt from the
conventions rather than copied from O<sup>2</sup> source. They are not cosmetic: they feed
the impact-parameter cut, so getting either wrong changes *which tracks train*. Both are
isolated in that one header, so a correction is a one-line change.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
