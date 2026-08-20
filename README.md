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

## Two geometry backends

The module reads the detector geometry through `YDetectorGeometry`, which has two
interchangeable backends behind one unchanged public interface. Which one a job is built
with is a compile guard, `YGEOM_USE_O2`, so **both modes are the same tree built two
ways** — set it in the run console rather than by editing anything.

| `GEOM_BACKEND` | Reads geometry via | Needs O<sup>2</sup> at run time |
|---|---|---|
| **`o2`** *(default)* | `o2::its::GeometryTGeo`, as the module always has | yes — load it with `alienv` first |
| `cache` | a per-chip file built once by `tools/export_geometry_cache.C` | no |

`o2` is the default and the reference. `cache` exists for machines where O<sup>2</sup>
cannot be installed: the entire O<sup>2</sup> runtime surface turned out to be about
3.2 MB of static per-chip data — transforms and addressing — which the exporter writes
out once.

Because the two differ only by that guard, running one configuration under each and
comparing the outputs is how the cache gets checked against O<sup>2</sup>. That
comparison has not been done yet; see the limitations below.

To use the cache backend, build the cache once from the two committed inputs:

```sh
# with O2 loaded, this is all you need -- O2 supplies the AlignParam dictionary
root -l -b -q tools/export_geometry_cache.C   # writes geometry/its2_geom.root

# without O2, build a dictionary first from the alignment file's own StreamerInfo
root -l -b -q tools/make_alignlib.C
root -l -b -q tools/export_geometry_cache.C
```

The exporter reads `AlignParam` through ROOT reflection rather than by naming its
fields, because the real O<sup>2</sup> class keeps them private while the
MakeProject-built one makes them public. Reflection reports offsets for both, so the
same macro works either way and produces a byte-identical cache.

`export_geometry_cache.C` is the only macro here that uses TGeo, so it needs ROOT's
geometry component. Where ROOT is packaged in pieces — EPEL splits it into `root-core`,
`root-tree`, `root-geom` and so on — the core headers resolve and `TGeoManager.h` does
not. Check with:

```sh
root -l -b -q -e 'gSystem->Load("libGeom"); printf("%s\n", gSystem->Which(TROOT::GetIncludeDir(), "TGeoManager.h"))'
```

A null answer means the geometry package is missing and has to be installed.

---

## Running a job

Everything one run needs lives in `config/runconsole.conf`. Drive it from a terminal or
from a ROOT window — the window shells out to the same script, so anything it can do
works over ssh with no display.

```sh
eval `alienv load -w $O2_DIR/sw O2/latest`   # default backend is o2, so load it

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
| `GEOM_BACKEND` | `o2` | `config/runconsole.conf` |

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
