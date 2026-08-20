# Run console

Everything one training run needs is in `runconsole.conf`. Nothing else in
the tree has to be edited to change which data file is used, which module
tree is run, how big the job is, or where the output lands.

This replaces the earlier `tools/run_console.py`, which needed Python 3.8 —
CentOS 7 ships 2.7.5, so it could not run on the machine it was meant for.
Nothing here needs Python at all: ROOT reads the input file, and ROOT is
present by definition on a machine that can train.

The format follows
[ALICE-ITS2-ML-Alignment-Manager](https://github.com/squirrel337/ALICE-ITS2-ML-Alignment-Manager)
— one bash-sourced config, one library, one dispatcher, one ROOT window —
so the two trees are operated the same way.

`docs/run-console.html` is the same material as a page, if that reads better.

## Quick start

```sh
eval `alienv load -w $O2_DIR/sw O2/latest`   # default backend is o2, so load it

./config/runctl.sh ui         # set everything in one window
./config/runctl.sh doctor     # check this machine has what the run needs
./config/runctl.sh compose    # build the job directory
./config/runctl.sh run        # launch it
./config/runctl.sh log -f     # watch
```

## Command line

| Command | Does |
|---|---|
| `runctl.sh show` | Print every setting, plus what follows from it |
| `runctl.sh get KEY` | Print one value. `RC_*` reads a derived one |
| `runctl.sh set KEY=VALUE ...` | Change settings, then re-validate |
| `runctl.sh keys` | List every key |
| `runctl.sh validate` | Check types and relationships between values |
| `runctl.sh doctor` | Check inputs, disk, memory and ROOT on this machine |
| `runctl.sh compose` | Build `OUTPUT_DIR/JOB_TAG` from the module |
| `runctl.sh run` | Launch the composed job, detached |
| `runctl.sh status` | Is it running, and where is the log |
| `runctl.sh log [-f]` | Show or follow the run log |
| `runctl.sh stop` | Send TERM to the running job |
| `runctl.sh outputs` | List what the job produced |
| `runctl.sh ui` | Open the ROOT window |

Exit status is 0 on success and non-zero when a check fails, so these
compose into a batch script without parsing their output.

## The window

`runctl.sh ui` opens a ROOT GUI with six tabs — Inputs, Module, Job,
Learning, Selection, Run — an action bar, and a log pane. Path fields have a
**Browse…** button: files use ROOT's file dialog, directories use the same
`TBrowser`-style tree the Manager uses.

The window never writes the configuration file, never composes and never
launches anything itself. Every button shells out to `runctl.sh`, so the
file format, the validation rules and the job layout live in one place and
the GUI cannot drift away from the command line. Whatever the window can
do, you can do over ssh with no display.

## Geometry backend

`GEOM_BACKEND` picks how the job reads the detector geometry:

- **`o2`** (default) — through `o2::its::GeometryTGeo`, as the module always has. O2 must
  be loaded before the run; `doctor` checks `O2_ROOT` and says so if it is not.
- **`cache`** — from a per-chip file built once by `tools/export_geometry_cache.C`. No O2
  at run time. For machines where O2 cannot be installed.

It is a compile guard (`YGEOM_USE_O2`) written into the job's own copy of
`Ymlp/inc/YDetectorGeometry.h` at compose time, so the two modes are one tree built two
ways and the module checkout is still only ever read. Composing again switches cleanly
in either direction.

To compare the two, run one configuration under each and diff the outputs:

```sh
./config/runctl.sh set GEOM_BACKEND=o2    JOB_TAG=cmp-o2    && ./config/runctl.sh compose && ./config/runctl.sh run
./config/runctl.sh set GEOM_BACKEND=cache JOB_TAG=cmp-cache && ./config/runctl.sh compose && ./config/runctl.sh run
```

## Steps

`FIRST_STEP` and `N_STEPS` set a range. Step N reads `MLPTrain_Step<N-1>/`
and writes `MLPTrain_Step<N>/`, so the chain continues on its own: only the
first step needs the seed archive. Compose writes `run_steps.sh` into the job
directory and `run` launches that, detached; it rewrites `batch_train` per
step, files `train.log` into the step directory and tars the result, the same
way `process_all_train.sh` does.

If a step fails the run stops there rather than feeding a broken step into the
next one. `status` reads `run.step`, so it can answer "step 3 of 10" even
after the window has been closed and reopened.

## The learning rate

eta is not a `#define`. `YMultiLayerPerceptron.cxx` computes it in the
constructor from three plain globals:

```
eta = JOB_ETA_CONSTANT * JOB_ETA_SCALE * (JOB_ETA_DETRES * 1e-4)^2
```

so the console patches that **source file** in the job copy, not a header.
Nothing is rebuilt for it: `run_train_circle.C` includes `YAlignment.cxx`
directly, so cling reinterprets the source on every run.

**This is the starting value.** The module backtracks on its own: if the cost
rises above `training_Epast * 1.005` or `Init_training_E * 1.02`, it reloads
the previous epoch's weights and divides eta by `sqrt(10)`, up to four times.
The eta in the log will differ from the configured one whenever that has
happened — that is the module working, not the setting being ignored.

`JOB_VALID_WINDOW` sits beside it and does a related job: it clamps how much
one hit's residual may contribute to the update.

## Selection

Two pairs of chi thresholds, deliberately different: `JOB_CHI_IB` /
`JOB_CHI_OB` decide which hits spoil the **cost**, `JOB_CHI_IB_TRAIN` /
`JOB_CHI_OB_TRAIN` which hits spoil the **weight update**. `validate` refuses
a training threshold looser than its cost counterpart, since that would let
the update accept hits the cost has already rejected.

`JOB_IP_RANGE_R` and `JOB_IP_RANGE_Z` are the impact-parameter acceptance. In
this legacy tree they decide how much the layer-0 defect costs: a track whose
reference point was never written lands far outside and is dropped from both
the cost and the update. Widening them is the only lever available here, short
of patching the module.

## What compose actually builds

`compose` creates `OUTPUT_DIR/JOB_TAG` and fills it with:

- a copy of `Ymlp/`, `monitor/`, `geometry/`, `NetworkParameters/`, `tools/`
  and the top-level macros
- **symlinks**, not copies, for the three big inputs — the data file alone
  is ~800 MB, and copying it per job would fill the disk for nothing
- the seed archive unpacked into `MLPTrain_Step<STEP-1>/`, which is where
  `run_train_circle.C` looks for `SetPrevUSL` and `SetPrevWeight`
- a regenerated `YMLPParallel.h` (it is exactly four defines), an in-place
  patch of fourteen `#define`s in that job's own `DetectorConstant.h`, the
  four eta/clip globals in its `YMultiLayerPerceptron.cxx`, and the learning
  method and tree name in its `run_train_circle.C`
- `run_steps.sh`, the generated step loop

**The module checkout is only ever read.** That is what lets this drive a
tree that is meant to stay untouched, and it is worth keeping true.

## Things doctor checks because they cost time

- the input tree exists and holds at least `JOB_NDATA` entries
- the seed archive carries `weightsDU.txt`; without it the detector-unit
  normalisations stay uninitialised and the cost comes out `-nan` after a
  full run
- `STEP >= 1`; at step 0 the module hands `LoadUpdateSensorList` an empty
  name and errors out
- the geometry cache exists when the module is the cache-backed kind
- free disk, and free memory against the ~8 GB a job holds resident — two
  jobs at once have OOM-killed each other
- whether this module tree needs O2 at runtime, read from its own sources
  rather than assumed

It also prints the expected wall clock, from a model fitted on completed
runs of this module: `4.8 + 0.00337·nDATA + nEPOCH·0.01431·nDATA` minutes,
with the evaluation term at 75 % when `nEPOCH` is 0 because `Train()`
returns before the test pass.

## Driving another module version

Set `MODULE_DIR` to any alignment checkout. The console reads that tree's
own `YMLPParallel.h`, `DetectorConstant.h` and `YDetectorGeometry` rather
than assuming this one's, so it reports that tree's knob values and whether
it needs O2. Version independence is by discovery, not by a table.
