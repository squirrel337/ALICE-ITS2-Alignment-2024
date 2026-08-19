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

## Quick start

```sh
eval `alienv load -w $O2_DIR/sw O2/latest`   # only if this tree needs O2

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
| `runctl.sh reduce [OUT]` | Build an explorer payload from the run |
| `runctl.sh ui` | Open the ROOT window |

Exit status is 0 on success and non-zero when a check fails, so these
compose into a batch script without parsing their output.

## The window

`runctl.sh ui` opens a ROOT GUI with four tabs — Inputs, Module, Job, Run —
an action bar, and a log pane. Path fields have a **Browse…** button: files
use ROOT's file dialog, directories use the same `TBrowser`-style tree the
Manager uses.

The window never writes the configuration file, never composes and never
launches anything itself. Every button shells out to `runctl.sh`, so the
file format, the validation rules and the job layout live in one place and
the GUI cannot drift away from the command line. Whatever the window can
do, you can do over ssh with no display.

## What compose actually builds

`compose` creates `OUTPUT_DIR/JOB_TAG` and fills it with:

- a copy of `Ymlp/`, `monitor/`, `geometry/`, `NetworkParameters/`, `tools/`
  and the top-level macros
- **symlinks**, not copies, for the three big inputs — the data file alone
  is ~800 MB, and copying it per job would fill the disk for nothing
- the seed archive unpacked into `MLPTrain_Step<STEP-1>/`, which is where
  `run_train_circle.C` looks for `SetPrevUSL` and `SetPrevWeight`
- a regenerated `YMLPParallel.h` (it is exactly four defines) and an
  in-place patch of `nTrackMax`, `DET_MAG`, `Update_pTmin` and
  `Update_pTmax` in that job's own `DetectorConstant.h`

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
