#!/bin/bash
# ==========================================================================
#  runconf.sh -- load, validate and expand the run-console configuration
# ==========================================================================
#  Sourced by runctl.sh. Not meant to be run directly.
#
#  Targets bash 4.2 (CentOS 7): no namerefs, no ${var@Q}, no associative
#  arrays. ROOT is assumed present -- it is what reads the input tree, so
#  there is no Python dependency anywhere in this file.
# ==========================================================================

# Every key the configuration understands, in display order, as
# NAME:TYPE:GROUP. TYPE is one of dir, path, name, int, num.
RC_KEYS="
DATA_FILE:path:inputs
DATA_TREE:name:inputs
GEOM_FILE:path:inputs
ALIGN_FILE:path:inputs
PARAMS_ARCHIVE:path:inputs
MODULE_DIR:dir:module
FIRST_STEP:int:module
N_STEPS:int:module
GEOM_BACKEND:name:module
JOB_NDATA:int:size
JOB_NEPOCH:int:size
JOB_NCORE:int:size
JOB_JPARALLEL:int:size
JOB_NTRACKMAX:int:model
JOB_DET_MAG:num:model
JOB_FITMODEL:int:model
JOB_VERTEXFIT:name:model
JOB_LEARNING_METHOD:name:model
JOB_ETA_CONSTANT:num:learning
JOB_ETA_SCALE:num:learning
JOB_ETA_DETRES:num:learning
JOB_VALID_WINDOW:num:learning
JOB_PT_MIN:num:selection
JOB_PT_MAX:num:selection
JOB_CHI_IB:num:selection
JOB_CHI_OB:num:selection
JOB_CHI_IB_TRAIN:num:selection
JOB_CHI_OB_TRAIN:num:selection
JOB_TRACK_REJECT:num:selection
JOB_IP_RANGE_R:num:selection
JOB_IP_RANGE_Z:num:selection
JOB_MIN_CLUSTER:int:selection
OUTPUT_DIR:dir:output
JOB_TAG:name:output
ROOTSYS_OVERRIDE:dir:env
"

rc_keys()  { echo "$RC_KEYS" | awk -F: 'NF{print $1}'; }
rc_type()  { echo "$RC_KEYS" | awk -F: -v k="$1" '$1==k{print $2}'; }
rc_group() { echo "$RC_KEYS" | awk -F: -v k="$1" '$1==k{print $3}'; }

rc_die()   { echo "runconf: $*" >&2; return 1; }

# Which header each job knob is patched into. DetectorConstant.h carries far
# more than these, so it is edited in place; YMLPParallel.h is exactly four
# defines and is regenerated whole.
RC_DETCONST="Ymlp/inc/DetectorConstant.h"
# eta and the gradient clip are plain globals in a source file, not macros,
# so the job's own copy of this .cxx is patched too. That costs nothing at
# run time: run_train_circle.C includes YAlignment.cxx directly, so cling
# reinterprets the source on every run and there is nothing to rebuild.
RC_MLPSRC="Ymlp/src/YMultiLayerPerceptron.cxx"
# The learning method and the source tree name are arguments in the driver.
RC_DRIVER="run_train_circle.C"
# Backend selection is a compile guard in the module's own header, so the two modes
# are one tree built two ways -- which is what makes an O2-vs-cache comparison mean
# anything. compose writes the guard into the job's copy.
RC_GEOMHDR="Ymlp/inc/YDetectorGeometry.h"
RC_GEOM_MARK="// --- run console: geometry backend ---"

# The learning methods YMultiLayerPerceptron.h:505 declares.
RC_METHODS="kStochastic kBatch kBatchDetectorUnitUser kSteepestDescent kRibierePolak kFletcherReeves kBFGS kOffsetTuneByMean"

# --- location -------------------------------------------------------------

# Repository root, derived from this file's own location so the scripts work
# from any working directory.
rc_root() {
  local here
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || return 1
  cd "$here/.." && pwd
}

rc_conf_file() { echo "${RUN_CONF:-$(rc_root)/config/runconsole.conf}"; }

# --- load -----------------------------------------------------------------

# Sources the configuration after refusing anything that is not a plain
# assignment, so a stray command in the file cannot execute.
rc_load() {
  local f offenders
  f=$(rc_conf_file)
  [ -r "$f" ] || { rc_die "cannot read $f"; return 1; }

  offenders=$(grep -nvE '^[[:space:]]*(#|$)|^[A-Z_][A-Z0-9_]*=|^[[:space:]]+[^=]*$' "$f")
  if [ -n "$offenders" ]; then
    rc_die "refusing to source $f -- these lines are not assignments:"
    echo "$offenders" >&2
    return 1
  fi

  # shellcheck disable=SC1090
  . "$f" || { rc_die "failed to source $f"; return 1; }
  RC_CONF_LOADED=1
  rc_derive
}

# --- derived values -------------------------------------------------------

rc_derive() {
  RC_ROOT=$(rc_root)

  # An empty MODULE_DIR means this repository, so the common case needs no
  # absolute path in the file.
  if [ -n "$MODULE_DIR" ]; then RC_MODULE="$MODULE_DIR"; else RC_MODULE="$RC_ROOT"; fi

  # A relative OUTPUT_DIR is relative to the repository, not to wherever the
  # caller happens to stand.
  case "$OUTPUT_DIR" in
    /*) RC_OUTPUT="$OUTPUT_DIR" ;;
    *)  RC_OUTPUT="$RC_ROOT/$OUTPUT_DIR" ;;
  esac

  RC_JOB_DIR="$RC_OUTPUT/$JOB_TAG"
  RC_SEED_STEP=$(( FIRST_STEP - 1 ))
  RC_LAST_STEP=$(( FIRST_STEP + N_STEPS - 1 ))
  RC_SEED_DIR="$RC_JOB_DIR/MLPTrain_Step${RC_SEED_STEP}"
  RC_LOG="$RC_JOB_DIR/run.log"
  RC_MANIFEST="$RC_JOB_DIR/run_console_manifest.txt"
  RC_RUNNER="$RC_JOB_DIR/run_steps.sh"
  # The loop writes the step it is on here, so status can answer "3 of 10"
  # after the window has been closed and reopened.
  RC_PROGRESS="$RC_JOB_DIR/run.step"

  rc_inspect_module

  # eta as the module computes it, so the window never shows a number the
  # run will not use.
  RC_ETA=$(awk -v c="$JOB_ETA_CONSTANT" -v s="$JOB_ETA_SCALE" -v d="$JOB_ETA_DETRES" \
             'BEGIN{ printf "%.6g", c * s * (d*1e-4)^2 }')

  RC_EST_STEP=$(rc_estimate "$JOB_NDATA" "$JOB_NEPOCH")
  RC_EST_MIN=$(awk -v m="$RC_EST_STEP" -v n="$N_STEPS" 'BEGIN{printf "%.0f", m*n}')
}

# --- module inspection ----------------------------------------------------

# The first value of one #define in a header, or empty.
rc_read_define() {
  [ -r "$1" ] || return 0
  awk -v n="$2" '$1=="#define" && $2==n {print $3; exit}' "$1"
}

# What this module version offers, discovered rather than assumed, so the
# console can drive a 2025 tree as well as this one.
rc_inspect_module() {
  local par det geo_h geo_c src
  par="$RC_MODULE/YMLPParallel.h"
  det="$RC_MODULE/$RC_DETCONST"
  geo_h="$RC_MODULE/Ymlp/inc/YDetectorGeometry.h"
  geo_c="$RC_MODULE/Ymlp/src/YDetectorGeometry.cxx"

  RC_MODULE_VALID=0
  if [ -f "$par" ] && [ -f "$det" ] && [ -f "$geo_h" ] && [ -f "$RC_MODULE/run_train_circle.C" ]; then
    RC_MODULE_VALID=1
  fi

  # A cache-backed geometry runs without O2; an O2-only one cannot, and
  # saying so up front beats a compile error twenty minutes in.
  src=$(cat "$geo_h" "$geo_c" 2>/dev/null)
  RC_CACHE_CAPABLE=0
  case "$src" in *LoadCache*|*YGEOM_CACHE*) RC_CACHE_CAPABLE=1 ;; esac
  RC_O2_REQUIRED=0
  if [ "$RC_CACHE_CAPABLE" -eq 0 ]; then
    case "$src" in *ITSBase/GeometryTGeo.h*) RC_O2_REQUIRED=1 ;; esac
  fi

  # What the module currently holds, for the GUI to show beside what the
  # configuration would set.
  RC_MOD_NDATA=$(rc_read_define "$par" nDATA)
  RC_MOD_NEPOCH=$(rc_read_define "$par" nEPOCH)
  RC_MOD_NTRACKMAX=$(rc_read_define "$det" nTrackMax)
  RC_MOD_DET_MAG=$(rc_read_define "$det" DET_MAG)
}

# --- runtime estimate -----------------------------------------------------

# Fitted on completed runs of this module. At nEPOCH 0 Train() returns before
# the test pass (YMultiLayerPerceptron.cxx:1279), so the epoch -1 cost is the
# training pass alone -- about 75% of the work. Ignoring that over-predicted a
# 50k evaluation by 55%.
RC_COST_FIXED="4.8"
RC_COST_EVAL="0.00337"
RC_COST_EPOCH="0.01431"

rc_estimate() {   # ndata nepoch -> minutes
  awk -v n="$1" -v e="$2" -v a="$RC_COST_FIXED" -v b="$RC_COST_EVAL" -v c="$RC_COST_EPOCH" \
    'BEGIN{ if (n=="" || e=="") { print ""; exit }
            s = (e <= 0) ? 0.75 : 1.0;
            if (e < 0) e = 0;
            printf "%.0f", a + b*s*n + e*c*n }'
}

# --- validation -----------------------------------------------------------

# Structural checks only: types and relationships between values. Whether the
# paths exist on this machine is rc_doctor's job.
rc_validate() {
  local k t v bad=0

  for k in $(rc_keys); do
    t=$(rc_type "$k")
    eval "v=\$$k"
    case "$t" in
      int)
        if ! echo "$v" | grep -qE '^[0-9]+$'; then
          echo "  $k: expected a whole number, got '$v'" >&2; bad=1
        fi ;;
      num)
        # Scientific notation is accepted: JOB_ETA_SCALE ships as 1.0e-7.
        if ! echo "$v" | grep -qE '^-?[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$'; then
          echo "  $k: expected a number, got '$v'" >&2; bad=1
        fi ;;
    esac
    case "$t" in
      dir|path|name|num)
        case "$v" in
          *'"'*)        echo "  $k: must not contain a double quote" >&2; bad=1 ;;
          *'$('*|*'`'*) echo "  $k: must not contain command substitution" >&2; bad=1 ;;
        esac ;;
    esac
  done
  [ $bad -eq 0 ] || return 1

  # Step 0 hands LoadUpdateSensorList an empty name and errors out, so the
  # seed always lands in a real MLPTrain_Step<FIRST_STEP-1>.
  [ "$FIRST_STEP" -ge 1 ] || { echo "  FIRST_STEP must be at least 1; at step 0 LoadUpdateSensorList gets an empty name" >&2; bad=1; }
  [ "$N_STEPS" -ge 1 ]    || { echo "  N_STEPS must be at least 1" >&2; bad=1; }

  case "$GEOM_BACKEND" in
    o2) ;;
    cache)
      [ "$RC_CACHE_CAPABLE" -eq 1 ] || {
        echo "  GEOM_BACKEND is cache, but $RC_MODULE has no cache backend; use o2" >&2; bad=1; } ;;
    *) echo "  GEOM_BACKEND must be o2 or cache, got '$GEOM_BACKEND'" >&2; bad=1 ;;
  esac

  case "$JOB_FITMODEL" in
    1|2) ;;
    *) echo "  JOB_FITMODEL must be 1 (Line) or 2 (Circle), got '$JOB_FITMODEL'" >&2; bad=1 ;;
  esac
  case "$JOB_VERTEXFIT" in
    kTRUE|kFALSE) ;;
    *) echo "  JOB_VERTEXFIT must be kTRUE or kFALSE, got '$JOB_VERTEXFIT'" >&2; bad=1 ;;
  esac
  case " $RC_METHODS " in
    *" $JOB_LEARNING_METHOD "*) ;;
    *) echo "  JOB_LEARNING_METHOD '$JOB_LEARNING_METHOD' is not one of: $RC_METHODS" >&2; bad=1 ;;
  esac

  # A zero or negative learning rate would train nothing while looking busy.
  awk -v e="$RC_ETA" 'BEGIN{exit !(e+0 > 0)}' || {
    echo "  the eta constants give $RC_ETA; it must be above zero" >&2; bad=1; }
  awk -v w="$JOB_VALID_WINDOW" 'BEGIN{exit !(w+0 > 0)}' || {
    echo "  JOB_VALID_WINDOW must be above zero" >&2; bad=1; }

  # The training thresholds are meant to be tighter than the cost ones. This
  # is a warning in spirit, but the pair is easy to swap by accident.
  awk -v a="$JOB_CHI_IB_TRAIN" -v b="$JOB_CHI_IB" 'BEGIN{exit !(a+0 > b+0)}' && {
    echo "  JOB_CHI_IB_TRAIN ($JOB_CHI_IB_TRAIN) is looser than JOB_CHI_IB ($JOB_CHI_IB); the update would accept hits the cost rejects" >&2; bad=1; }
  awk -v a="$JOB_CHI_OB_TRAIN" -v b="$JOB_CHI_OB" 'BEGIN{exit !(a+0 > b+0)}' && {
    echo "  JOB_CHI_OB_TRAIN ($JOB_CHI_OB_TRAIN) is looser than JOB_CHI_OB ($JOB_CHI_OB)" >&2; bad=1; }

  for k in JOB_TRACK_REJECT JOB_IP_RANGE_R JOB_IP_RANGE_Z JOB_CHI_IB JOB_CHI_OB \
           JOB_CHI_IB_TRAIN JOB_CHI_OB_TRAIN JOB_ETA_DETRES; do
    eval "v=\$$k"
    awk -v x="$v" 'BEGIN{exit !(x+0 > 0)}' || { echo "  $k must be above zero, got '$v'" >&2; bad=1; }
  done
  [ "$JOB_MIN_CLUSTER" -ge 0 ] || { echo "  JOB_MIN_CLUSTER must not be negative" >&2; bad=1; }

  [ "$JOB_NDATA" -ge 1 ]     || { echo "  JOB_NDATA must be at least 1" >&2; bad=1; }
  [ "$JOB_NCORE" -ge 1 ]     || { echo "  JOB_NCORE must be at least 1" >&2; bad=1; }
  [ "$JOB_NTRACKMAX" -ge 2 ] || { echo "  JOB_NTRACKMAX must be at least 2" >&2; bad=1; }
  [ -n "$JOB_TAG" ]          || { echo "  JOB_TAG is empty; it names the job directory" >&2; bad=1; }
  [ -n "$DATA_TREE" ]        || { echo "  DATA_TREE is empty" >&2; bad=1; }

  awk -v lo="$JOB_PT_MIN" -v hi="$JOB_PT_MAX" 'BEGIN{exit !(lo+0 < hi+0)}' || {
    echo "  JOB_PT_MIN ($JOB_PT_MIN) must be below JOB_PT_MAX ($JOB_PT_MAX)" >&2; bad=1; }

  # DET_MAG 0 would divide by zero in GetSigma and make every pT zero.
  awk -v b="$JOB_DET_MAG" 'BEGIN{exit !(b+0 == 0)}' && {
    echo "  JOB_DET_MAG must not be zero" >&2; bad=1; }

  return $bad
}

# --- generators -----------------------------------------------------------

_rc_banner() {
  echo "// Generated by config/runctl.sh from config/runconsole.conf."
  echo "// Edits here are overwritten on the next compose; change the"
  echo "// configuration instead."
}

# YMLPParallel.h is exactly four defines, so it is written whole.
rc_gen_ymlpparallel() {
  local out="$1"
  { _rc_banner
    echo "#define jparallel $JOB_JPARALLEL"
    echo "#define nCORE $JOB_NCORE"
    echo "#define nDATA $JOB_NDATA"
    echo "#define nEPOCH $JOB_NEPOCH"
  } > "$out"
}

# DetectorConstant.h carries far more than the knobs, so its values are
# rewritten in place. The leading spacing and any trailing comment on the
# line are preserved, which keeps the diff against the module readable.
rc_patch_define() {   # file name value
  local f="$1" n="$2" v="$3"
  [ -f "$f" ] || return 0
  sed -i "s|^\([[:space:]]*#define[[:space:]][[:space:]]*$n[[:space:]][[:space:]]*\)[^[:space:]][^[:space:]]*|\1$v|" "$f"
}

rc_patch_detconst() {
  local f="$1"
  rc_patch_define "$f" nTrackMax             "$JOB_NTRACKMAX"
  rc_patch_define "$f" DET_MAG               "$JOB_DET_MAG"
  rc_patch_define "$f" FITMODEL              "$JOB_FITMODEL"
  rc_patch_define "$f" VERTEXFIT             "$JOB_VERTEXFIT"
  rc_patch_define "$f" Update_pTmin          "$JOB_PT_MIN"
  rc_patch_define "$f" Update_pTmax          "$JOB_PT_MAX"
  rc_patch_define "$f" RANGE_CHI_IB          "$JOB_CHI_IB"
  rc_patch_define "$f" RANGE_CHI_OB          "$JOB_CHI_OB"
  rc_patch_define "$f" RANGE_CHI_IB_TRAINING "$JOB_CHI_IB_TRAIN"
  rc_patch_define "$f" RANGE_CHI_OB_TRAINING "$JOB_CHI_OB_TRAIN"
  rc_patch_define "$f" TrackRejection        "$JOB_TRACK_REJECT"
  rc_patch_define "$f" RANGE_IMPACTPARAMS_R  "$JOB_IP_RANGE_R"
  rc_patch_define "$f" RANGE_IMPACTPARAMS_Z  "$JOB_IP_RANGE_Z"
  rc_patch_define "$f" Min_Cluster_by_Sensor "$JOB_MIN_CLUSTER"
}

# One file-scope `double NAME = value;` in a source file. Anchored at the
# start of the line so the commented-out duplicates a few lines below
# (`//double UpdateConstant = 4.0;`) are left alone.
rc_patch_global() {   # file name value
  local f="$1" n="$2" v="$3"
  [ -f "$f" ] || return 0
  sed -i "s|^\(double[[:space:]][[:space:]]*$n[[:space:]]*=[[:space:]]*\)[^;]*;|\1$v;|" "$f"
}

rc_patch_mlpsrc() {
  local f="$1"
  rc_patch_global "$f" UpdateConstant "$JOB_ETA_CONSTANT"
  rc_patch_global "$f" UpdateScale    "$JOB_ETA_SCALE"
  rc_patch_global "$f" DETRES         "$JOB_ETA_DETRES"
  rc_patch_global "$f" ValidWindow    "$JOB_VALID_WINDOW"
}

# Writes or removes the YGEOM_USE_O2 guard in the job's own header. Idempotent: any
# marker block a previous compose left is dropped first, so switching modes back and
# forth does not accumulate defines.
rc_patch_geom() {   # file
  local f="$1" tmp
  [ -f "$f" ] || return 0
  tmp=$(mktemp "${TMPDIR:-/tmp}/runconf.XXXXXX") || return 1
  awk -v want="$GEOM_BACKEND" -v mark="$RC_GEOM_MARK" '
    $0 == mark { drop = 1; next }
    drop == 1  { drop = 0; next }
    /^#ifdef YGEOM_USE_O2$/ && !done {
      if (want == "o2") { print mark; print "#define YGEOM_USE_O2 1" }
      done = 1
    }
    { print }
  ' "$f" > "$tmp" && cat "$tmp" > "$f"
  rm -f "$tmp"
}

# The driver names the learning method and the source tree. Patching it is
# what makes DATA_TREE mean anything: before this the key was checked by
# doctor and then ignored, because SetSourceTreeName was hardcoded.
rc_patch_driver() {
  local f="$1"
  [ -f "$f" ] || return 0
  sed -i "s|\(ELearningMethod method = YMultiLayerPerceptron::\)k[A-Za-z]*|\1$JOB_LEARNING_METHOD|" "$f"
  sed -i "s|\(SetSourceTreeName(\"\)[^\"]*\(\")\)|\1$DATA_TREE\2|" "$f"
}

# --- environment checks ---------------------------------------------------

_rc_ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
_rc_warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; RC_DOCTOR_WARN=$((RC_DOCTOR_WARN+1)); }
_rc_bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; RC_DOCTOR_FAIL=$((RC_DOCTOR_FAIL+1)); }

# Entries in one tree of a ROOT file, via ROOT itself. This is what replaces
# the uproot dependency the Python console carried: on the target machine
# ROOT is present by definition, so nothing else is needed to read the file.
rc_tree_entries() {   # file tree -> entries, or empty
  command -v root >/dev/null 2>&1 || return 1
  root -l -b -q "$RC_ROOT/tools/tree_entries.C(\"$1\",\"$2\")" 2>/dev/null |
    sed -n 's/^RC_ENTRIES //p'
}

rc_tree_names() {     # file -> tree names on one line
  command -v root >/dev/null 2>&1 || return 1
  root -l -b -q "$RC_ROOT/tools/tree_entries.C(\"$1\",\"\")" 2>/dev/null |
    sed -n 's/^RC_TREES //p'
}

rc_doctor() {
  local n avail free
  RC_DOCTOR_FAIL=0
  RC_DOCTOR_WARN=0

  echo "module"
  if [ "$RC_MODULE_VALID" -eq 1 ]; then _rc_ok "checkout at $RC_MODULE"
  else _rc_bad "$RC_MODULE does not look like an alignment checkout"; fi
  if [ "$RC_O2_REQUIRED" -eq 1 ] && [ "$GEOM_BACKEND" != o2 ]; then
    _rc_bad "this tree only has the O2 backend; set GEOM_BACKEND=o2"
  fi

  if [ "$GEOM_BACKEND" = o2 ]; then
    # alienv exports O2_ROOT. Without it the job compiles against headers that are
    # not there, twenty minutes after launch rather than now.
    if [ -n "${O2_ROOT:-}" ]; then
      _rc_ok "backend o2 -- O2 loaded from ${O2_ROOT}"
    elif command -v alienv >/dev/null 2>&1; then
      _rc_warn "backend o2 -- alienv is present but O2 is not loaded in this shell; run: eval \`alienv load -w \$O2_DIR/sw O2/latest\`"
    else
      _rc_bad "backend o2 -- no O2 in this environment (O2_ROOT unset, no alienv). Load O2, or set GEOM_BACKEND=cache"
    fi
  else
    _rc_ok "backend cache -- no O2 needed at run time"
    if [ ! -f "$RC_MODULE/geometry/its2_geom.root" ]; then
      _rc_bad "no geometry/its2_geom.root -- build it with tools/export_geometry_cache.C (needs ROOT's geometry component)"
    else
      _rc_ok "geometry cache present"
    fi
  fi

  echo "inputs"
  if [ -z "$DATA_FILE" ]; then _rc_bad "DATA_FILE is not set"
  elif [ ! -f "$DATA_FILE" ]; then _rc_bad "DATA_FILE does not exist: $DATA_FILE"
  else
    n=$(rc_tree_entries "$DATA_FILE" "$DATA_TREE")
    if [ -z "$n" ]; then
      _rc_warn "DATA_FILE present; root could not report '$DATA_TREE' (trees: $(rc_tree_names "$DATA_FILE"))"
    else
      _rc_ok "DATA_FILE $DATA_TREE: $n entries"
      [ "$JOB_NDATA" -gt "$n" ] && _rc_warn "JOB_NDATA is $JOB_NDATA but the tree holds $n"
    fi
  fi
  for pair in "GEOM_FILE:$GEOM_FILE" "ALIGN_FILE:$ALIGN_FILE" "PARAMS_ARCHIVE:$PARAMS_ARCHIVE"; do
    k=${pair%%:*}; v=${pair#*:}
    if [ -z "$v" ]; then _rc_bad "$k is not set"
    elif [ -e "$v" ]; then _rc_ok "$k $(basename "$v")"
    elif [ -e "$RC_MODULE/$v" ]; then _rc_ok "$k $v (in the module)"
    else _rc_bad "$k does not exist: $v"; fi
  done

  # Without weightsDU.txt the detector-unit normalisations stay uninitialised
  # and the cost comes out -nan after a full run. Cheap to check, expensive to
  # discover later.
  if [ -f "$PARAMS_ARCHIVE" ]; then
    if tar tzf "$PARAMS_ARCHIVE" 2>/dev/null | grep -q 'weightsDU\.txt'; then
      _rc_ok "seed carries weightsDU.txt"
    else
      _rc_warn "seed has no weightsDU.txt -- detector-unit normalisations stay uninitialised and the cost comes out -nan"
    fi
  elif [ -d "$PARAMS_ARCHIVE" ]; then
    [ -f "$PARAMS_ARCHIVE/weights/weightsDU.txt" ] && _rc_ok "seed carries weightsDU.txt" \
      || _rc_warn "seed has no weights/weightsDU.txt -- the cost comes out -nan"
  fi

  echo "output"
  if mkdir -p "$RC_OUTPUT" 2>/dev/null; then
    free=$(df -Pm "$RC_OUTPUT" 2>/dev/null | awk 'NR==2{printf "%.1f", $4/1024}')
    if [ -n "$free" ]; then
      awk -v g="$free" 'BEGIN{exit !(g < 2)}' && _rc_warn "$RC_OUTPUT -- only ${free} GB free" \
                                               || _rc_ok "$RC_OUTPUT -- ${free} GB free"
    else _rc_ok "$RC_OUTPUT"; fi
  else
    _rc_bad "cannot create $RC_OUTPUT"
  fi
  [ -e "$RC_JOB_DIR" ] && _rc_warn "$RC_JOB_DIR already exists; compose will overwrite it"

  echo "machine"
  command -v root >/dev/null 2>&1 && _rc_ok "root on PATH" \
    || _rc_bad "root not on PATH -- load the environment first"
  # Each job holds ~8 GB resident; two of them OOM-killed each other once.
  avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo 2>/dev/null)
  if [ -n "$avail" ]; then
    awk -v g="$avail" 'BEGIN{exit !(g > 9)}' && _rc_ok "${avail} GB available; a job holds ~8 GB" \
                                             || _rc_warn "${avail} GB available; a job holds ~8 GB resident"
  fi
  if [ -n "$(rc_running_pids)" ]; then
    _rc_warn "a job is already running (pid $(rc_running_pids)); a second needs 8 GB more"
  fi

  echo
  echo "estimated runtime  ${RC_EST_MIN} min ($(awk -v m="$RC_EST_MIN" 'BEGIN{printf "%.1f", m/60}') h)"
  echo "$RC_DOCTOR_FAIL failed, $RC_DOCTOR_WARN warnings"
  [ "$RC_DOCTOR_FAIL" -eq 0 ]
}

# --- display --------------------------------------------------------------

rc_print() {
  local k t v group last=""
  for k in $(rc_keys); do
    t=$(rc_type "$k"); group=$(rc_group "$k")
    if [ "$group" != "$last" ]; then printf '\n[%s]\n' "$group"; last=$group; fi
    eval "v=\$$k"
    printf '  %-24s %s\n' "$k" "$v"
  done
  printf '\n[derived]\n'
  printf '  %-24s %s\n' "steps"            "$FIRST_STEP .. $RC_LAST_STEP ($N_STEPS)"
  printf '  %-24s %s\n' "eta (start)"      "$RC_ETA"
  printf '  %-24s %s\n' "module"           "$RC_MODULE"
  printf '  %-24s %s\n' "geometry backend" \
    "$GEOM_BACKEND$( [ "$RC_CACHE_CAPABLE" -eq 1 ] && echo '  (tree supports both)' || echo '  (tree is O2-only)' )"
  printf '  %-24s %s\n' "job directory"    "$RC_JOB_DIR"
  printf '  %-24s %s\n' "seed unpacks to"  "MLPTrain_Step${RC_SEED_STEP}/"
  printf '  %-24s %s\n' "ndf per event"    "$(( 12 * JOB_NTRACKMAX + 1 ))"
  printf '  %-24s %s\n' "estimated runtime" \
    "$RC_EST_MIN min ($(awk -v m="$RC_EST_MIN" 'BEGIN{printf "%.1f", m/60}') h) total; $RC_EST_STEP min per step"
  printf '  %-24s %s\n' "module now holds" \
    "nDATA=$RC_MOD_NDATA nEPOCH=$RC_MOD_NEPOCH nTrackMax=$RC_MOD_NTRACKMAX DET_MAG=$RC_MOD_DET_MAG"
}

# --- mutation -------------------------------------------------------------

# Rewrites one key in place, preserving comments and layout.
rc_set() {
  local key="$1" val="$2" f tmp
  f=$(rc_conf_file)

  rc_type "$key" | grep -q . || { rc_die "unknown key: $key"; return 1; }
  case "$val" in *'"'*) rc_die "value must not contain a double quote"; return 1 ;; esac

  tmp=$(mktemp "${TMPDIR:-/tmp}/runconf.XXXXXX") || return 1
  awk -v key="$key" -v val="$val" '
    skipping { if ($0 ~ /"/) skipping = 0; next }
    index($0, key "=") == 1 {
      line = $0
      n = gsub(/"/, "", line)
      print key "=\"" val "\""
      if (n < 2) skipping = 1
      found = 1
      next
    }
    { print }
    END { if (!found) exit 3 }
  ' "$f" > "$tmp"

  case $? in
    0) ;;
    3) rm -f "$tmp"; rc_die "key $key not present in $f"; return 1 ;;
    *) rm -f "$tmp"; rc_die "rewrite failed"; return 1 ;;
  esac

  cat "$tmp" > "$f" && rm -f "$tmp"
}

# --- run composition ------------------------------------------------------

RC_COPY_TREE="Ymlp monitor geometry NetworkParameters tools"
RC_COPY_FILE="run_train_circle.C run_profile_beam.C batch_train YMLPParallel.h
              YMLPBeamProfile.h TrendingNetwork.tgz OffsetSlopeCorrectionParams.txt
              UpdateSensorsList.txt NTracksBySensor.txt
              ITSClusterDictionary_20220903.root o2simtopology_13294.json"

# An input may be given absolute, relative to the working directory, or by the
# bare name it has inside the module.
rc_resolve() {
  [ -z "$1" ] && return 1
  if [ -e "$1" ]; then (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")")
  elif [ -e "$RC_MODULE/$1" ]; then printf '%s/%s\n' "$RC_MODULE" "$1"
  else return 1; fi
}

# Builds a self-contained run directory. The module checkout is only ever read.
rc_compose() {
  local d f src top

  [ "$RC_MODULE_VALID" -eq 1 ] || { rc_die "$RC_MODULE is not an alignment checkout"; return 1; }
  mkdir -p "$RC_JOB_DIR" || return 1
  : > "$RC_MANIFEST"

  _m() { echo "$*" >> "$RC_MANIFEST"; echo "$*"; }
  _m "composed  $(date '+%Y-%m-%d %H:%M:%S')"
  _m "module    $RC_MODULE"
  _m "job       $RC_JOB_DIR"
  _m "steps     $FIRST_STEP..$RC_LAST_STEP ($N_STEPS)"

  for d in $RC_COPY_TREE; do
    [ -d "$RC_MODULE/$d" ] || continue
    rm -rf "$RC_JOB_DIR/$d"
    cp -a "$RC_MODULE/$d" "$RC_JOB_DIR/$d" || return 1
    rm -rf "$RC_JOB_DIR/$d/AlignLib" "$RC_JOB_DIR/$d/__pycache__"
  done
  for f in $RC_COPY_FILE; do
    [ -f "$RC_MODULE/$f" ] && cp -a "$RC_MODULE/$f" "$RC_JOB_DIR/$f"
  done

  # Big read-only inputs are linked, not copied: the source file alone is
  # ~800 MB and copying it per job would fill the disk for no benefit.
  for pair in "DATA_FILE:XXXXinput.root" "GEOM_FILE:o2sim_geometry.root" "ALIGN_FILE:ITSAlignment.root"; do
    eval "src=\$${pair%%:*}"
    src=$(rc_resolve "$src") || { rc_die "cannot resolve ${pair%%:*}"; return 1; }
    rm -f "$RC_JOB_DIR/${pair#*:}"
    ln -s "$src" "$RC_JOB_DIR/${pair#*:}" || return 1
    _m "link      ${pair#*:} -> $src"
  done

  # Seed parameters land in MLPTrain_Step<FIRST_STEP-1>, which is where
  # run_train_circle looks for SetPrevUSL and SetPrevWeight. Later steps need
  # no seeding: step N reads MLPTrain_Step<N-1>, which step N-1 just wrote.
  rm -rf "$RC_SEED_DIR"; mkdir -p "$RC_SEED_DIR"
  src=$(rc_resolve "$PARAMS_ARCHIVE") || { rc_die "cannot resolve PARAMS_ARCHIVE"; return 1; }
  if [ -d "$src" ]; then
    cp -a "$src/." "$RC_SEED_DIR/" || return 1
  else
    # The archive usually carries a single MLPTrain_Step<N>/ top level; strip
    # it so the contents land directly in the seed directory.
    top=$(tar tzf "$src" 2>/dev/null | sed -n '1s|/.*||p')
    if [ -n "$top" ] && tar tzf "$src" 2>/dev/null | grep -qv "^$top/"; then top=""; fi
    if [ -n "$top" ]; then tar xzf "$src" -C "$RC_SEED_DIR" --strip-components=1 || return 1
    else tar xzf "$src" -C "$RC_SEED_DIR" || return 1; fi
  fi
  _m "seed      $src -> MLPTrain_Step${RC_SEED_STEP}/"

  rc_gen_ymlpparallel "$RC_JOB_DIR/YMLPParallel.h"
  _m "YMLPParallel.h  jparallel=$JOB_JPARALLEL nCORE=$JOB_NCORE nDATA=$JOB_NDATA nEPOCH=$JOB_NEPOCH"
  rc_patch_detconst "$RC_JOB_DIR/$RC_DETCONST"
  _m "DetectorConstant.h  nTrackMax=$JOB_NTRACKMAX DET_MAG=$JOB_DET_MAG FITMODEL=$JOB_FITMODEL VERTEXFIT=$JOB_VERTEXFIT"
  _m "                    pT=[$JOB_PT_MIN,$JOB_PT_MAX] chi_cost=[$JOB_CHI_IB,$JOB_CHI_OB] chi_train=[$JOB_CHI_IB_TRAIN,$JOB_CHI_OB_TRAIN]"
  _m "                    TrackRejection=$JOB_TRACK_REJECT ip=[$JOB_IP_RANGE_R,$JOB_IP_RANGE_Z] minCluster=$JOB_MIN_CLUSTER"

  rc_patch_mlpsrc "$RC_JOB_DIR/$RC_MLPSRC"
  _m "YMultiLayerPerceptron.cxx  eta=$RC_ETA (const=$JOB_ETA_CONSTANT scale=$JOB_ETA_SCALE DETRES=$JOB_ETA_DETRES) ValidWindow=$JOB_VALID_WINDOW"

  rc_patch_driver "$RC_JOB_DIR/$RC_DRIVER"
  _m "run_train_circle.C  method=$JOB_LEARNING_METHOD tree=$DATA_TREE"

  rc_patch_geom "$RC_JOB_DIR/$RC_GEOMHDR"
  if [ "$GEOM_BACKEND" = o2 ]; then
    _m "YDetectorGeometry.h  backend=o2 (YGEOM_USE_O2 defined; O2 must be loaded to run)"
  else
    _m "YDetectorGeometry.h  backend=cache (no O2 needed; reads geometry/its2_geom.root)"
  fi

  rc_gen_runner
  _m "run_steps.sh  steps $FIRST_STEP..$RC_LAST_STEP ($N_STEPS)"
  unset -f _m
}

# --- the multi-step runner -------------------------------------------------

# Written into the job directory rather than run inline, so the loop survives
# the shell that launched it and status can read its progress. Mirrors
# process_all_train.sh: rewrite batch_train for the step, run it, file the log,
# archive the step.
rc_gen_runner() {
  cat > "$RC_RUNNER" <<EOF
#!/bin/bash
# Generated by config/runctl.sh compose. Regenerated on every compose.
cd "\$(dirname "\$0")" || exit 1

first=$FIRST_STEP
last=$RC_LAST_STEP

for step in \$(seq \$first \$last); do
  echo "=== step \$step of \$last (\$((step - first + 1)) / $N_STEPS) \$(date '+%Y-%m-%dT%H:%M:%S') ==="
  echo "\$step \$first \$last" > run.step

  echo 'gROOT->ProcessLine(".L ./Ymlp/src/YDetectorGeometry.cxx");'  > batch_train
  echo "gROOT->ProcessLine(\".x run_train_circle.C(\$step)\");"  >> batch_train

  root -l -b < batch_train > train.log 2>&1
  rc=\$?

  if [ -d "MLPTrain_Step\$step" ]; then
    mv train.log "MLPTrain_Step\$step/train_done.log"
    tar -zcf "MLPTrain_Step\$step.tgz" "MLPTrain_Step\$step"
  else
    echo "step \$step produced no MLPTrain_Step\$step/ -- stopping"
    cat train.log | tail -40
    echo "STEPFAIL \$step" > run.step
    exit 1
  fi

  if [ \$rc -ne 0 ]; then
    echo "step \$step exited \$rc -- stopping"
    echo "STEPFAIL \$step" > run.step
    exit \$rc
  fi
done

echo "DONE \$last" > run.step
echo "=== all $N_STEPS step(s) complete \$(date '+%Y-%m-%dT%H:%M:%S') ==="
EOF
  chmod +x "$RC_RUNNER"
}

# --- launching ------------------------------------------------------------

rc_pidfile() { echo "$RC_JOB_DIR/run.pid"; }

# The pid of this job if it is still alive, else nothing.
rc_running_pids() {
  local p f
  f=$(rc_pidfile)
  [ -r "$f" ] || return 0
  p=$(cat "$f" 2>/dev/null)
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"
}

rc_run() {
  local p
  [ -d "$RC_JOB_DIR" ] || { rc_die "no job directory; run 'runctl.sh compose' first"; return 1; }
  [ -x "$RC_RUNNER" ]  || { rc_die "no runner at $RC_RUNNER; compose again"; return 1; }
  p=$(rc_running_pids)
  [ -n "$p" ] && { rc_die "a job is already running here (pid $p)"; return 1; }

  if [ -n "$ROOTSYS_OVERRIDE" ]; then
    export ROOTSYS="$ROOTSYS_OVERRIDE"
    export PATH="$ROOTSYS/bin:$PATH"
    export LD_LIBRARY_PATH="$ROOTSYS/lib:${LD_LIBRARY_PATH:-}"
  fi
  command -v root >/dev/null 2>&1 || { rc_die "root is not on PATH"; return 1; }

  echo "START $(date '+%Y-%m-%dT%H:%M:%S')" > "$RC_LOG"
  rm -f "$RC_PROGRESS"
  # setsid so the run outlives the shell that started it -- these take hours,
  # and with N_STEPS above 1 considerably longer.
  ( setsid nohup "$RC_RUNNER" >> "$RC_LOG" 2>&1 &
    echo $! > "$(rc_pidfile)" )
  sleep 1
  p=$(cat "$(rc_pidfile)" 2>/dev/null)
  echo "started pid $p"
  echo "steps   $FIRST_STEP..$RC_LAST_STEP ($N_STEPS)"
  echo "log     $RC_LOG"
  echo "expect  ~$RC_EST_MIN min ($(awk -v m="$RC_EST_MIN" 'BEGIN{printf "%.1f", m/60}') h) for all steps"
}

# Where the loop has got to, read from the file the runner writes, so the
# answer survives closing and reopening the window.
rc_progress() {
  local a b c
  [ -r "$RC_PROGRESS" ] || { echo "not started"; return 0; }
  read -r a b c < "$RC_PROGRESS"
  case "$a" in
    DONE)     echo "all steps complete (last $b)" ;;
    STEPFAIL) echo "FAILED on step $b" ;;
    *)        echo "step $a of $c" ;;
  esac
}

rc_status() {
  local p
  p=$(rc_running_pids)
  if [ -n "$p" ]; then
    echo "running   pid $p"
    echo "progress  $(rc_progress)"
    echo "job       $RC_JOB_DIR"
    [ -r "$RC_LOG" ] && echo "log       $RC_LOG ($(wc -l < "$RC_LOG") lines)"
  elif [ -r "$RC_LOG" ]; then
    echo "not running"
    echo "progress  $(rc_progress)"
    echo "log       $RC_LOG"
    tail -3 "$RC_LOG" | sed 's/^/  /'
  else
    echo "no run recorded in $RC_JOB_DIR"
  fi
}

rc_stop() {
  local p
  p=$(rc_running_pids)
  [ -z "$p" ] && { echo "nothing running"; return 0; }
  kill "$p" 2>/dev/null && echo "sent TERM to $p"
}

# --- outputs --------------------------------------------------------------

rc_outputs() {
  local step d n any=0
  for step in $(seq "$FIRST_STEP" "$RC_LAST_STEP"); do
    d="$RC_JOB_DIR/MLPTrain_Step${step}"
    [ -d "$d" ] || continue
    any=1
    n=$(find "$d" -type f 2>/dev/null | wc -l)
    echo "MLPTrain_Step${step}/  ($n files)"
    find "$d" -type f \( -name '*.root' -o -name '*.log' \) 2>/dev/null |
      sort | while read -r f; do
        printf '    %10s  %s\n' "$(stat -c%s "$f" 2>/dev/null)" "${f#$d/}"
      done
  done
  if [ "$any" -eq 0 ]; then
    echo "no MLPTrain_Step${FIRST_STEP}..${RC_LAST_STEP}/ in $RC_JOB_DIR yet"
    [ -d "$RC_JOB_DIR" ] && ls -1 "$RC_JOB_DIR" | sed 's|^|  |'
    return 1
  fi
}
