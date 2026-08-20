#!/bin/bash
# ==========================================================================
#  runctl.sh -- configure, check, compose and launch one alignment run
# ==========================================================================
#  Usage:
#    runctl.sh show                 print every setting and what follows
#    runctl.sh get KEY              print one value (RC_* reads a derived one)
#    runctl.sh set KEY=VALUE ...    change settings in place
#    runctl.sh keys                 list every key
#    runctl.sh validate             check types and relationships
#    runctl.sh doctor               check this machine has what the run needs
#    runctl.sh compose              build the job directory
#    runctl.sh run                  launch the composed job
#    runctl.sh status               is it running, and where is the log
#    runctl.sh log [-f]             show the run log
#    runctl.sh stop                 stop the running job
#    runctl.sh outputs              list what the job produced
#    runctl.sh ui                   open the ROOT run-console window
# ==========================================================================
set -u

_here=$(cd "$(dirname "$0")" && pwd)
. "$_here/runconf.sh"

# Printed from the header block above rather than a hardcoded line range: a
# verb added or removed there must not silently desynchronise the help text.
# Stops at the second rule, which closes the block.
usage() {
  awk 'NR<3{next} /^# ={20,}/{if(++n==2)exit; next} /^#/{sub(/^#[ ]?[ ]?/,""); print}' "$0"
}

cmd=${1:-show}
[ $# -gt 0 ] && shift

case "$cmd" in

  show)
    rc_load || exit 1
    rc_print
    ;;

  get)
    [ $# -eq 1 ] || { echo "usage: runctl.sh get KEY" >&2; exit 2; }
    rc_load || exit 1
    # Derived RC_* values are readable too, so the GUI shows the runtime
    # estimate and the detected backend without recomputing either.
    case "$1" in
      RC_*) eval "printf '%s\n' \"\${$1:-}\"" ;;
      *)
        rc_type "$1" | grep -q . || { echo "unknown key: $1" >&2; exit 2; }
        eval "printf '%s\n' \"\$$1\"" ;;
    esac
    ;;

  set)
    [ $# -ge 1 ] || { echo "usage: runctl.sh set KEY=VALUE [KEY=VALUE ...]" >&2; exit 2; }
    for pair in "$@"; do
      case "$pair" in
        *=*) ;;
        *) echo "not a KEY=VALUE pair: $pair" >&2; exit 2 ;;
      esac
      rc_set "${pair%%=*}" "${pair#*=}" || exit 1
      echo "set ${pair%%=*}"
    done
    # Re-read so a change that breaks an invariant is reported immediately.
    rc_load || exit 1
    if ! rc_validate; then
      echo "configuration is now invalid -- see above" >&2
      exit 1
    fi
    ;;

  keys)
    rc_keys
    ;;

  validate)
    rc_load || exit 1
    if rc_validate; then echo "configuration is consistent"; else exit 1; fi
    ;;

  doctor)
    rc_load || exit 1
    rc_validate || { echo "fix the configuration first" >&2; exit 1; }
    rc_doctor
    ;;

  compose)
    rc_load || exit 1
    rc_validate || { echo "refusing to compose from an invalid configuration" >&2; exit 1; }
    rc_compose
    ;;

  run)
    rc_load || exit 1
    rc_validate || { echo "refusing to run from an invalid configuration" >&2; exit 1; }
    rc_run
    ;;

  status)
    rc_load || exit 1
    rc_status
    ;;

  log)
    rc_load || exit 1
    [ -r "$RC_LOG" ] || { echo "no log at $RC_LOG" >&2; exit 1; }
    if [ "${1:-}" = "-f" ]; then tail -f "$RC_LOG"; else tail -40 "$RC_LOG"; fi
    ;;

  stop)
    rc_load || exit 1
    rc_stop
    ;;

  outputs)
    rc_load || exit 1
    rc_outputs
    ;;

  ui)
    rc_load || exit 1
    if ! command -v root >/dev/null 2>&1; then
      echo "root is not on PATH. Load the environment first, for example:" >&2
      echo "  eval \`alienv load -w \$O2_DIR/sw O2/latest\`" >&2
      exit 1
    fi
    root -l "$RC_ROOT/tools/ConfigUI/RunConsoleUI.C(\"$(rc_conf_file)\")"
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
