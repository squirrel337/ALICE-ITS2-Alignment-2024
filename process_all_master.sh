#!/bin/bash

Tag=$1
stepI=$2
stepF=$3

homedir=$PWD

# Only the YGEOM_USE_O2 build needs O2 at runtime. The default build reads the
# geometry from the exported per-chip cache (see Ymlp/src/YDetectorGeometry.cxx),
# so the campaign runs on ROOT alone and this load is skipped.
o2dir=${O2DIR:-/home/alice/Software/v20230501}

echo "homedir : ${homedir}"

if [ -n "${YGEOM_USE_O2}" ]; then
  echo "o2dir   : ${o2dir}"
  echo "YGEOM_USE_O2 set -- loading O2"
  eval `alienv load -w ${o2dir}/sw O2/latest`
else
  echo "cache-backed geometry -- O2 not required"
fi

./process_all_train.sh ${Tag} ${stepI} ${stepF}
