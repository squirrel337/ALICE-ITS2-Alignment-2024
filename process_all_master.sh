#!/bin/bash

Tag=$1
stepI=$2
stepF=$3

homedir=$PWD

# O2 is loaded by default, because the default geometry backend reads through
# o2::its::GeometryTGeo. Set YGEOM_CACHE=1 for the cache backend, which needs no O2
# -- the mode for machines where O2 cannot be installed. O2DIR overrides the
# installation path, which used to be hardcoded to one machine.
o2dir=${O2DIR:-/home/alice/Software/v20230501}

echo "homedir : ${homedir}"

if [ -n "${YGEOM_CACHE:-}" ]; then
  echo "YGEOM_CACHE set -- cache-backed geometry, O2 not loaded"
else
  echo "o2dir   : ${o2dir}"
  eval `alienv load -w ${o2dir}/sw O2/latest`
fi

./process_all_train.sh ${Tag} ${stepI} ${stepF}
