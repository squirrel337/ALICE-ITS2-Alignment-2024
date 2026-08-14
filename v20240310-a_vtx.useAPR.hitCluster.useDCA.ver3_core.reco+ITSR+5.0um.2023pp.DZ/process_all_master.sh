#!/bin/bash

Tag=$1
stepI=$2
stepF=$3

homedir=$PWD

o2dir=/home/alice/Software/v20230501

echo "homedir : ${homedir}"
echo "o2dir   : ${o2dir}"

eval `alienv load -w ${o2dir}/sw O2/latest`

./process_all_train.sh ${Tag} ${stepI} ${stepF}
