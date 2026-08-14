#!/bin/bash

Tag=$1
stepI=$2
stepF=$3

StepLIST=$(seq ${stepI} ${stepF})

# Apply Corrections once before submission (if required)

#cd MLPTrain_Step200
#./control_all.sh &> log.out
#cd ..
#cp -r MLPTrain_Step200 MLPTrain_Step200.${Tag}

# Beam Profile

root -l -b -q run_profile_beam.C &> profile_beam.log

for s in ${StepLIST}
do
  nStep=$s
  echo "Current Step $nStep"
  echo "gROOT->ProcessLine("'"'".L ./Ymlp/src/YDetectorGeometry.cxx"'"'");"     > batch_train
  echo "gROOT->ProcessLine("'"'".x run_train_circle.C(${nStep})"'"'");" 	>> batch_train
  
  ./process.sh
  wait
  mv train.log MLPTrain_Step${nStep}/train_done.log
  
  tar -zcvf MLPTrain_Step${nStep}.tgz MLPTrain_Step${nStep}
done
