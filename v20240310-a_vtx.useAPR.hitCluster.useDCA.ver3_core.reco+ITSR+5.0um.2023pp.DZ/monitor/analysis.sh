#!/bin/bash
root -l -b < batch_CheckQuality &> /dev/null &
root -l -b < batch_ImpactParams &> /dev/null &
root -l -b < batch_Residual2D &> /dev/null &
root -l -b < batch_Profile &> /dev/null &
root -l -b < batch_Residual &> /dev/null &      
