#!/bin/bash
# usage: analyze_fam.sh FAM GPU   -> logs to data/an_<FAM>.log
F=$1; export CUDA_VISIBLE_DEVICES=$2
{
python ldt.py extract --fam $F ep_a bat_ep
python ldt0.py $F
for s in fp8_tensor int8_channel; do python llr.py fam:$F $s ep_a bat_ep; done
} > data/an_$F.log 2>&1
