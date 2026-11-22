#!/bin/bash
# usage: ldt_fam.sh FAM GPU DATA... -> data/ldt_<FAM>.log
F=$1; export CUDA_VISIBLE_DEVICES=$2; shift 2
{ python ldt.py extract --fam $F "$@" && python ldt0.py $F; } > data/ldt_$F.log 2>&1
