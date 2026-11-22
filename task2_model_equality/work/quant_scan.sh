#!/bin/bash
# usage: quant_scan.sh TARGET GPU DATA...   (TARGET = fam:Q15 | HF model id) -> data/qs_<name>.log
T=$1; export CUDA_VISIBLE_DEVICES=$2; shift 2
name=$(echo $T | tr ':/' '__')
for s in fp8_tensor fp8_channel int8_channel int4_g128 nf4_b64; do python llr.py $T $s "$@"; done > data/qs_$name.log 2>&1
