#!/bin/bash
# full feature extraction v2 on all ep data (ep_a, bat_ep, bat3_ep) + ref (bat_ref, s1_explore, bat3_ref if present)
fam() { python -c "from fams import FAMILIES as F; m,e=F['$1']; print(m, ','.join(e))"; }
read mQ15 eQ15 < <(fam Q15); read mQ3 eQ3 < <(fam Q3); read mL1 eL1 < <(fam L1)
R=ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09
QS=fp8_tensor,awq
LS=fp8_tensor,ctint8
RD="bat_ref s1_explore"; [ -s data/bat3_ref.jsonl ] && RD="$RD bat3_ref"
CUDA_VISIBLE_DEVICES=0 python feat.py Q15_v15 $mQ15 ep_a bat_ep bat3_ep --eps $eQ15 --schemes $QS > data/feat_Q15_v15.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python feat.py Q3_v15 $mQ3 ep_a bat_ep bat3_ep --eps $eQ3 --schemes $QS > data/feat_Q3_v15.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python feat.py L1_v15 $mL1 ep_a bat_ep bat3_ep --eps $eL1 --schemes $LS > data/feat_L1_v15.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python feat.py ref_v15 HuggingFaceTB/SmolLM2-360M-Instruct $RD --eps $R --schemes fp8_tensor > data/feat_ref_v15.log 2>&1 &
wait
