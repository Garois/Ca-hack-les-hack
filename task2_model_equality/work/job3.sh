#!/bin/bash
# full feature extraction v2 (ep_a, bat_ep, bat3_ep) + quiz calibration (local sampling), one family per GPU; ref on GPU 3
fam() { python -c "from fams import FAMILIES as F; m,e=F['$1']; print(m, ','.join(e))"; }
read mQ15 eQ15 < <(fam Q15); read mQ3 eQ3 < <(fam Q3); read mL1 eL1 < <(fam L1)
R=ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09
run() { # tag gpu model eps
  CUDA_VISIBLE_DEVICES=$2 python feat.py $1_v2 $3 ep_a bat_ep bat3_ep --eps $4 --schemes fp8_tensor --nprev 1 > data/feat_$1_v2.log 2>&1
  for i in $(seq 1 90); do grep -q "budget after" data/quizzes.log && break; sleep 20; done
  CUDA_VISIBLE_DEVICES=$2 python qcal.py $1 $3 --eps $4 --n 200 > data/qcal_$1.log 2>&1
}
run Q15 0 $mQ15 $eQ15 &
run Q3 1 $mQ3 $eQ3 &
run L1 2 $mL1 $eL1 &
CUDA_VISIBLE_DEVICES=3 python feat.py ref_v2 HuggingFaceTB/SmolLM2-360M-Instruct bat_ref s1_explore --eps $R --schemes fp8_tensor --nprev 1 > data/feat_ref_v2.log 2>&1 &
wait
