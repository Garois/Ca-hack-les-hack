#!/bin/bash
# one job, 4 GPUs: one family per GPU.  usage (inside gpujob.sh): ./feat_all.sh SUFFIX "EP_DATA..." "REF_DATA..."
S=$1; EPD=$2; REFD=$3
fam() { python -c "from fams import FAMILIES as F; m,e=F['$1']; print(m, ','.join(e))"; }
run() { # tag gpu model eps data...
  local tag=$1 gpu=$2 model=$3 eps=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu python feat.py $tag $model "$@" --eps $eps > data/feat_$tag.log 2>&1
}
read mQ15 eQ15 < <(fam Q15); read mQ3 eQ3 < <(fam Q3); read mL1 eL1 < <(fam L1)
run Q15$S 0 $mQ15 $eQ15 $EPD &
run Q3$S 1 $mQ3 $eQ3 $EPD &
run L1$S 2 $mL1 $eL1 $EPD &
run ref$S 3 HuggingFaceTB/SmolLM2-360M-Instruct ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09 $REFD &
wait
