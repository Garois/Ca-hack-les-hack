#!/bin/bash
# external quant scores (Tfit from feat v1) + decoding fit, one family per GPU
fam() { python -c "from fams import FAMILIES as F; m,e=F['$1']; print(m, ','.join(e))"; }
read mQ15 eQ15 < <(fam Q15); read mQ3 eQ3 < <(fam Q3); read mL1 eL1 < <(fam L1)
R=ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09
(export CUDA_VISIBLE_DEVICES=0; python feat.py Q15_v1 $mQ15 ep_a bat_ep --eps $eQ15 --quant-only Q15_v1 --schemes awq,gptq4,gptq8 > data/qx_Q15.log 2>&1; python decfit.py Q15_v1 $mQ15 ep_a bat_ep --eps $eQ15 > data/dec_Q15.log 2>&1) &
(export CUDA_VISIBLE_DEVICES=1; python feat.py Q3_v1 $mQ3 ep_a bat_ep --eps $eQ3 --quant-only Q3_v1 --schemes awq,gptq4,gptq8 > data/qx_Q3.log 2>&1; python decfit.py Q3_v1 $mQ3 ep_a bat_ep --eps $eQ3 > data/dec_Q3.log 2>&1) &
(export CUDA_VISIBLE_DEVICES=2; python feat.py L1_v1 $mL1 ep_a bat_ep --eps $eL1 --quant-only L1_v1 --schemes ctint8,ctfp8 > data/qx_L1.log 2>&1; python decfit.py L1_v1 $mL1 ep_a bat_ep --eps $eL1 > data/dec_L1.log 2>&1) &
(export CUDA_VISIBLE_DEVICES=3; python decfit.py ref_v1 HuggingFaceTB/SmolLM2-360M-Instruct bat_ref s1_explore --eps $R > data/dec_ref.log 2>&1) &
wait
