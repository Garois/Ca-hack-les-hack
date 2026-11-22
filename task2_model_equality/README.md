# Task 2: Model equality (public score 1.00, tied 1st)

There are 33 black-box chat endpoints `ep-01..ep-33`. For every pair we decide **same family** (base model, 10 % of
the score) and **same model** (same checkpoint + quantization + LoRA, 90 %). System prompts, personas, watermarks and
sampling settings are disguises that do not change the model. The budget is 100 answers per endpoint.

- `work/`: all our code. [`work/README.md`](work/README.md) is the detailed working notes (French), with a
  file-by-file table and the run order.
- `task_template.py`, `query_reference.py`, `score_practice.py`, `practice_answers.csv`: the organisers' kit and
  the 9 labelled reference endpoints, which we used as a validation set.
- [`work/subs/FINAL_public1.0.csv`](work/subs/FINAL_public1.0.csv): **final submission**.
  [`work/subs/log.jsonl`](work/subs/log.jsonl) lists all submissions with their partitions and scores.

## Method: score every returned token under the local base model

| Question | Test | Code |
|---|---|---|
| Which family? | Mean log-likelihood of answers under Qwen2.5-1.5B-Instruct / Qwen2.5-3B-Instruct / Llama-3.2-1B-Instruct. The winner is unambiguous: 3 × 11 endpoints | [`family_table.py`](work/family_table.py), [`fams.py`](work/fams.py) |
| Sampling disguise? | Maximum-likelihood temperature and top-p / top-k / min-p per endpoint (bootstrap CI). A base model served with top-p has ~0 tokens outside the nucleus | [`decfit.py`](work/decfit.py), [`decode_fit.py`](work/decode_fit.py), [`tailstat.py`](work/tailstat.py), [`tboot.py`](work/tboot.py) |
| Quantized? | Likelihood ratio base vs local quantized copy (fp8 tensor/channel, int8, int4-g128, nf4), centred scores with their variance, official AWQ/GPTQ/compressed-tensors checkpoints de-quantized in PyTorch | [`llr.py`](work/llr.py), [`feat.py`](work/feat.py), [`extq.py`](work/extq.py), [`quant_scan.sh`](work/quant_scan.sh) |
| LoRA? Same adapter? | Learned-direction test: fit a vocabulary-bias (or hidden-space) direction explaining an endpoint and measure the strength α of the other endpoints along it. Endpoints sharing an adapter recognise each other, base endpoints have α ≈ 0 (validated on the refs: LoRA 0.3 / 0.7 / 1.5 recovered and ordered) | [`ldt0.py`](work/ldt0.py), [`ldt.py`](work/ldt.py), [`ana2.py`](work/ana2.py), [`ana3.py`](work/ana3.py) |
| Weight quirks | Quiz prompts: identical answers that are rare under the base model | [`qcal.py`](work/qcal.py), [`qpair.py`](work/qpair.py), [`qfmt.py`](work/qfmt.py) |
| Data collection | Batteries of long-answer prompts (resumable), 3 prompts per endpoint for family detection | [`ep_a.py`](work/ep_a.py), [`battery.py`](work/battery.py), [`run_battery.py`](work/run_battery.py), [`battery2.py`](work/battery2.py), [`battery3.py`](work/battery3.py) |
| Submission | Partition → CSV; queued submissions | [`write_sub.py`](work/write_sub.py), [`probe.py`](work/probe.py) |

Final structure: each family has one base model behind 6–7 disguised endpoints, plus 4–5 distinct variants
(fp8-quantized or LoRA at different strengths). Candidates built from the analysis alone scored 0.30–0.55 on the
public split, with the remaining doubt in the Qwen-3B family. Splitting out `ep-18` and then `ep-25` as their own
models, after comparing alternative partitions on the public split, gave 0.80 and then 1.00.

## Reproduce

```bash
cd work && export PATH=../.venv/bin:$PATH          # API key in ../.env (API_KEY=...), never committed
python ep_a.py                                      # families (3 requests per endpoint)
.venv-llm/bin/python family_table.py ep_a Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct unsloth/Llama-3.2-1B-Instruct
python run_battery.py ep ; python run_battery2.py   # data collection (long)
sbatch gpujob.sh './quant_scan.sh fam:Q15 0 ep_a bat_ep bat2_ep & ...'   # quantization tests, see work/README.md
sbatch gpujob.sh './ldt_fam.sh Q15 0 ep_a bat_ep bat2_ep & ...'          # LoRA tests
python write_sub.py / sub_candB.py                  # partition -> CSV
```
