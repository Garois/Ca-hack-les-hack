for m in Qwen/Qwen2.5-1.5B-Instruct-AWQ Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8 Qwen/Qwen2.5-3B-Instruct-AWQ Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8 RedHatAI/Llama-3.2-1B-Instruct-quantized.w4a16 RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8 RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic; do
  .venv-llm/bin/hf download $m --include "*.json" --include "*.safetensors" > data/dlq_$(basename $m).err 2>&1 && echo "ok $m" || echo "FAIL $m"
done
echo DONE
