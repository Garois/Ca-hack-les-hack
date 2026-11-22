for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct unsloth/Llama-3.2-1B-Instruct Qwen/Qwen2.5-3B-Instruct unsloth/Llama-3.2-3B-Instruct allenai/OLMo-2-0425-1B-Instruct Qwen/Qwen3-0.6B Qwen/Qwen2.5-7B-Instruct unsloth/Llama-3.1-8B-Instruct; do
  .venv-llm/bin/hf download $m --include "*.json" "*.safetensors" "tokenizer*" "*.txt" "*.model" >/dev/null 2>&1 && echo "ok $m" || echo "FAIL $m"
done
echo DONE
