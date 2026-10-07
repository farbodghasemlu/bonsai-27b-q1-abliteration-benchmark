#!/usr/bin/env bash
# Builds two Q1_0 repacks of Bonsai-27B and benchmarks them against PrismML's official Q1_0 GGUF.
#   official    = prism-ml/Bonsai-27B-gguf                          (PrismML's own 1-bit file)
#   control     = prism-ml/Bonsai-27B-unpacked, repacked by us      -> measures our packing loss
#   abliterated = s3nh/Bonsai-27B-unpacked-abliterated-uncensored, repacked by us
# Every step skips itself if its output exists, so after a crash just run the script again.
# Run inside tmux:  bash run_bonsai_bench.sh 2>&1 | tee ~/bonsai/run.log
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$HOME/bonsai
LCPP=$ROOT/llama.cpp
BIN=$LCPP/build/bin
MODELS=$ROOT/models
RES=$ROOT/results
THREADS=16
PORT=8090            # 8080 belongs to audit-llama

# Sample sizes. Identical for every model so the comparison stays fair.
PPL_CHUNKS=40        # x 512 tokens of WikiText-2
EVAL_LIMIT=150       # questions per lm-eval task (gsm8k, ifeval)
REFUSAL_N=100        # harmful prompts, same count Heretic evaluates on

mkdir -p "$MODELS" "$RES"
cd "$ROOT"

echo "== 1/7 build llama.cpp (mainline, CPU)"
if [ ! -x "$BIN/llama-server" ]; then
  apt-get update
  apt-get install -y build-essential cmake git python3-venv curl wget unzip
  [ -d "$LCPP" ] || git clone https://github.com/ggml-org/llama.cpp "$LCPP"
  cmake -S "$LCPP" -B "$LCPP/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LCPP/build" -j "$THREADS"
fi
QUANTIZE_HELP=$("$BIN/llama-quantize" --help 2>&1 || true)
if ! grep -Eq '(^|[[:space:]])Q1_0[[:space:]]' <<< "$QUANTIZE_HELP"; then
  echo "This llama.cpp build cannot quantize to Q1_0. Stopping before the big downloads."
  exit 1
fi

echo "== 2/7 python env"
if [ ! -f "$ROOT/venv/.done" ]; then
  rm -rf "$ROOT/venv"
  python3 -m venv "$ROOT/venv"
  "$ROOT/venv/bin/pip" install -U pip
  "$ROOT/venv/bin/pip" install -r "$LCPP/requirements/requirements-convert_hf_to_gguf.txt"
  "$ROOT/venv/bin/pip" install -U 'huggingface_hub>=0.34,<1' 'datasets<4' requests "lm_eval[api,ifeval]"
  touch "$ROOT/venv/.done"
fi
source "$ROOT/venv/bin/activate"

echo "== 3/7 models"
if ! ls "$MODELS"/official/*.gguf >/dev/null 2>&1; then
  hf download prism-ml/Bonsai-27B-gguf --include "*Q1_0*.gguf" --local-dir "$MODELS/official"
fi
OFFICIAL=$(ls -S "$MODELS"/official/*.gguf | head -n 1)
echo "official model: $OFFICIAL"

repack() {
  local repo=$1 name=$2
  local out=$MODELS/$name-Q1_0.gguf f16=$MODELS/$name-f16.gguf src=$MODELS/src-$name
  [ -f "$out" ] && return 0
  if [ ! -f "$f16" ]; then
    hf download "$repo" --local-dir "$src"
    # Conversion writes a full FP16 GGUF while the downloaded safetensors remain on disk.
    local free_kib
    free_kib=$(df -Pk "$MODELS" | awk 'NR == 2 {print $4}')
    if (( free_kib < 60 * 1024 * 1024 )); then
      echo "Need about 60 GiB free to convert $name to FP16 and quantize it; $(df -h "$MODELS" | awk 'NR == 2 {print $4}') available."
      exit 1
    fi
    python "$LCPP/convert_hf_to_gguf.py" "$src" --outtype f16 --outfile "$f16.tmp"
    mv "$f16.tmp" "$f16"
    rm -rf "$src"   # ~54 GB; re-downloadable if ever needed
  fi
  # --pure: official Bonsai is binary end to end (embeddings and LM head too). Without it,
  # llama-quantize keeps those tensors at higher precision and the files aren't comparable.
  "$BIN/llama-quantize" --pure "$f16" "$out.tmp" Q1_0 "$THREADS"
  mv "$out.tmp" "$out"
  rm -f "$f16"
}
repack prism-ml/Bonsai-27B-unpacked control
repack s3nh/Bonsai-27B-unpacked-abliterated-uncensored abliterated

# The unpacked HF configs report 65 blocks and an extra prediction layer, but
# both converted GGUFs contain only blk.0 through blk.63. Match the official
# GGUF metadata so llama.cpp does not expect tensors that are absent.
for name in control abliterated; do
  model="$MODELS/$name-Q1_0.gguf"
  python "$LCPP/gguf-py/gguf/scripts/gguf_set_metadata.py" \
    "$model" qwen35.block_count 64 --force
  if python -c 'import gguf, sys; r = gguf.GGUFReader(sys.argv[1]); sys.exit(not any(k in r.fields for k in ("qwen35.attention.recurrent_layers", "qwen35.nextn_predict_layers")))' "$model"; then
    python "$LCPP/gguf-py/gguf/scripts/gguf_new_metadata.py" \
      "$model" "$model.metadata.tmp" \
      --remove-metadata qwen35.attention.recurrent_layers \
      --remove-metadata qwen35.nextn_predict_layers --force
    mv "$model.metadata.tmp" "$model"
  fi
done

declare -A MODEL=(
  [official]=$OFFICIAL
  [control]=$MODELS/control-Q1_0.gguf
  [abliterated]=$MODELS/abliterated-Q1_0.gguf
)
NAMES=(official control abliterated)

echo "== 4/7 hashes + speed"
for n in "${NAMES[@]}"; do
  sha256sum "${MODEL[$n]}" > "$RES/sha256_$n.txt"
  if [ ! -f "$RES/speed_$n.md" ]; then
    "$BIN/llama-bench" -m "${MODEL[$n]}" -t "$THREADS" -p 512 -n 128 -o md > "$RES/speed_$n.md.tmp"
    mv "$RES/speed_$n.md.tmp" "$RES/speed_$n.md"
  fi
done
cat > "$RES/settings.json" <<EOF
{"llama_cpp_commit": "$(git -C "$LCPP" rev-parse --short HEAD)",
 "hardware": "$(lscpu | sed -n 's/^Model name:[[:space:]]*//p'), $(nproc) threads, $(free -g | awk '/^Mem:/{print $2}') GiB RAM, CPU only",
 "ppl_chunks": $PPL_CHUNKS, "eval_limit": $EVAL_LIMIT, "refusal_n": $REFUSAL_N}
EOF

echo "== 5/7 perplexity + KL divergence vs official"
WIKI=$ROOT/wikitext-2-raw/wiki.test.raw
[ -f "$WIKI" ] || bash "$LCPP/scripts/get-wikitext-2.sh"
BASE=$MODELS/official-logits.kld
if [ ! -f "$RES/ppl_official.log" ]; then
  "$BIN/llama-perplexity" -m "$OFFICIAL" -f "$WIKI" -c 512 --chunks "$PPL_CHUNKS" -t "$THREADS" \
    --kl-divergence-base "$BASE" > "$RES/ppl_official.log.tmp" 2>&1
  mv "$RES/ppl_official.log.tmp" "$RES/ppl_official.log"
fi
for n in control abliterated; do
  if [ ! -f "$RES/kld_$n.log" ]; then
    "$BIN/llama-perplexity" -m "${MODEL[$n]}" -c 512 -t "$THREADS" \
      --kl-divergence-base "$BASE" --kl-divergence > "$RES/kld_$n.log.tmp" 2>&1
    mv "$RES/kld_$n.log.tmp" "$RES/kld_$n.log"
  fi
done

SERVER_PID=
stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}
trap stop_server EXIT
start_server() {
  # Thinking off: on CPU, long <think> blocks would multiply eval time, and all three
  # models are treated the same way.
  "$BIN/llama-server" -m "$1" --host 127.0.0.1 --port "$PORT" -c 8192 -np 1 -t "$THREADS" --jinja \
    --chat-template-kwargs '{"enable_thinking":false}' > "$RES/server_$2.log" 2>&1 &
  SERVER_PID=$!
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "llama-server died, see $RES/server_$2.log"; exit 1; }
    sleep 5
  done
}

echo "== 6/7 refusals + lm-eval (the long part)"
for n in "${NAMES[@]}"; do
  if [ -f "$RES/refusals_$n.json" ] && ls "$RES/lmeval_$n"/*/results_*.json >/dev/null 2>&1; then
    continue
  fi
  start_server "${MODEL[$n]}" "$n"
  if [ ! -f "$RES/refusals_$n.json" ]; then
    echo "-- refusals: $n"
    python "$SCRIPT_DIR/refusal_eval.py" "$PORT" "$REFUSAL_N" "$RES/refusals_$n.jsonl" > "$RES/refusals_$n.json.tmp"
    mv "$RES/refusals_$n.json.tmp" "$RES/refusals_$n.json"
  fi
  if ! ls "$RES/lmeval_$n"/*/results_*.json >/dev/null 2>&1; then
    echo "-- lm-eval: $n"
    rm -rf "$RES/lmeval_$n"
    lm_eval --model local-chat-completions \
      --model_args "model=bonsai,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False" \
      --tasks gsm8k,ifeval --apply_chat_template --fewshot_as_multiturn \
      --limit "$EVAL_LIMIT" --output_path "$RES/lmeval_$n" --log_samples
  fi
  stop_server
done

echo "== 7/7 summary"
python "$SCRIPT_DIR/summarize.py" "$RES" > "$RES/RESULTS.md"
echo "Done: $RES/RESULTS.md"
