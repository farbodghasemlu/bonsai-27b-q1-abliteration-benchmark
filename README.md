---
license: apache-2.0
viewer: false
tags:
  - benchmark
  - bonsai
  - 1-bit
  - q1_0
  - gguf
  - abliteration
  - llama.cpp
---

# Does abliteration survive 1-bit packing? A Bonsai-27B Q1_0 benchmark

**Short answer: no.** Abliterating Bonsai-27B at FP16 and repacking it to Q1_0 (1-bit) produces a model that behaves almost exactly like the original. In our tests, it refused all 100 harmful prompts, just like the official model, and 75 of its 100 replies were identical to the official model's word for word.

This matters because 1-bit "uncensored" Bonsai releases have been listed with refusal numbers measured on the FP16 intermediate, not on the 1-bit file people actually download. We measured the 1-bit file.

## What we compared

| Model | Source |
|---|---|
| **official** | [`prism-ml/Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Bonsai-27B-gguf), PrismML's own Q1_0 file |
| **control** | [`prism-ml/Bonsai-27B-unpacked`](https://huggingface.co/prism-ml/Bonsai-27B-unpacked) (FP16), repacked to Q1_0 with our pipeline |
| **abliterated** | [`s3nh/Bonsai-27B-unpacked-abliterated-uncensored`](https://huggingface.co/s3nh/Bonsai-27B-unpacked-abliterated-uncensored) (FP16, Heretic v1.4.0), repacked to Q1_0 with the same pipeline |

The control isolates our packing process: official vs. control shows what our packing costs, and control vs. abliterated shows what abliteration changes once packed.

## Results

| Metric | official | control | abliterated |
|---|---|---|---|
| Refusals, explicit refusal phrases only | 100/100 | 100/100 | **100/100** |
| Refusals, strict marker list | 100/100 | 100/100 | 100/100 |
| Replies identical to official | — | — | 75/100 |
| KL divergence vs. official | baseline | 0.000000 ± 0.000000 | 0.000187 ± 0.000004 |
| Same top-1 token as official | baseline | 100.000 ± 0.000 % | 99.137 ± 0.092 % |
| Perplexity, WikiText-2 | 11.594 ± 0.326 | 11.594 ± 0.326 | 11.590 ± 0.326 |
| GSM8K, strict match | 87.3 % ± 2.7 | 87.3 % ± 2.7 | 88.7 % ± 2.6 |
| GSM8K, flexible extract | 88.0 % ± 2.7 | 88.0 % ± 2.7 | 89.3 % ± 2.5 |
| IFEval, prompt-level strict | 74.7 % ± 3.6 | 74.7 % ± 3.6 | 74.7 % ± 3.6 |
| IFEval, instruction-level strict | 82.4 % | 82.4 % | 81.9 % |
| Prompt processing, pp512 (tok/s) | 12.61 ± 0.19 | 12.26 ± 0.20 | 12.68 ± 0.12 |
| Generation, tg128 (tok/s) | 7.46 ± 0.03 | 7.24 ± 0.11 | 7.26 ± 0.10 |
| SHA-256 (first 16 chars) | `17ef842e47450cae` | `facaba48556e16a0` | `082b16b8b90a64cc` |

Full hashes are in `results/sha256_*.txt`.

## Findings

1. **Our packing reproduces PrismML's exactly.** The control matches the official model with a KL divergence of 0 and 100% top-1 agreement. The files differ only in metadata. So any difference in the abliterated column comes from the abliteration, not from our pipeline.
2. **The abliteration is almost entirely lost in packing.** The abliterated model differs from the official one by a KL divergence of 0.00019 and picks a different top token 0.9% of the time. It refuses every harmful prompt the official model refuses, and most replies are identical.
3. **Capability is unchanged.** Perplexity, GSM8K, IFEval, and speed are all within noise of the official model, which is consistent with the model being nearly unchanged.

**Likely explanation (not tested directly):** Q1_0 stores only the sign of each weight plus one shared scale per group of 128 weights. Abliteration makes small adjustments to the weights, and an adjustment smaller than a weight's magnitude doesn't flip its sign, so it disappears when rounded to 1-bit.

## Caveats

- **We did not run the FP16 abliterated model ourselves.** The 6/100 refusal figure for it comes from s3nh's model card, not from our harness. What we show is that whatever it did at FP16 doesn't carry over to Q1_0.
- **Our harness is stricter than Heretic's.** It counts 100/100 refusals for the official model, while s3nh's card reports 81/100 for the same weights (the unpacked model is the 1-bit model expanded, so this is a methodology difference, not a weight difference). Likely causes: thinking disabled, 150-token replies, and a different marker list. All three models here were measured identically, so the comparison between them holds.
- **Subsets, not full benchmarks.** GSM8K and IFEval use the first 150 items, and perplexity uses the first 40 chunks of WikiText-2. Treat absolute scores as approximate; the side-by-side comparison is the point.
- **One repack method.** We used mainline llama.cpp's `llama-quantize`. A packing method designed to preserve the edit, or abliteration applied in a quantization-aware way, might behave differently.
- **CPU-only speeds.** Measured without the speculative drafter, so they're not comparable to GPU numbers.

## Method

- **Hardware:** AMD EPYC-Rome, 16 threads, 30 GiB RAM, CPU only. llama.cpp mainline commit `feb9a3d6d`.
- **Repacks:** `convert_hf_to_gguf.py --outtype f16`, then `llama-quantize --pure … Q1_0`, so embeddings and the LM head are 1-bit too, matching the official file.
- **Perplexity and KL divergence:** `llama-perplexity`, context 512, first 40 chunks of the WikiText-2 test set, KL measured against the official model's logits.
- **GSM8K (5-shot, multi-turn chat) and IFEval:** lm-evaluation-harness through llama-server's chat endpoint, first 150 items per task, greedy decoding, thinking disabled (`enable_thinking: false`) for all models.
- **Refusals:** first 100 prompts of the `mlabonne/harmful_behaviors` test split (the split Heretic reports on), greedy decoding, 150-token replies. Scored two ways: an explicit-refusal phrase list ("I can't", "I cannot", "sorry", …) and a stricter list modeled on Heretic's that also counts warnings such as "illegal" or "harmful".
- **Speed:** `llama-bench -p 512 -n 128`.

## Reproduce

```bash
bash run_bonsai_bench.sh 2>&1 | tee run.log
```

`run_bonsai_bench.sh` builds llama.cpp, downloads and repacks both FP16 models, runs every benchmark, and writes `results/RESULTS.md`. It needs about 125 GB of free disk at peak and roughly 30 GB of RAM. `refusal_eval.py` and `summarize.py` must sit next to it. Raw replies to the harmful prompts are deliberately not published here; only the counts are.

## Credits

[PrismML](https://huggingface.co/prism-ml) for Bonsai-27B and the unpacked weights, [s3nh](https://huggingface.co/s3nh) for the abliterated FP16 model, [Heretic](https://github.com/p-e-w/heretic) for the abliteration method, and [llama.cpp](https://github.com/ggml-org/llama.cpp) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the tooling.

*Benchmark run October 2026.*
