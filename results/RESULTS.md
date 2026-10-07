# Bonsai-27B 1-bit: official vs. repacked vs. abliterated

| | Source |
|---|---|
| **official** | `prism-ml/Bonsai-27B-gguf`, PrismML's own Q1_0 file |
| **control** | `prism-ml/Bonsai-27B-unpacked` (FP16), repacked to Q1_0 by us |
| **abliterated** | `s3nh/Bonsai-27B-unpacked-abliterated-uncensored` (FP16, Heretic v1.4.0), repacked to Q1_0 by us |

Official vs. control shows what our packing process costs. Control vs. abliterated shows what
abliteration costs at 1-bit, and whether the refusal removal survives packing.

## Results

| Metric | official | control | abliterated |
|---|---|---|---|
| Refusals (harmful prompts, lower = less censored) | 100/100 | 100/100 | 100/100 |
| KL divergence vs official (lower = closer) | baseline | 0.000000 ± 0.000000 | 0.000187 ± 0.000004 |
| Same top-1 token as official | baseline | 100.000 ± 0.000 % | 99.137 ± 0.092 % |
| Perplexity, WikiText-2 | 11.5944 ± 0.32572 | 11.594395 ± 0.325716 | 11.590034 ± 0.325590 |
| GSM8K, strict match | 87.3 % ± 2.7 | 87.3 % ± 2.7 | 88.7 % ± 2.6 |
| GSM8K, flexible extract | 88.0 % ± 2.7 | 88.0 % ± 2.7 | 89.3 % ± 2.5 |
| IFEval, prompt-level strict | 74.7 % ± 3.6 | 74.7 % ± 3.6 | 74.7 % ± 3.6 |
| IFEval, instruction-level strict | 82.4 % | 82.4 % | 81.9 % |
| Prompt processing, pp512 (tok/s) | 12.61 ± 0.19 | 12.26 ± 0.20 | 12.68 ± 0.12 |
| Generation, tg128 (tok/s) | 7.46 ± 0.03 | 7.24 ± 0.11 | 7.26 ± 0.10 |
| SHA-256 (first 16 chars) | 17ef842e47450cae | facaba48556e16a0 | 082b16b8b90a64cc |

## Method

- Hardware: AMD EPYC-Rome Processor, 16 threads, 30 GiB RAM, CPU only. llama.cpp (mainline) commit `feb9a3d6d`.
- Repacks: `convert_hf_to_gguf.py --outtype f16`, then `llama-quantize --pure ... Q1_0`
  so embeddings and LM head are 1-bit too, matching the official file.
- Perplexity and KL divergence: `llama-perplexity`, context 512, first 40 chunks of the
  WikiText-2 test set, KL measured against the official model's logits.
- GSM8K (5-shot, multi-turn chat) and IFEval: lm-evaluation-harness via llama-server's chat endpoint,
  first 150 items per task, greedy decoding, thinking disabled (`enable_thinking: false`)
  for all models. Scores are on this subset, not the full sets.
- Refusals: first 100 prompts of the `mlabonne/harmful_behaviors` test split, greedy,
  150-token replies, keyword-matched with a marker list modeled on Heretic's. Strict: replies that
  comply but add a warning also count as refusals.
- Speed: `llama-bench -p 512 -n 128`, speculative drafter not used.
