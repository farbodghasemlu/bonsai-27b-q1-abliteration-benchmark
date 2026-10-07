"""Collects everything in the results dir into one Markdown report on stdout.

Usage: summarize.py RESULTS_DIR   (can be run mid-way; unfinished steps show as "—")
"""
import glob
import json
import re
import sys
from pathlib import Path

res = Path(sys.argv[1])
NAMES = ["official", "control", "abliterated"]
settings = json.loads((res / "settings.json").read_text())


def grab(path, pattern):
    """Missing file means the step hasn't run yet. A file that doesn't parse is a real error."""
    if not path.exists():
        return None
    m = re.search(pattern, path.read_text())
    if not m:
        raise SystemExit(f"Could not parse {path} with /{pattern}/ — check that log.")
    return m.groups()


def pm(g, suffix=""):
    return "—" if g is None else f"{g[0]} ± {g[1]}{suffix}"


def speed(n, test):
    p = res / f"speed_{n}.md"
    if not p.exists():
        return "—"
    rows = [[c.strip() for c in line.strip().strip("|").split("|")]
            for line in p.read_text().splitlines() if line.startswith("|")]
    header = rows[0]
    for r in rows[2:]:
        if r[header.index("test")] == test:
            return r[header.index("t/s")]
    raise SystemExit(f"No {test} row in {p}")


def ppl(n):
    if n == "official":
        return pm(grab(res / "ppl_official.log", r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)"))
    return pm(grab(res / f"kld_{n}.log", r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)"))


def kld(n):
    if n == "official":
        return "baseline"
    return pm(grab(res / f"kld_{n}.log", r"Mean\s+KLD:\s*([-\d.]+)\s*±\s*([\d.]+)"))


def top1(n):
    if n == "official":
        return "baseline"
    return pm(grab(res / f"kld_{n}.log", r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)\s*%"), " %")


def refusals(n):
    p = res / f"refusals_{n}.json"
    if not p.exists():
        return "—"
    r = json.loads(p.read_text())
    return f"{r['refused']}/{r['total']}"


LMEVAL = [
    ("GSM8K, strict match", "gsm8k", "exact_match,strict-match"),
    ("GSM8K, flexible extract", "gsm8k", "exact_match,flexible-extract"),
    ("IFEval, prompt-level strict", "ifeval", "prompt_level_strict_acc,none"),
    ("IFEval, instruction-level strict", "ifeval", "inst_level_strict_acc,none"),
]


def lmeval(n, task, key):
    files = sorted(glob.glob(str(res / f"lmeval_{n}" / "*" / "results_*.json")))
    if not files:
        return "—"
    r = json.loads(Path(files[-1]).read_text())["results"][task]
    value = f"{r[key] * 100:.1f} %"
    stderr = r.get(key.replace(",", "_stderr,"))
    # lm-eval writes "N/A" for metrics without a stderr, so only numbers get a ± part.
    return f"{value} ± {stderr * 100:.1f}" if isinstance(stderr, (int, float)) else value


def sha(n):
    p = res / f"sha256_{n}.txt"
    return p.read_text().split()[0][:16] if p.exists() else "—"


rows = [
    ("Refusals (harmful prompts, lower = less censored)", refusals),
    ("KL divergence vs official (lower = closer)", kld),
    ("Same top-1 token as official", top1),
    ("Perplexity, WikiText-2", ppl),
    *[(label, lambda n, t=task, k=key: lmeval(n, t, k)) for label, task, key in LMEVAL],
    ("Prompt processing, pp512 (tok/s)", lambda n: speed(n, "pp512")),
    ("Generation, tg128 (tok/s)", lambda n: speed(n, "tg128")),
    ("SHA-256 (first 16 chars)", sha),
]

s = settings
print(f"""# Bonsai-27B 1-bit: official vs. repacked vs. abliterated

| | Source |
|---|---|
| **official** | `prism-ml/Bonsai-27B-gguf`, PrismML's own Q1_0 file |
| **control** | `prism-ml/Bonsai-27B-unpacked` (FP16), repacked to Q1_0 by us |
| **abliterated** | `s3nh/Bonsai-27B-unpacked-abliterated-uncensored` (FP16, Heretic v1.4.0), repacked to Q1_0 by us |

Official vs. control shows what our packing process costs. Control vs. abliterated shows what
abliteration costs at 1-bit, and whether the refusal removal survives packing.

## Results

| Metric | official | control | abliterated |
|---|---|---|---|""")
for label, fn in rows:
    print(f"| {label} | " + " | ".join(fn(n) for n in NAMES) + " |")

print(f"""
## Method

- Hardware: {s['hardware']}. llama.cpp (mainline) commit `{s['llama_cpp_commit']}`.
- Repacks: `convert_hf_to_gguf.py --outtype f16`, then `llama-quantize --pure ... Q1_0`
  so embeddings and LM head are 1-bit too, matching the official file.
- Perplexity and KL divergence: `llama-perplexity`, context 512, first {s['ppl_chunks']} chunks of the
  WikiText-2 test set, KL measured against the official model's logits.
- GSM8K (5-shot, multi-turn chat) and IFEval: lm-evaluation-harness via llama-server's chat endpoint,
  first {s['eval_limit']} items per task, greedy decoding, thinking disabled (`enable_thinking: false`)
  for all models. Scores are on this subset, not the full sets.
- Refusals: first {s['refusal_n']} prompts of the `mlabonne/harmful_behaviors` test split, greedy,
  150-token replies, keyword-matched with a marker list modeled on Heretic's. Strict: replies that
  comply but add a warning also count as refusals.
- Speed: `llama-bench -p 512 -n 128`, speculative drafter not used.""")
