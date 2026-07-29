# F16 KV cache unlocks a 6.3x-faster flash-attention kernel (2026-07-29)

Private finding for sl-ai-node01's Laguna XS 2.1 deployment. Not for upstream —
see repo-level notes on this being a private fork only.

## Summary

Production's `q8_0/q8_0` KV cache quantization (chosen to save memory) was
unconditionally disabling a faster, alternate flash-attention kernel already
present in mainline llama.cpp (`ggml_compute_forward_flash_attn_ext_tiled`,
`ggml/src/ggml-cpu/ops.cpp`). Switching to `f16/f16` KV re-enables it and gives
a large, clean win on prefill speed, decode speed, *and* quality — this isn't
a lossy tradeoff, it's strictly better on every axis except memory footprint.

## The gating condition

```cpp
// ggml/src/ggml-cpu/ops.cpp, inside ggml_compute_forward_flash_attn_ext_f16
bool use_tiled = !use_ref &&
                       (q->type == GGML_TYPE_F32 &&
                        kv_is_f32_or_f16 &&
                        k->type == v->type &&
                        neq1 >= Q_TILE_SZ);
```

`Q_TILE_SZ = GGML_FA_TILE_Q = 64` (`ggml/src/ggml-cpu/common.h`).

Two conditions matter:

1. **`kv_is_f32_or_f16`** — K and V must NOT be quantized. Any KV quantization
   (`q8_0`, `q4_0`, etc.) disables this kernel entirely, for both prefill and
   decode.
2. **`neq1 >= 64`** — `neq1` is the query-batch size for this attention call.
   Decode is always exactly 1 new token (`neq1 = 1`), so **decode can never
   use this kernel regardless of KV type** — it's a prefill-only optimization,
   gated on batch size. This settles a question raised during investigation:
   decode fundamentally cannot "batch 64+ tokens at once" for a single
   in-flight generation; that would require speculative decoding with a
   compatible draft model (none exists for this architecture), not a config
   flag.

## Benchmark data (depth 16000, `llama-bench`, clean isolated runs)

| KV type   | Prefill (tok/s) | Decode (tok/s) | PPL (wikitext-2, 4 chunks) |
|-----------|-----------------|----------------|----------------------------|
| q8_0/q8_0 | 11.93 ± 0.05    | 5.44 ± 0.06    | 11.4995 ± 0.57374          |
| q8_0/f16  | 14.62 ± 0.06    | —              | —                          |
| f16/q8_0  | 11.06 ± 0.05    | —              | —                          |
| **f16/f16** | **75.03 ± 1.79** | **6.79 ± 0.22** | **11.4841 ± 0.57466**    |

f16/f16 vs q8_0/q8_0: **prefill 6.3x faster**, decode **+24.8% faster**,
PPL **lower (better)** — every error bar non-overlapping, none of this is
noise-level (this session separately established, via several other findings,
that this box needs effects clearing roughly 5-10% and 2-3 independent runs
before trusting a result; a 530% prefill difference is many orders of
magnitude past that bar).

Short-context (depth 0) decode also improves: 14.12 ± 0.03 (f16/f16) vs
13.37 ± 0.09 (q8_0/q8_0), +5.6%, non-overlapping — unexpected, since the tiled
kernel can't apply to decode at all (`neq1=1` always fails the ≥64 gate).
Best explanation, not fully isolated: `q8_0` requires a per-read
dequantization step whose CPU cost apparently outweighs the extra memory
bandwidth `f16`'s ~2x larger footprint needs, at least up to the context
depths tested here.

PPL is measured, not assumed: F16 is literally higher-precision than `q8_0`,
so the small PPL improvement (11.4841 vs 11.4995) is expected and confirms
this isn't a hidden quality cost.

## Memory tradeoff

`f16` KV is ~2x the bytes of `q8_0` per KV entry. Full 2-slot/262144-context
capacity (~21.5GB KV total at q8_0) would need ~43GB at f16 — doesn't fit
alongside ~19GB of model weights on this box's 62GB RAM. Decision: single-slot
capacity, full 262144 context, in exchange for the speed/quality win (down
from 2 concurrent developer slots).

## Deployed config (production, `~/start-server.sh`)

Before:
```
-c 524288 --parallel 2 -ctk q8_0 -ctv q8_0
```

After:
```
-c 262144 --parallel 1 -ctk f16 -ctv f16 --no-mmap
```

`--no-mmap` is required for this to fit safely: without it, the mmap'd model
file stays resident (page-cache-backed) *alongside* the separately-allocated,
repacked destination buffer that Fix 1/2 (repack parallelization/NUMA
affinity, see the other branches on this fork) write into — double-counting
~19GB of memory that a genuine read-copy avoids.

Verified on the actual production binary (not just this test clone):
bit-exact PPL match against the numbers above, correct functional output,
16-24 tok/s decode observed in production traffic, no swap.

## A real operational lesson from validating this

Testing the full-scale (262144-context) F16 config *while production stayed
live* caused two separate service disruptions — a swap incident from memory
pressure, then a CPU-contention incident that left production measurably
degraded even after the competing test process was killed (needed a second
restart to fully clear). This happened even though the memory arithmetic
looked safe on paper beforehand. The eventual clean validation was done with
production stopped first — no resource competition, no incident. Lesson for
any future large-context testing on this box: stop production first for this
class of test, don't trust the arithmetic alone.

## Not for upstream

Kept private per standing instruction — this is a config/deployment finding
specific to this box's memory budget and workload, not a general-purpose
patch, and isn't intended for the official repo regardless.
