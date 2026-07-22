# KV cache shapes of the four served models

What the KV cache of each model in this directory actually looks like, derived from each
model's `config.json` on disk (paths below). Explains why the per-model scripts differ in
`NEED_CHUNK` while the `--kv-transfer-config` JSON is byte-identical for all of them: the
connector config only says *which server to dial*; the shape/layout is handed to the server
automatically at worker startup via `REGISTER_KV_CACHE` (see
`code_structure/request_lifecycle.md`, "Registration prerequisite").

All sizes are per **one** full copy of the cache (TP splits K/V heads across ranks for GQA;
MLA latent handling under TP is backend-specific and not covered here).

## Summary table

| Model | Structure | KV per token | 20k-token doc | LMCache view | chunk |
|---|---|---|---|---|---|
| Qwen3-8B | GQA × 36 layers | 144 KB | ~2.8 GB | 1 group | free (16) |
| Qwen3.5-9B | GQA × 8 + linear-attn × 24 | 32 KB (+ ~50 MB/seq constant) | ~0.64 GB live; ~2.5 GB archived¹ | attn group + block-boundary state snapshots | **= 528** (hard) |
| DeepSeek-V4-Flash | MLA + sparse indexer × 43 | ~34 KB | ~0.7 GB | fp8 latent group + fp8 indexer group, interleaved | 256 (recipe) |
| Kimi-K2.6 | MLA × 61 | ~69 KB | ~1.4 GB | 1 latent group | 256 (recipe) |

¹ *live* = a running sequence's footprint; *archived* = the same prefix kept for reuse
under `--mamba-cache-mode align`, which adds a ~50 MB state checkpoint per 528-token block
(≈ 96 KB/token) — see the three-ledger note in the Qwen3.5-9B section.

## Qwen3-8B — plain GQA (`/dev/shm/models/Qwen3-8B/config.json`)

36 layers, 8 KV heads × head_dim 128, bf16. vLLM tensor per layer:

```
(2, num_blocks, block_size=16, 8, 128)      # 2 = K and V
```

Per token per layer: 2 × 8 × 128 × 2 B = 4 KB → **144 KB/token** over 36 layers.
The only model here whose cache is "just K and V". Single group; chunk-size is a free
performance knob, hence `NEED_CHUNK=16` default.

## Qwen3.5-9B — hybrid mamba (`/dev/shm/models/Qwen3.5-9B/config.json`, `text_config`)

32 layers, `layer_types` = 1 full-attention every 4 layers (8 total), the other 24 are
linear attention. Two different cache kinds coexist:

- **8 full-attn layers**: GQA, 4 KV heads × head_dim 256 → 4 KB/token/layer,
  **32 KB/token** total. Grows with sequence length as usual.
- **24 linear-attn layers**: **no per-token KV.** Each layer holds a fixed-size recurrent
  state per sequence: SSM state `(32 v_heads, 128, 128)` float32 ≈ 2 MB, plus a small conv
  state (kernel 4, ~64 KB). ≈ **50 MB per sequence, constant** in length.

Why block/chunk = 528: vLLM pools both kinds in one paged allocator, so one attention page
(528 tok × 4 KB ≈ 2.16 MB) must be ≥ one mamba state page (≈ 2.1 MB) — the startup log's
"Setting attention block size to 528" is exactly this inequality. The linear-layer state
only has snapshots at block boundaries, so LMCache's chunk **must equal** the unified block
size: `NEED_CHUNK=528` is structural, not tuning. Enforced mechanically at registration
(`register_kv_caches` raises if chunk % tokens_per_block != 0).

### How linear-attn prefix caching works, and what it costs

Attention KV is a **log** (per-token, append-only, any prefix usable); mamba state is a
**snapshot** (an irreversible compression of tokens 1..t — resumable only from a saved
copy). `--mamba-cache-mode align` therefore checkpoints: at every 528-token block boundary
the 24 layers' SSM states (~50 MB total) are copied into that block's mamba pages. On a
prefix hit, vLLM's `MambaManager.find_longest_cache_hit` searches **right-to-left and
stops at the first match** — one snapshot summarizes the whole prefix, so only the
rightmost matched block is needed (earlier positions filled with null blocks), while the
attention groups still need every matched block's KV. Hit granularity is locked to
multiples of 528. A running request keeps only its latest state block and frees older ones
back to the pool, where they stay hash-addressable (LRU-evictable) for other requests.

**The three-ledger accounting** (why "linear saves memory" needs qualification):

1. **Per-step compute/bandwidth — the real win, O(L) → O(1).** Each decode step a
   full-attn layer reads all past KV (dense Qwen3-8B at 128k ctx: ~18 GB/step); a linear
   layer reads+updates a fixed 2 MB state, independent of context length. Prefill: O(L²)
   → O(L).
2. **Resident memory of a live request — also saved.** 32 KB/token + 50 MB constant vs
   144 KB/token dense; at 128k ctx that is ~4 GB vs ~18 GB → ~4× more concurrent seqs.
3. **Prefix-cache archival — NOT saved, actually worse per token.** Checkpoints amortize
   to 50 MB / 528 ≈ 96 KB/token; add the 32 KB/token attention KV and cached prefixes
   cost ~128 KB/token — nearly dense parity. Snapshots share nothing with each other,
   so the archive is coarse-grained and expensive. Mitigations: it is a *feature* cost
   (mode=none zeroes it, ledgers 1–2 keep their savings), it is droppable (eviction only
   lowers hit rate, never correctness — dense KV of a live request can't be dropped),
   and checkpoint spacing is a latent knob (vLLM aligns per-block, the densest choice).

Linear attention saves the **time** dimension (per-step compute/bandwidth), not the
**cross-request reuse** dimension (archival for resumability). That asymmetry is the deep
reason mamba prefix caching is marked experimental and non-bit-exact in the recipe — and
why 96 KB/token of cold snapshot archive is a prime candidate for L2 offload (Part 2's
story applies *more*, not less, to hybrid models).

## DeepSeek-V4-Flash — MLA + sparse indexer (`/data1/bo/models/DeepSeek-V4-Flash/config.json`)

43 layers. `num_key_value_heads: 1` means MLA: per token per layer a single compressed
latent (512 latent + 64 rope = 576 dims) shared by all 64 attention heads. On top of MLA
sits DSA (DeepSeek sparse attention, V3.2 lineage): a lightweight per-layer **indexer**
(`index_n_heads 64, index_head_dim 128`) scores every past token in fp8 and picks the
top-512 (`index_topk`); the main MLA attention then only attends to those 512 positions
(plus the 128-token local `sliding_window`). Sparsity cuts attention compute/reads, **not**
KV storage — any old token may be selected later, so both caches keep every token:

- **MLA latent**: `(num_blocks, block_size, 576)` packed as `fp8_ds_mla`
  (~0.66 KB/token/layer incl. scales — approximate, see caveats)
- **indexer K cache**: one 128-dim fp8 key/token/layer + fp32 scale = 132 B
  (vLLM `DeepseekV32IndexerCache`, `deepseek_v2.py` `Indexer`; MQA-style, not per-head)

≈ **34 KB/token** over 43 layers — an order of magnitude below what a dense cache of this
width would cost. These are the interleaved KV cache groups the `DeepSeek-V4-Flash.sh`
header mentions (its "float32 indexers" phrasing overstates it: fp32 is only the scale
segment); LMCache picks both groups up from `engine_group_infos` at registration, no extra
connector config. Small KV per token is why this model sits near the transfer break-even
line (pending experiment D).

## Kimi-K2.6 — classic MLA (`/data1/bo/models/Kimi-K2.6/config.json`, `text_config`)

DeepSeek-V3 lineage: `kv_lora_rank 512 + qk_rope_head_dim 64 = 576`. 61 layers, bf16,
no indexer:

```
(num_blocks, block_size, 576)               # per layer, single latent, no K/V split
```

576 × 2 B × 61 ≈ **69 KB/token**. A ~1T-param model with half the KV of Qwen3-8B.

## Caveats

- `fp8_ds_mla` per-token packing (fp8 payload + scale segments + bf16 rope segment) is
  estimated from vLLM's layout; exact bytes may differ by a few tens per token.
- Qwen3.5 conv state (~64 KB/layer) is ignored in the totals; SSM state dominates.
- "20k-token doc" column is one sequence's KV at 20 000 tokens, model weights excluded.
