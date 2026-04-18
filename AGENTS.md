# AGENTS.md

## Structure

No build step, no framework. ESM throughout (`package.json` has `"type": "module"`).

**Browser files** (ESM via CDN imports from jsDelivr):
- `index.html` — UI: HTML/CSS, GGUF parsing, HF API resolution, result rendering
- `calculations.js` — Architecture registry, KV cache, activations, MoE, weight calculations
- `parsing.js` — GGUF metadata parsing + HF URL resolution

**Node CLI files** (ESM via npm, all in project root):
- `run-calc.js` — CLI entry point
- `node-calculations.js` — Mirror of `calculations.js` with npm imports
- `node-parsing.js` — Mirror of `parsing.js` with npm imports

**Critical pattern**: Browser and Node files share identical logic but differ only in import source (`https://cdn.jsdelivr.net/npm/@huggingface/gguf@0.4.2/+esm` vs `@huggingface/gguf`). Changes to calculation or parsing logic **must be applied to both file pairs**.

**Gitignored reference dirs**: `llama.cpp/`, `ik_llama.cpp/`, `gguf-parser-go/` — local clones for quantization type reference, not part of the app.

## Must serve via HTTP

`file://` blocks CORS headers needed to fetch GGUF metadata from HuggingFace:
```bash
python3 -m http.server 8000
```
Then visit `http://localhost:8000`.

## CLI usage

```bash
node run-calc.js bartowski/Llama-3.1-8B-Instruct-GGUF --ctx 8192 --kvTypeK Q8_0
node run-calc.js --batch testModels.list
```

Options: `--ctx N` (default 4096), `--batchSize N` (default 1), `--kvTypeK TYPE` (default F16), `--kvTypeV TYPE` (default F16). Batch file has one HF repo per line, `#` comments supported. Outputs JSON to stdout, progress to stderr.

## BigInt gotcha

`@huggingface/gguf` returns `tensorInfos[].shape` as `bigint[]`. Never multiply a `bigint` by a `number` (BPE values are `number`). Always convert first:
```js
// WRONG: t.shape.reduce((a, b) => a * Number(b), 1n)
// CORRECT: t.shape.map(Number).reduce((a, b) => a * b, 1)
```

## HF URL normalization

HF's `/blob/` endpoint lacks CORS headers. Always normalize:
```js
url = path.replace(/\/blob\//, '/resolve/').replace(/#.*$/, '');
```

## MoE VRAM/RAM split

- **Dense**: all weights + KV + activations → VRAM
- **MoE**: only `expert_used_count` experts in VRAM, `(expert_count - expert_used_count)` in RAM
- KV cache and activations always in VRAM
- VRAM fit check compares against `vramBytes` (not `totalBytes`)

## Bytes-per-element hardcoded

`GGML_QUANT_SIZES` is NOT exported from the browser build. BPE values are hardcoded as the `BPE` object in `calculations.js` / `node-calculations.js`. Standard types use `GGMLQuantizationType` enum keys as indices; ik_llama.cpp extensions use numeric IDs (e.g., `151` for Q8_KV). The `BPE` object is the sole source of truth for bytes-per-element.

## CDN version pin

Browser files import from `https://cdn.jsdelivr.net/npm/@huggingface/gguf@0.4.2/+esm`. Check updates at `https://data.jsdelivr.com/v1/package/npm/@huggingface/gguf`.

## Sharded GGUF

Files matching `*-of-*.gguf` are auto-detected as shards in `parseGGUF()`. Calls `ggufAllShards()`, merges tensor infos from all shards, takes metadata from the first shard.

## Adding a new architecture

Add to the `ARCHITECTURES` registry in **both** `calculations.js` and `node-calculations.js`. Each entry declares categories and provides handlers for KV cache, activations, and MoE weights.

### Step 1: Identify the architecture

```bash
node --experimental-vm-modules -e "
import { gguf } from '@huggingface/gguf';
const r = await gguf('https://huggingface.co/owner/model/resolve/main/model.gguf');
console.log(r.metadata['general.architecture']);
console.log(Object.keys(r.metadata).filter(k => k.startsWith(r.metadata['general.architecture'] + '.')));
"
```

### Step 2: Determine categories

| Category | Trigger | What changes |
|----------|---------|-------------|
| `mla` | Has `attention.kv_lora_rank` + `attention.key_length_mla` | KV cache uses compressed latent dimensions; activations use `q_lora_rank` + `kv_lora_rank` |
| `iswa` | Has `attention.sliding_window` or per-layer `head_count_kv` array | Per-layer GQA; SWA layers use `min(sliding_window, ctxSize)` |
| `moe` | Has `expert_count > 0` | Expert tensor grouping, VRAM/RAM split for inactive experts |
| `vl` / `embedding` / `diffusion` | Model type markers | Calculation-only markers, no handler changes |

### Step 3: Check for aliases

If the GGUF-returned architecture name differs from the registry key (e.g., hyphens vs underscores), add a mapping to `ARCH_ALIASES`:
```js
ARCH_ALIASES = { 'ernie4_5-moe': 'ernie4_5_moe', 'hunyuan-moe': 'hunyuan_moe', 'lfm2moe': 'lfm2_moe' }
```

### Step 4: Add registry entry

Most architectures reuse the shared `llamaKvCache`, `llamaActivations`, `llamaMoe` handlers. Only add custom handlers for non-standard KV cache (MLA), per-layer heads (ISWA), or special activation patterns:

```js
myarch: {
  name: 'myarch',
  categories: ['transformer', 'moe'],
  kvCache: llamaKvCache,
  activations: llamaActivations,
  moe: llamaMoe,
  tensorGroups: LLAMA_TENSOR_GROUPS,
},
```

### Key patterns for tensor matching

- **Expert weights**: `_exps.` in name
- **Router/gate**: `ffn_gate_inp` or `attn_sinks` or `altup_router`
- **Shared experts**: `_shexp.` or `_chexp.`
- **MoE bias**: `exp_probs_b`

Patterns use glob → regex conversion via `globMatch()`.

### Special cases

**MLA (DeepSeek2-style)**: KV cache uses `kv_lora_rank` for K, `key_length_mla` for V:
```js
const totalElemsK = n_layer * kv_lora_rank * ctxSize;
const totalElemsV = n_layer * key_length_mla * ctxSize;
```

**ISWA (Gemma4/Llama4-style)**: `head_count_kv` may be an array (per-layer):
```js
const n_head_kv_arr = Array.isArray(n_head_kv)
  ? (() => { const a = Array(n_layer).fill(n_head[0] || 1); for (let i = 0; i < n_layer; i++) if (n_head_kv[i]) a[i] = Number(n_head_kv[i]); return a; })()
  : Array(n_layer).fill(n_head_kv);
```

**Multiple router tensors**: Gemma4/GPT-OSS have 2 `ffn_gate_inp` per block. Use `.filter()`, not `.find()`.

**Leading dense blocks**: Some MoE architectures (`ernie4_5_moe`, `hunyuan_moe`, `lfm2_moe`, `afmoe`) have `leading_dense_block_count` — initial layers use standard FFN, later layers are MoE. Activations must split: `denseBytes + moeBytes`.

**Hybrid recurrent/attention**: `lfm2_moe` and `nemotron_h_moe` have per-layer `head_count_kv` where recurrent layers have `n_head_kv == 0` (no KV cache). Only sum KV for layers where `n_head_kv_arr[i] > 0`.

**Mixed DeltaNet/attention**: `qwen35moe` uses `full_attention_interval` — only every Nth layer has KV cache. Activations use `2 * n_embd + expertUsedCount * expertFF` (residual + shared + routed experts).

### Step 5: Test

```bash
node --experimental-vm-modules -e "
import { gguf } from '@huggingface/gguf';
const r = await gguf('https://huggingface.co/owner/model/resolve/main/model.gguf');
const arch = r.metadata['general.architecture'];
console.log('Arch:', arch);
console.log('Expert tensors:', r.tensorInfos.filter(t => t.name.includes('_exps.')).length);
console.log('Router tensors:', r.tensorInfos.filter(t => t.name.includes('ffn_gate_inp')).length);
"
```

Then open in browser and load the model to verify.

### Fallback

Unknown architectures fall back to the `llama` handler. A warning logs to the console.
