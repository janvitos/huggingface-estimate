# AGENTS.md

## Structure

Two files — no build step, no dependencies. Loads `@huggingface/gguf` from jsDelivr CDN (`0.4.2`).

- **`calculations.js`** — Pure calculation module with architecture registry, KV cache, activations, MoE, and weight calculations
- **`index.html`** — Display layer with HTML/CSS, GGUF parsing, HF API resolution, and result rendering

## Must serve via HTTP

`file://` protocol blocks CORS headers needed to fetch GGUF metadata from HuggingFace. Run:
```bash
python3 -m http.server 8000
```
Then visit `http://localhost:8000`.

## BigInt gotcha

`@huggingface/gguf` returns `tensorInfos[].shape` as `bigint[]`. Never multiply a `bigint` by a `number` (BPE values are `number`). Always convert first:
```js
// WRONG: t.shape.reduce((a, b) => a * Number(b), 1n)  // 1n * Number → TypeError
// CORRECT: t.shape.map(Number).reduce((a, b) => a * b, 1)
```

## HF URL normalization

HF's `/blob/` endpoint lacks CORS headers. Always normalize user-pasted URLs:
```js
url = path.replace(/\/blob\//, '/resolve/').replace(/#.*$/, '');
```

## MoE VRAM/RAM split

- **Dense**: all weights + KV + activations → VRAM
- **MoE**: only `expert_used_count` experts in VRAM, `(expert_count - expert_used_count)` experts in RAM
- KV cache and activations always in VRAM
- VRAM fit check compares against `vramBytes` (not `totalBytes`)

## CDN version pin

`@huggingface/gguf` imported from `https://cdn.jsdelivr.net/npm/@huggingface/gguf@0.4.2/+esm`. Check for updates via `https://data.jsdelivr.com/v1/package/npm/@huggingface/gguf`.

## Bytes-per-element hardcoded

`GGML_QUANT_SIZES` is NOT exported from the browser build. BPE values are hardcoded as the `BPE` object in `calculations.js` (lines 5–40). Use `GGMLQuantizationType` enum keys as indices.

## Talking to huggingface

Use node and the `@huggingface/gguf` package when you need metadata or examples.

## Adding a new architecture

New architectures are added to the `ARCHITECTURES` registry in `calculations.js` (lines 52–447). Each entry declares categories and provides handlers for KV cache, activations, and MoE weights.

### Step 1: Identify the architecture

Parse the GGUF file to find the `general.architecture` value and metadata keys:

```bash
node --experimental-vm-modules -e "
import { gguf } from '@huggingface/gguf';
const r = await gguf('https://huggingface.co/owner/model/resolve/main/model.gguf');
console.log(r.metadata['general.architecture']);
console.log(Object.keys(r.metadata).filter(k => k.startsWith(r.metadata['general.architecture'] + '.')));
"
```

### Step 2: Determine categories

Match the architecture to one of these categories:

| Category | Trigger | What changes |
|----------|---------|-------------|
| `mla` | Has `attention.kv_lora_rank` + `attention.key_length_mla` | KV cache uses compressed latent dimensions instead of head counts; activations use `q_lora_rank` + `kv_lora_rank` |
| `iswa` | Has `attention.sliding_window` or per-layer `head_count_kv` array | Reads `head_count_kv` as array for per-layer GQA |
| `moe` | Has `expert_count > 0` | Expert tensor grouping, VRAM/RAM split for inactive experts |

### Step 3: Add registry entry

Add a new entry to `ARCHITECTURES` object. Copy an existing handler and modify:

```js
myarch: {
  name: 'myarch',
  categories: ['transformer', 'moe', 'iswa'],
  kvCache(meta, ctxSize, kvTypeK, kvTypeV) {
    // Return { bytesK, bytesV, totalBytes, layers, headsK, headsV, totalHeadsKV, avgHeadsKV }
    const arch = meta['general.architecture'];
    // ... calculation logic ...
  },
  activations(meta, ctxSize, batchSize) {
    // Return { totalBytes, perLayerBytes, isMoe, expertCount, expertUsedCount, expertFF }
    const arch = meta['general.architecture'];
    // ... calculation logic ...
  },
  moe(meta, tensorInfos) {
    // Return { expertCount, expertUsedCount, expertWeightBytes, routerBytes, sharedBytes, ... }
    const arch = meta['general.architecture'];
    // ... tensor matching logic ...
  },
  tensorGroups: {
    expert: ['*ffn_gate_exps*', '*ffn_up_exps*', '*ffn_down_exps*'],
    router: ['*ffn_gate_inp*'],
    shared: ['*ffn_gate_shexp*', '*ffn_up_shexp*', '*ffn_down_shexp*'],
  },
},
```

### Key patterns for tensor matching

- **Expert weights**: tensors containing `_exps.` in the name
- **Router/gate**: tensors containing `ffn_gate_inp` or `attn_sinks`
- **Shared experts**: tensors containing `_shexp.` or `_chexp.`
- **MoE bias**: tensors containing `exp_probs_b`

Use glob patterns in `tensorGroups` for flexible matching:
- `*` matches any characters
- Patterns are converted to regex internally via `globMatch()`

### Special cases

**MLA (DeepSeek2-style)**: KV cache uses `kv_lora_rank` for K and `key_length_mla` for V:
```js
const totalElemsK = n_layer * kv_lora_rank * ctxSize;
const totalElemsV = n_layer * key_length_mla * ctxSize;
```

MLA activations use `q_lora_rank` + `kv_lora_rank` as attention output dimensions instead of `n_embd`.

**ISWA (Gemma4/Llama4-style)**: `head_count_kv` may be an array (per-layer):
```js
const n_head_kv_arr = Array.isArray(n_head_kv)
  ? (() => { const a = Array(n_layer).fill(n_head[0] || 1); for (let i = 0; i < n_layer; i++) if (n_head_kv[i]) a[i] = Number(n_head_kv[i]); return a; })()
  : Array(n_layer).fill(n_head_kv);
```

**Multiple router tensors**: Some architectures (Gemma4, GPT-OSS) have 2 `ffn_gate_inp` tensors per block. Use `.filter()` instead of `.find()`.

### Step 4: Test

Verify with the target GGUF file:

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

Then open in browser and load the model to verify calculations.

### Fallback behavior

If an architecture is not in the registry, `getArchHandler()` falls back to the `llama` handler (standard transformer). Unknown architectures log a warning to the console.