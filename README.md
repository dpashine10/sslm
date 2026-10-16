# Small selective-SSM language model

This repository implements a small decoder-only language model trained from
random initialization. Its sequence mixer is a **genuine input-selective state
space model**, not self-attention, an LSTM, or a GRU. The intended result is a
technically sound experimental text completer that can later be instruction
fine-tuned; a 50M model trained on a modest corpus should not be expected to
match ChatGPT.

The code is designed for CPU tests and for longer runs on a Kaggle CUDA GPU.
No dataset, checkpoint, credential, or pretrained model is committed.

## What is implemented

- PyTorch selective SSM blocks with causal depthwise convolution, residual
  connections, RMSNorm, and gated SwiGLU-style MLPs.
- Full-sequence batched scans for training and recurrent one-token `step()`
  updates for efficient generation.
- Tied token/output embeddings, configurable vocabulary, state width, context,
  layer count, and model width.
- `base_50m.yaml` (about 49.9M parameters with a 32k vocabulary) and
  `base_75m.yaml` (about 74.7M with a 32k vocabulary). Counts are printed at
  startup and can be measured with the command below; the exact count depends
  on the actual tokenizer vocabulary.
- Streaming Hugging Face acquisition with schema detection, retries, raw
  metadata, deterministic cleaning/deduplication/splitting, JSONL shards,
  train-only ByteLevel BPE tokenizer training, and memory-mapped token shards.
- Token budgets during encoding (`--max-tokens`) rather than pretending a
  document count is a token count.
- Optional Dolly/OASST/ShareGPT-style conversation normalization with
  assistant-only SFT loss masks.
- Mixed precision selection, warmup/cosine learning rate, gradient clipping,
  validation, throughput/memory logs, fixed samples, atomic resumable
  checkpoints, retention, and best-checkpoint preservation.
- CPU-compatible single-prompt generation and terminal chat.
- Offline tests covering shapes, causal loss, backpropagation, full-vs-step
  state equivalence, tokenizer/data filtering, checkpoints, and CPU inference.

## Architecture

For one inner channel and state dimension `N`, the selective mixer applies the
following causal recurrence for token `t`:

```text
A = -exp(A_log)                         # strictly negative, learned
delta_t = clamp(softplus(W_dt x_t), dt_min, dt_max)
h_t = exp(delta_t A) * h_(t-1)
      + delta_t * B_t * u_t
y_t = sum_n(C_t,n * h_t,n) + D * u_t
```

`B_t`, `C_t`, and `delta_t` are produced from the current token-dependent SSM
branch, so different tokens can write to or read from the recurrent memory in
different ways. `A < 0` makes the homogeneous dynamics contractive, while the
bounded positive `delta` and tanh-bounded `B/C` make initialization practical.
The actual implementation also applies a causal depthwise convolution before
the SSM and a learned gate after it. Each block is:

```text
x = x + SelectiveSSM(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

There is no attention matrix and no positional embedding. Information crosses
time only through the convolution history and `h_t`. During generation each
layer retains those states, so one new token does not recompute the whole
prefix.

## Reproducible setup

Python 3.11.2 is pinned in `.python-version`. A local virtual environment is
recommended:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e .
```

For a CPU-only install, use the CPU wheel index if necessary:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.2
python -m pip install -r requirements.lock --no-deps
```

Kaggle normally already has a CUDA-enabled PyTorch. The notebook installs the
non-PyTorch packages without replacing Kaggle's CUDA build. The Dockerfile
uses the pinned lockfile and CPU PyTorch wheels.

## Tests and smoke test

The one offline project test command is:

```bash
make test
```

It uses only local fixtures. A syntax check is:

```bash
make format-check
```

The model-only smoke test validates a forward pass, aligned causal loss,
backward pass, optimizer update, and recurrent generation without any dataset:

```bash
python -m src.training.train --config configs/tiny.yaml --dry-run
```

Or run both from the helper:

```bash
bash scripts/run_smoke_test.sh
```

The checked environment used CPU PyTorch and reported 13 passing tests plus a
successful tiny dry run. No GPU-specific benchmark or live network download is
claimed here.

## Data preparation

The default mixture script streams a configurable subset of FineWeb and
FineWeb-Edu, merges their raw JSONL files while preserving per-source
metadata, cleans/deduplicates them, trains a tokenizer on the training split
only, and tokenizes up to a token budget:

```bash
# Start with a small explicit validation run.
MAX_RECORDS=10000 TOKEN_BUDGET=1000000 bash scripts/prepare_data.sh

# Initial general-language target requested by this project.
TOKEN_BUDGET=100000000 bash scripts/prepare_data.sh
```

`MAX_RECORDS` is optional and is a record cap only. `TOKEN_BUDGET` is enforced
by `tokenize_data.py`, where the final record is truncated to fit. For a larger
experiment set it to `300000000` or `500000000` after validating storage and
throughput. The script does not run unless you invoke it.

The individual stages are also available:

```bash
python -m src.data.download_data --help
python -m src.data.prepare_data --help
python -m src.data.merge_raw --help
python -m src.data.train_tokenizer --help
python -m src.data.tokenize_data --help
```

The downloader uses `datasets.load_dataset(..., streaming=True)`, retries
failed streams, detects common fields such as `text`, `content`, `body`, and
`document`, and records dataset/config/split/revision/schema metadata plus an
explicit license field. It does
not bypass authentication or access restrictions. If a dataset card changes
its configuration name, pass the current name with `FINEWEB_CONFIG` or
`FINEWEB_EDU_CONFIG`.

After reviewing each current dataset card, set `FINEWEB_LICENSE` and
`FINEWEB_EDU_LICENSE` if you want the reviewed license identifiers copied into
the raw manifests. If they are omitted, the manifest records `license: null`
and a reminder rather than inventing a license.

The preparation stage normalizes Unicode and whitespace, removes empty/short or
overlong documents, deduplicates SHA-256 document hashes, and assigns a stable
hash split. Raw, processed, and tokenized data are separate directories. Token
shards are flat `uint16`/`uint32` files read with NumPy memmaps; they are not
loaded as one giant RAM array.

### Dataset terms

The supported source identifiers are public Hugging Face datasets:

| Stage | Identifier | Use in this repository |
|---|---|---|
| General text | `HuggingFaceFW/fineweb` | Default web-text source |
| Educational text | `HuggingFaceFW/fineweb-edu` | Quality-oriented mixture component |
| Alternative | `Skylion007/openwebtext` | Use only after reviewing its current card/terms |
| Small smoke corpus | `roneneldan/TinyStories` | Useful for tiny language-model checks |
| Optional SFT | `databricks/databricks-dolly-15k`, `OpenAssistant/oasst1` | Separate instruction stage |

Licenses and access conditions belong to each current dataset card and can
change. The generated raw `metadata.json` records the requested dataset,
configuration, split, revision, and any supplied license; review the cards and preserve those
records before redistribution. In particular, do not assume that a dataset is
permissively licensed just because it is public. This project does not package
or redistribute any of these datasets.

## Model parameter counts

```bash
python - <<'PY'
from src.models import ModelConfig, SSMLanguageModel
for name, values in {
    "50m": dict(vocab_size=32000, d_model=480, n_layers=12, d_state=16, max_seq_len=512),
    "75m": dict(vocab_size=32000, d_model=608, d_inner=1216, n_layers=12, d_state=16, max_seq_len=1024),
}.items():
    model = SSMLanguageModel(ModelConfig(**values))
    print(name, model.num_parameters())
PY
```

The current architecture reports approximately 49.9M and 74.7M respectively
at a 32k vocabulary. These are target-sized configurations, not pretrained
weights.

## Training

First validate the token manifest and model with the tiny config. A real run
must have `data.token_manifest` pointing at the directory containing
`manifest.json`, and the generation tokenizer path must point at
`tokenizer.json`.

```bash
python -m src.training.train --config configs/base_50m.yaml
```

The trainer detects CPU, CUDA, or MPS and prints PyTorch version, GPU name,
VRAM, device, precision, and parameter count. `precision: auto` selects BF16
when the CUDA device supports it, FP16 otherwise on CUDA, and FP32 on CPU.
The effective tokens per optimizer step are:

```text
batch_size * block_size * gradient_accumulation_steps
```

For `base_50m`, this is `8 * 512 * 8 = 32,768` tokens/step, so the default
100M-token budget is about 3,052 optimizer steps. Actual elapsed time depends
on the Kaggle GPU, data storage, precision, and validation frequency. Use the
reported `tok/s` from the smoke/real run to estimate duration:

```text
estimated hours = requested tokens / measured tokens_per_second / 3600
```

The training objective is next-token cross entropy. The dataset supplies a
window of input tokens; the model internally compares `logits[:, :-1]` to
`input_ids[:, 1:]`, so the shift is performed exactly once. The trainer checks
for non-finite loss, clips gradients, and logs validation loss, perplexity,
throughput, and GPU memory where available.

### Checkpoints and resume

Checkpoints are written atomically under the configured `output_dir` as
`checkpoint_step_XXXXXXXX.pt`. They include model, optimizer, scheduler,
scaler, step, token count, configuration, and Python/NumPy/Torch RNG states.
Incomplete `.tmp` files are ignored. The latest valid checkpoint is resumed
automatically when `training.resume: true`; the best validation checkpoint is
also copied to `best.pt`. Recent checkpoint retention is controlled by
`keep_checkpoints`.

At a completed run the trainer also writes `inference.pt`, a smaller
weights/config-only export with no optimizer, scheduler, or RNG state. It can
be used anywhere a full checkpoint is accepted by the inference commands.

Kaggle `/kaggle/working` is not automatically a durable cross-session dataset.
Before ending a session, save the notebook version/output or publish the
checkpoint directory and tokenizer as a versioned Kaggle Dataset. Attach that
dataset in the next session and point the YAML paths at the attached files.

## Evaluation and text generation

Evaluate a checkpoint on validation windows:

```bash
python -m src.training.evaluate \
  --checkpoint outputs/checkpoints/base_50m/best.pt \
  --token-manifest data/tokenized/general/tokens \
  --max-batches 50 \
  --output outputs/reports/base_50m-validation.json
```

The report includes loss, perplexity, checkpoint step, tokens processed,
throughput, parameter count, and measurable memory. Compare fixed prompts and
validation perplexity across checkpoints; do not judge success by loss alone.
Base-model completions are not instruction-following evaluation.

Single prompt, CPU-compatible inference:

```bash
python -m src.inference.generate \
  --checkpoint outputs/checkpoints/base_50m/best.pt \
  --tokenizer data/tokenized/general/tokenizer.json \
  --prompt "The history of computers" \
  --device cpu --max-new-tokens 128
```

Interactive chat:

```bash
python -m src.inference.chat \
  --checkpoint outputs/checkpoints/base_50m/best.pt \
  --tokenizer data/tokenized/general/tokenizer.json \
  --device cpu
```

Use `/exit` or `/quit` to leave. The process keeps only the supplied
conversation history within the model context; there is no persistent memory.
Use `--greedy` for deterministic decoding, or adjust temperature, top-k, and
top-p for sampling. A tokenizer vocabulary mismatch is reported as an error.

## Optional supervised fine-tuning

SFT is intentionally independent from base pretraining. First download an
allowed instruction dataset and inspect its card/license, then normalize it:

```bash
python -m src.data.download_data \
  --dataset databricks/databricks-dolly-15k \
  --split train --output-dir data/raw/dolly
python -m src.data.prepare_sft \
  --input data/raw/dolly/raw.jsonl \
  --output-dir data/processed/sft \
  --tokenizer data/tokenized/general/tokenizer.json \
  --max-length 512
```

The normalizer accepts Dolly `instruction/context/response` and common
`messages`, `conversations`, or `conversation` arrays from OASST/ShareGPT-style
records. It removes malformed/exact duplicate conversations, preserves
multiple turns, emits `loss_mask`, and reports truncation and assistant-token
statistics. Only assistant target positions contribute to the loss; user and
system tokens remain recurrent context.

Start the optional trainer from a base checkpoint:

```bash
python -m src.training.sft --config configs/sft.yaml
```

Edit `configs/sft.yaml` to point at `best.pt` and the prepared SFT directory.
A small 1--5M-token SFT experiment is a reasonable first target; a small
instruction dataset does not guarantee useful conversation quality.

## Kaggle notebook

`notebooks/kaggle_training.ipynb` prints the environment/GPU, installs the
compatible non-PyTorch dependencies, validates an attached token manifest,
runs tests and the dry run, and requires explicit switches before preparation,
training, or evaluation. It explains how to save `/kaggle/working/outputs` as
notebook output or a versioned Kaggle Dataset. It does not silently start a
multi-day download.

## Storage and limitations

- Token storage is approximately 2 bytes/token for vocabularies up to 65,535
  and 4 bytes/token otherwise, plus source JSONL and tokenizer overhead. A
  100M-token `uint16` training/validation set is roughly 200 MB of token data;
  raw web documents can be much larger.
- The reference scan has a Python loop over time, which is simple and
  kernel-free but not as fast as fused Mamba CUDA kernels. It is chosen for
  portability and auditability. Kaggle GPU throughput must be measured.
- Training from random initialization on 100M tokens is an experiment, not a
  guarantee of factuality, safety, or strong dialogue. Evaluate generations
  manually and with held-out text before relying on the model.
- The default generic-text stage is not instruction tuning. Use the separate
  SFT pipeline for assistant behavior, and review data rights before use.
- A validation split is deterministic by document hash, but an external source
  can itself contain duplicates or leakage; no web-corpus pipeline can prove
  perfect independence.

## Troubleshooting

- **No validation windows:** provide a validation shard with at least
  `block_size + 1` tokens or use `--split train` for a diagnostic evaluation.
- **Tokenizer/model mismatch:** use the tokenizer produced from the same
  training split and vocabulary as the checkpoint.
- **Out of memory:** lower batch size, increase gradient accumulation to keep
  effective tokens similar, lower block size, or use the tiny/50M config.
- **Network failure:** rerun the download command; it retries and discards a
  partial raw file. If a gated dataset is selected, authenticate through the
  supported Hugging Face mechanism rather than bypassing access controls.
- **Resume after Kaggle restart:** attach the saved checkpoint/tokenizer/data
  dataset, restore their paths in YAML, and leave `resume: true`.
