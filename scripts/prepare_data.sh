#!/usr/bin/env bash
set -euo pipefail

# This script never starts a download without an explicit command invocation.
# Set MAX_RECORDS for a small validation run; omit it only when intentionally
# selecting a larger stream. The token budget is enforced at tokenization time.
RAW_ROOT="${RAW_ROOT:-data/raw/general}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed/general}"
TOKEN_ROOT="${TOKEN_ROOT:-data/tokenized/general}"
TOKEN_BUDGET="${TOKEN_BUDGET:-100000000}"
FINEWEB_TOKEN_BUDGET="${FINEWEB_TOKEN_BUDGET:-70000000}"
FINEWEB_EDU_TOKEN_BUDGET="${FINEWEB_EDU_TOKEN_BUDGET:-30000000}"
MAX_RECORDS="${MAX_RECORDS:-}"
FINEWEB_CONFIG="${FINEWEB_CONFIG:-sample-10BT}"
FINEWEB_EDU_CONFIG="${FINEWEB_EDU_CONFIG:-sample-10BT}"
FINEWEB_REVISION="${FINEWEB_REVISION:-9bb295ddab0e05d785b879661af7260fed5140fc}"
FINEWEB_EDU_REVISION="${FINEWEB_EDU_REVISION:-87f09149ef4734204d70ed1d046ddc9ca3f2b8f9}"
FINEWEB_LICENSE="${FINEWEB_LICENSE:-}"
FINEWEB_EDU_LICENSE="${FINEWEB_EDU_LICENSE:-}"
DOWNLOAD_BACKEND="${DOWNLOAD_BACKEND:-viewer_api}"
VIEWER_PAGE_DELAY="${VIEWER_PAGE_DELAY:-0.25}"

download_args=()
if [[ -n "${MAX_RECORDS}" ]]; then
  download_args+=(--max-records "${MAX_RECORDS}")
fi
fineweb_license_args=()
fineweb_edu_license_args=()
if [[ -n "${FINEWEB_LICENSE}" ]]; then fineweb_license_args+=(--license "${FINEWEB_LICENSE}"); fi
if [[ -n "${FINEWEB_EDU_LICENSE}" ]]; then fineweb_edu_license_args+=(--license "${FINEWEB_EDU_LICENSE}"); fi

python -m src.data.download_data \
  --dataset HuggingFaceFW/fineweb \
  --config "${FINEWEB_CONFIG}" \
  --split train \
  --revision "${FINEWEB_REVISION}" \
  --backend "${DOWNLOAD_BACKEND}" \
  --page-delay "${VIEWER_PAGE_DELAY}" \
  --output-dir "${RAW_ROOT}/fineweb" \
  --max-tokens "${FINEWEB_TOKEN_BUDGET}" \
  "${download_args[@]}" "${fineweb_license_args[@]}"

python -m src.data.download_data \
  --dataset HuggingFaceFW/fineweb-edu \
  --config "${FINEWEB_EDU_CONFIG}" \
  --split train \
  --revision "${FINEWEB_EDU_REVISION}" \
  --backend "${DOWNLOAD_BACKEND}" \
  --page-delay "${VIEWER_PAGE_DELAY}" \
  --output-dir "${RAW_ROOT}/fineweb-edu" \
  --max-tokens "${FINEWEB_EDU_TOKEN_BUDGET}" \
  "${download_args[@]}" "${fineweb_edu_license_args[@]}"

python -m src.data.merge_raw --input-dir "${RAW_ROOT}" --output-dir "${RAW_ROOT}/merged"
python -m src.data.prepare_data \
  --input "${RAW_ROOT}/merged" \
  --output-dir "${PROCESSED_ROOT}" \
  --validation-ratio 0.01 \
  --seed general-v1 \
  --min-chars 200 \
  --max-chars 200000 \
  --shard-size 100000
python -m src.data.train_tokenizer \
  --processed-dir "${PROCESSED_ROOT}" \
  --output-dir "${TOKEN_ROOT}" \
  --vocab-size 32000 \
  --min-frequency 2
python -m src.data.tokenize_data \
  --processed-dir "${PROCESSED_ROOT}" \
  --tokenizer "${TOKEN_ROOT}/tokenizer.json" \
  --output-dir "${TOKEN_ROOT}/tokens" \
  --max-tokens "${TOKEN_BUDGET}"

echo "Prepared approximately ${TOKEN_BUDGET} tokens under ${TOKEN_ROOT}."
