#!/usr/bin/env bash
set -euo pipefail

# Build a self-contained folder containing the tracked source plus the prepared
# tokenized dataset. Raw and processed JSONL are deliberately excluded because
# Kaggle training needs only tokenizer.json, manifests, and mmap token shards.
ROOT="$(git rev-parse --show-toplevel)"
OUTPUT="${1:-${ROOT}/kaggle_upload/sslm_bundle}"
TOKENIZED="${ROOT}/data/tokenized/local_100m"

if [[ ! -f "${TOKENIZED}/tokenizer.json" || ! -f "${TOKENIZED}/tokens/manifest.json" ]]; then
  echo "Prepared tokenized data is missing under ${TOKENIZED}." >&2
  echo "Run the local preparation commands in README.md first." >&2
  exit 1
fi

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"
git -C "${ROOT}" archive --format=tar HEAD | tar -xf - -C "${OUTPUT}"
mkdir -p "${OUTPUT}/data/tokenized"
cp -a "${TOKENIZED}" "${OUTPUT}/data/tokenized/local_100m"

git -C "${ROOT}" rev-parse HEAD > "${OUTPUT}/SOURCE_COMMIT"
{
  printf '# SHA-256 checksums for the uploaded tokenized dataset\n'
  (
    cd "${OUTPUT}"
    find data/tokenized/local_100m -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum
  )
} > "${OUTPUT}/data/tokenized/local_100m/SHA256SUMS"

echo "Created Kaggle bundle: ${OUTPUT}"
du -sh "${OUTPUT}"
echo "Upload this folder as a Kaggle Dataset, or zip it without changing its internal layout."
