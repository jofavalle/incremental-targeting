#!/usr/bin/env bash
#
# download_data.sh: fetch Criteo-UPLIFT v2.1 and verify it against its published
# checksum.
#
# Usage: ./scripts/download_data.sh [RAW_DIR]     (default: data/raw)
#
# Source. The link on the Criteo AI Lab page (go.criteo.net) redirects to an
# Azure storage host that no longer resolves (checked 2026-10-16). The same file
# is published by the `criteo` organization on Hugging Face, whose LFS pointer
# states its SHA256 and size. Both are pinned below, together with the dataset
# revision, so the checksum comes from the source and not from a first download
# trusted blindly.
#
# The file stays compressed. Ingestion reads the .gz as a stream and never
# writes the ~3 GB CSV to disk.
#
# Idempotent: if the file is already there and its checksum matches, it does not
# download again. A file with the wrong checksum is never moved into place.

set -euo pipefail

FILE="criteo-research-uplift-v2.1.csv.gz"
REVISION="2424920019e49d52d72c13ac1143ec5d53af276b"
URL="https://huggingface.co/datasets/criteo/criteo-uplift/resolve/${REVISION}/${FILE}"
SHA256="2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
SIZE_BYTES=311422618
RAW_DIR="${1:-data/raw}"

for cmd in curl sha256sum; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: '${cmd}' was not found." >&2
    exit 1
  fi
done

mkdir -p "${RAW_DIR}"
target="${RAW_DIR}/${FILE}"

verify() {
  echo "${SHA256}  $1" | sha256sum --check --status
}

if [[ -f "${target}" ]]; then
  if verify "${target}"; then
    echo "${target} is already there and its SHA256 matches. Nothing to do."
    exit 0
  fi
  echo "ERROR: ${target} exists but its SHA256 does not match. Move it away first." >&2
  exit 1
fi

echo "Downloading ${FILE} (${SIZE_BYTES} bytes) into ${RAW_DIR} ..."
# --continue-at - resumes a partial download left by an interrupted run.
curl --fail --location --retry 3 --continue-at - --output "${target}.part" "${URL}"

actual_size=$(stat -c %s "${target}.part")
if [[ "${actual_size}" -ne "${SIZE_BYTES}" ]]; then
  echo "ERROR: expected ${SIZE_BYTES} bytes, got ${actual_size}. Kept ${target}.part." >&2
  exit 1
fi
if ! verify "${target}.part"; then
  echo "ERROR: SHA256 mismatch. Kept ${target}.part for inspection." >&2
  exit 1
fi
mv "${target}.part" "${target}"

cat <<'NOTICE'
SHA256 verified.

Criteo-UPLIFT is licensed under CC BY-NC-SA 4.0 (non-commercial, share alike).
It is never committed to this repository. Cite it as:
  Diemert, E., Betlei, A., Renaudin, C. and Amini, M.-R. (2018). A Large Scale
  Benchmark for Uplift Modeling. AdKDD and TargetAd Workshop, KDD, London.
NOTICE
