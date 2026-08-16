#!/usr/bin/env bash
# package_run.sh — normalize one run's scattered evidence into an immutable run directory.
#
# Usage: package_run.sh <source_dir> <dest_run_dir>
#
# Run-directory contract (tasks/canon_system_redesign/phase1_run_contract.md, outer repo):
#   <dest_run_dir>/
#     PACKAGING.txt        what was kept / deduplicated / compressed, and completeness verdict
#     SHA256SUMS           every file below, excluding SHA256SUMS itself
#     verdict.json         classifier output if present in source; else synthesized INCONCLUSIVE
#     *.log.gz|zst         logs compressed, NEVER truncated
#     *.tar.gz|zst         tars compressed
#     everything else      copied byte-identical
#
# Rules encoded from real incidents:
#   - duplicates are dropped, first path wins (P38s12e: five copies of one pod log);
#   - SHA256SUMS must not include itself (P38s13a packaging bug);
#   - a missing core piece is INCONCLUSIVE, never silently accepted and never a reason
#     to refuse packaging (evidence preservation precedes judgment);
#   - compression only, no truncation.
set -euo pipefail

SRC=${1:?usage: package_run.sh <source_dir> <dest_run_dir>}
DST=${2:?usage: package_run.sh <source_dir> <dest_run_dir>}
[ -d "$SRC" ] || { echo "[package_run] source dir missing: $SRC" >&2; exit 2; }
mkdir -p "$DST"
[ -z "$(ls -A "$DST")" ] || { echo "[package_run] dest not empty (run dirs are immutable): $DST" >&2; exit 3; }

if command -v zstd >/dev/null 2>&1; then CZ="zstd -q -19"; EXT="zst"; else CZ="gzip -9"; EXT="gz"; fi

PKG="$DST/PACKAGING.txt"
{
  echo "package_run 2026-08 v1"
  echo "source: $SRC"
  echo "compressor: $EXT"
} > "$PKG"

declare -A SEEN   # sha -> kept relative path
kept=0; dropped=0
while IFS= read -r -d '' f; do
  rel=${f#"$SRC"/}
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  if [ -n "${SEEN[$sha]:-}" ]; then
    echo "dedup: dropped $rel (identical to ${SEEN[$sha]})" >> "$PKG"
    dropped=$((dropped+1)); continue
  fi
  SEEN[$sha]=$rel
  out="$DST/$(basename "$rel")"
  case "$rel" in
    *.log)      $CZ -c "$f" > "$out.$EXT"; echo "compress: $rel -> $(basename "$out").$EXT sha_orig=$sha" >> "$PKG" ;;
    *.tar)      $CZ -c "$f" > "$out.$EXT"; echo "compress: $rel -> $(basename "$out").$EXT sha_orig=$sha" >> "$PKG" ;;
    SHA256SUMS) echo "skip: source SHA256SUMS (regenerated)" >> "$PKG" ;;
    *)          cp -p "$f" "$out"; echo "copy: $rel sha=$sha" >> "$PKG" ;;
  esac
  kept=$((kept+1))
done < <(find "$SRC" -type f -print0 | sort -z)

# Core-piece completeness: raw log, pre-alignment, capsule, serving archive, classification.
declare -A CORE=(
  [raw_log]='*.log.'"$EXT"
  [pre_alignment]='pre*alignment*.json*'
  [capsule]='*capsule*.npz'
  [serving_archive]='*serving*.tar.'"$EXT"
  [classification]='*classif*.json'
)
missing=()
for k in "${!CORE[@]}"; do
  found=$(find "$DST" -maxdepth 1 -name "${CORE[$k]}" | head -1)
  [ -n "$found" ] || missing+=("$k")
done

if [ ${#missing[@]} -gt 0 ]; then
  if ! ls "$DST"/verdict.json >/dev/null 2>&1; then
    printf '{"verdict": "INCONCLUSIVE", "reason": "missing core pieces: %s", "packaged_by": "package_run v1"}\n' \
      "$(IFS=,; echo "${missing[*]}")" > "$DST/verdict.json"
  fi
  echo "completeness: INCONCLUSIVE missing=${missing[*]}" >> "$PKG"
else
  echo "completeness: all core pieces present" >> "$PKG"
fi

echo "kept=$kept dropped_duplicates=$dropped" >> "$PKG"   # finalize PACKAGING before hashing it

( cd "$DST" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' | sort -z \
  | xargs -0 sha256sum > SHA256SUMS )

echo "[package_run] OK dest=$DST kept=$kept dropped=$dropped missing_core=${#missing[@]}"
