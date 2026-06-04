#!/usr/bin/env bash
# Bootstrap Modal storage for HNMD / filltables experiment artifacts.
#
# Uses one Modal Volume (default: hnmd-workspace) with the same layout as the repo:
#   /data            benchmark CSVs (read mostly, required before training)
#   /results         metrics.npy, pred.npy, true.npy  (filltables reads this)
#   /checkpoints     checkpoint.pth, training_state.pth (resume / recover)
#   /test_results    PDF plots from test() (persistent; not used by filltables)
#   /tables          filled_tables.xlsx (optional; usually built locally)
#
# These dirs are gitignored — too large for GitHub. Modal is the intended remote store.
#
# Prerequisites (activate your venv first if modal is installed there):
#   source env/bin/activate
#   pip install modal   # if needed
#   modal setup
#
# Usage:
#   ./modal_setup.sh upload          # create volume + upload local artifacts
#   ./modal_setup.sh upload --data   # upload only ./data
#   ./modal_setup.sh download        # pull all artifact dirs back to the repo
#   ./modal_setup.sh download --results --checkpoints
#   ./modal_setup.sh status          # list remote volume contents
#   ./modal_setup.sh create          # create empty volume only

set -euo pipefail

VOLUME_NAME="${HNMD_MODAL_VOLUME:-hnmd-workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARTIFACT_DIRS=(data results checkpoints test_results tables)
DEFAULT_DOWNLOAD_DIRS=(results checkpoints test_results tables)
MODAL_BIN=""

log() {
  printf '==> %s\n' "$*"
}

warn() {
  printf '!!> %s\n' "$*" >&2
}

die() {
  warn "$*"
  exit 1
}

find_modal() {
  if [[ -n "$MODAL_BIN" ]]; then
    return 0
  fi
  if command -v modal >/dev/null 2>&1; then
    MODAL_BIN="$(command -v modal)"
    return 0
  fi
  for candidate in \
    "${REPO_ROOT}/env/bin/modal" \
    "${REPO_ROOT}/.venv/bin/modal" \
    "${REPO_ROOT}/venv/bin/modal" \
    "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/modal}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      MODAL_BIN="$candidate"
      return 0
    fi
  done
  return 1
}

require_modal() {
  find_modal || die "Modal CLI not found. Activate your venv (source env/bin/activate) or run: pip install modal && modal setup"
}

modal_cmd() {
  require_modal
  "$MODAL_BIN" "$@"
}

create_volume() {
  require_modal
  if modal_cmd volume list 2>/dev/null | grep -q "${VOLUME_NAME}"; then
    log "Volume already exists: ${VOLUME_NAME}"
  else
    log "Creating volume: ${VOLUME_NAME}"
    modal_cmd volume create "${VOLUME_NAME}"
  fi
}

dir_size_human() {
  local path="$1"
  if [[ -d "$path" ]]; then
    du -sh "$path" 2>/dev/null | awk '{print $1}'
  else
    echo "missing"
  fi
}

upload_dir() {
  local name="$1"
  local local_path="${REPO_ROOT}/${name}"
  local remote_path="/${name}"

  if [[ ! -d "$local_path" ]]; then
    warn "Skipping ${name}: ${local_path} not found (creating empty remote prefix on first run is OK)"
    return 0
  fi

  local count
  count="$(find "$local_path" -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$count" == "0" ]]; then
    warn "Skipping ${name}: directory is empty"
    return 0
  fi

  log "Uploading ${name}/ ($(dir_size_human "$local_path"), ${count} files) -> ${VOLUME_NAME}:${remote_path}"
  modal_cmd volume put "${VOLUME_NAME}" "${local_path}" "${remote_path}" --force
}

download_dir() {
  local name="$1"
  local local_path="${REPO_ROOT}/${name}"
  local remote_path="/${name}"

  mkdir -p "${local_path}"
  log "Downloading ${VOLUME_NAME}:${remote_path} -> ${local_path}/"
  modal_cmd volume get "${VOLUME_NAME}" "${remote_path}" "${local_path}" --force
}

show_status() {
  require_modal
  create_volume
  log "Volume: ${VOLUME_NAME}"
  echo
  for name in "${ARTIFACT_DIRS[@]}"; do
    echo "--- /${name} ---"
    modal_cmd volume ls "${VOLUME_NAME}" "/${name}" 2>/dev/null | head -20 || echo "(empty or missing)"
    local remote_count
    remote_count="$(modal_cmd volume ls "${VOLUME_NAME}" "/${name}" 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$remote_count" -gt 20 ]]; then
      echo "... (${remote_count} entries total)"
    fi
    echo
  done
  echo "Local sizes:"
  for name in "${ARTIFACT_DIRS[@]}"; do
    printf '  %-14s %s\n' "${name}/" "$(dir_size_human "${REPO_ROOT}/${name}")"
  done
}

print_next_steps() {
  cat <<EOF

Next steps:
  1. Run missing experiments on Modal (parallel T4 workers):
       modal run modal_app.py --parallel 4
     (use the same venv where modal is installed)

  2. Pull artifacts back when jobs finish:
       ./modal_setup.sh download

  3. Build paper tables locally from synced results:
       python filltables.py --no-run_missing

Notes:
  - Upload includes existing results/checkpoints so --skip_if_done works on Modal.
  - test_results/ PDFs are uploaded too if present (large but persistent).
  - Do not git-push results/ or test_results/; GitHub rejects files >100 MB.
  - Override volume name: HNMD_MODAL_VOLUME=my-volume ./modal_setup.sh upload

EOF
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  create      Create the Modal volume (${VOLUME_NAME}) if it does not exist
  upload      Create volume and upload local artifact directories
  download    Download artifact directories from the volume to this repo
  status      Show remote volume listing and local directory sizes

Upload/download filters (optional, combine as needed):
  --data
  --results
  --checkpoints
  --test-results   (maps to test_results/)
  --tables

Examples:
  ./modal_setup.sh upload
  ./modal_setup.sh upload --data --results
  ./modal_setup.sh download --results --checkpoints
  ./modal_setup.sh status

Environment:
  HNMD_MODAL_VOLUME   Modal volume name (default: ${VOLUME_NAME})

EOF
}

selected_dirs=()
parse_dir_flags() {
  selected_dirs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --data) selected_dirs+=("data") ;;
      --results) selected_dirs+=("results") ;;
      --checkpoints) selected_dirs+=("checkpoints") ;;
      --test-results) selected_dirs+=("test_results") ;;
      --tables) selected_dirs+=("tables") ;;
      *)
        die "Unknown flag: $1"
        ;;
    esac
    shift
  done
}

cmd="${1:-}"
shift || true

case "$cmd" in
  create)
    create_volume
    log "Done."
    ;;
  upload)
    parse_dir_flags "$@"
    require_modal
    create_volume
    if [[ ${#selected_dirs[@]} -eq 0 ]]; then
      selected_dirs=("${ARTIFACT_DIRS[@]}")
    fi
    log "Repo: ${REPO_ROOT}"
    log "Target volume: ${VOLUME_NAME}"
    warn "Large uploads (results/, test_results/) can take a while."
    for name in "${selected_dirs[@]}"; do
      upload_dir "$name"
    done
    log "Upload complete."
    print_next_steps
    ;;
  download)
    parse_dir_flags "$@"
    require_modal
    if [[ ${#selected_dirs[@]} -eq 0 ]]; then
      selected_dirs=("${DEFAULT_DOWNLOAD_DIRS[@]}")
    fi
    for name in "${selected_dirs[@]}"; do
      download_dir "$name"
    done
    log "Download complete."
    log "Build tables: python filltables.py --no-run_missing"
    ;;
  status)
    show_status
    ;;
  -h|--help|help|"")
    usage
    [[ -z "$cmd" ]] && exit 0 || exit 0
    ;;
  *)
    die "Unknown command: $cmd\nRun $(basename "$0") --help"
    ;;
esac
