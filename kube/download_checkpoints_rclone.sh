#!/usr/bin/env bash
# Download a checkpoint subdirectory from a Google Drive/rclone data root.
# Adapted from VLN-Challenge/download_checkpoints_rclone.sh. Uses the InternNav Drive
# account's own My Drive root; files live under the "InternNav/checkpoints" subfolder there,
# prepended automatically via DRIVE_PREFIX so callers don't need to type it.
#
# Convenient 2-arg form: give a LOCAL path (absolute — the one you know) and a CLOUD path
# (relative to the fixed Drive folder). When one arg is an absolute/explicit path (/ ./ ../) it
# is taken as the local side, the other as the cloud side (order-independent). If the cloud path
# ends in "/" (or is empty), the local basename is appended. When BOTH args are relative the
# legacy order applies: <cloud_rel> <local_rel>.
#
# Usage:
#   ./download_checkpoints_rclone.sh <local_path> <cloud_rel>       # local absolute
#   ./download_checkpoints_rclone.sh <cloud_rel> <local_rel>        # legacy, both relative
#   ./download_checkpoints_rclone.sh <src_data_root> <rel_src> <dst_data_root> <rel_dst>
#   ./download_checkpoints_rclone.sh --check <local_path> <cloud_rel>
#   ./download_checkpoints_rclone.sh --dry-run <local_path> <cloud_rel>   # print resolved paths only, no network
#   ./download_checkpoints_rclone.sh --list <local_path> <cloud_rel>      # list remote files that would be pulled
#   ./download_checkpoints_rclone.sh --auth [remote]
#
# Defaults:
#   SRC_DATA_ROOT=gdrive:  (My Drive root, no folder ID needed)
#   DST_DATA_ROOT=/ws/src/InternNav/checkpoints
#   DRIVE_PREFIX=InternNav/checkpoints  (subfolder under My Drive root; prepended to rel_src)
#
# Arguments:
#   rel_src        Relative path under src_data_root (cloud), below DRIVE_PREFIX, to copy.
#                  Ending in "/" -> append the local basename.
#   rel_dst        Relative path under dst_data_root (local) to write, OR an absolute/explicit
#                  path (/ ./ ../) used verbatim.
#
# Download an image_base checkpoint to an absolute local path:
#   ./download_checkpoints_rclone.sh \
#     /ws/src/InternNav/checkpoints/image_base/base_s1.fpv_s2.fpv_20260709_130547 \
#     image_base/
#   # <- Drive: My Drive/InternNav/checkpoints/image_base/base_s1.fpv_s2.fpv_20260709_130547
#
# First-time auth, if needed:
#   ./download_checkpoints_rclone.sh --auth gdrive
#   On a headless server choose N, run the printed authorize command on a browser machine,
#   then paste the token back into this machine.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCLONE_VERSION="${RCLONE_VERSION:-v1.74.3}"
RCLONE_BIN="${RCLONE_BIN:-}"
REMOTE="${RCLONE_REMOTE:-gdrive}"
SRC_DATA_ROOT="${SRC_DATA_ROOT:-${REMOTE}:}"
DST_DATA_ROOT="${DST_DATA_ROOT:-/ws/src/InternNav/checkpoints}"
DRIVE_PREFIX="${DRIVE_PREFIX:-InternNav/checkpoints}"
TRANSFERS="${TRANSFERS:-2}"
CHECKERS="${CHECKERS:-8}"
FOLDER_ID="${FOLDER_ID:-}"
RCLONE_SRC=""
LOCAL_DST=""

usage() {
  sed -n '2,41p' "$0"
  cat <<EOF_USAGE

Options:
  --auth [remote]       Create/reconnect an rclone Google Drive remote. Default: $REMOTE
  --check               Verify remote files against local destination only.
  --dry-run             Print the resolved rclone source + local destination paths and exit
                         (no network access, no auth required). Use this to confirm --dst-data-root
                         / --src-data-root and the src/dst args resolve to what you expect.
  --list                List the remote files under the resolved source (recursive) and exit,
                         without copying anything. Requires remote auth (network access).
  --remote <name>       Remote to use for Drive URL/folder-id roots. Default: $REMOTE
  --src-data-root <src> Override source data root. Default: $SRC_DATA_ROOT
  --dst-data-root <dst> Override destination data root. Default: $DST_DATA_ROOT
  -h, --help            Show this help.

Environment:
  SRC_DATA_ROOT    Default source root. Default: $SRC_DATA_ROOT
  DST_DATA_ROOT    Default destination root. Default: $DST_DATA_ROOT
  DRIVE_PREFIX     Subfolder prepended to rel_src for root-scoped sources (bare "remote:",
                    URL/drive-folder:/bare ID), so multiple projects can share one Drive root.
                    Default: $DRIVE_PREFIX
                    Set to "" to disable. Ignored for explicit "remote:path" overrides.
  RCLONE_BIN       Explicit rclone binary path.
  RCLONE_REMOTE    Default remote name. Default: gdrive
  RCLONE_VERSION   Auto-downloaded rclone version. Default: v1.74.3
  TRANSFERS        Parallel transfers. Default: 2
  CHECKERS         Parallel checkers. Default: 8
  EXTRA_ARGS       Additional rclone flags, for example: --drive-shared-with-me
EOF_USAGE
}

log() {
  printf '[rclone-checkpoints] %s\n' "$*"
}

trim_slashes() {
  local value="$1"
  value="${value#/}"
  value="${value%/}"
  printf '%s\n' "$value"
}

is_drive_folder_id() {
  case "$1" in
    *:*) return 1 ;;
    http://*|https://*|/*|./*|../*) return 1 ;;
  esac
  [ ${#1} -ge 20 ] && [[ "$1" =~ ^[A-Za-z0-9_-]+$ ]]
}

extract_drive_folder_id() {
  local src="$1"
  case "$src" in
    # Handles both /drive/folders/<id> and account-scoped /drive/u/<N>/folders/<id>.
    http://drive.google.com/*folders/*|https://drive.google.com/*folders/*)
      src="${src##*/folders/}"
      src="${src%%[/?#]*}"
      ;;
    https://drive.google.com/open\?id=*|http://drive.google.com/open\?id=*)
      src="${src#*id=}"
      src="${src%%&*}"
      ;;
    *)
      return 1
      ;;
  esac

  [ -n "$src" ] || return 1
  printf '%s\n' "$src"
}

download_rclone() {
  local arch tools_dir zip dir bin
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
    *)
      echo "[err] unsupported architecture for automatic rclone install: $arch" >&2
      echo "      Install rclone manually or set RCLONE_BIN=/path/to/rclone." >&2
      exit 2
      ;;
  esac

  tools_dir="$ROOT_DIR/.tools"
  zip="$tools_dir/rclone-${RCLONE_VERSION}-linux-${arch}.zip"
  dir="$tools_dir/rclone-${RCLONE_VERSION}-linux-${arch}"
  bin="$dir/rclone"

  mkdir -p "$tools_dir"
  if [ ! -x "$bin" ]; then
    command -v curl >/dev/null 2>&1 || { echo "[err] curl is required to download rclone" >&2; exit 2; }
    command -v unzip >/dev/null 2>&1 || { echo "[err] unzip is required to unpack rclone" >&2; exit 2; }

    log "downloading rclone ${RCLONE_VERSION} for linux-${arch}"
    curl -L "https://downloads.rclone.org/${RCLONE_VERSION}/rclone-${RCLONE_VERSION}-linux-${arch}.zip" -o "$zip"
    unzip -o "$zip" -d "$tools_dir" >/dev/null
  fi

  RCLONE_BIN="$bin"
}

select_rclone() {
  if [ -n "$RCLONE_BIN" ]; then
    [ -x "$RCLONE_BIN" ] || { echo "[err] RCLONE_BIN is not executable: $RCLONE_BIN" >&2; exit 2; }
    return
  fi

  if [ -x "$ROOT_DIR/.tools/rclone-${RCLONE_VERSION}-linux-amd64/rclone" ]; then
    RCLONE_BIN="$ROOT_DIR/.tools/rclone-${RCLONE_VERSION}-linux-amd64/rclone"
    return
  fi

  if command -v rclone >/dev/null 2>&1; then
    local version
    version="$(rclone version | sed -n '1s/^rclone //p')"
    case "$version" in
      v1.[6-9]*|v[2-9].*) RCLONE_BIN="$(command -v rclone)"; return ;;
    esac
    log "system rclone is old ($version); using a repo-local current rclone"
  fi

  download_rclone
}

remote_exists() {
  # listremotes is the reliable existence check: `config show <name>` exits 0 even when the
  # config file is missing, which would wrongly skip creation.
  "$RCLONE_BIN" listremotes 2>/dev/null | grep -qx "${1}:"
}

auth_remote() {
  local remote="$1"
  if remote_exists "$remote"; then
    log "reconnecting existing remote: $remote (interactive OAuth)"
    "$RCLONE_BIN" config reconnect "${remote}:"
  else
    log "creating rclone Google Drive remote: $remote (interactive OAuth)"
    "$RCLONE_BIN" config create "$remote" drive scope=drive
  fi
}

resolve_paths() {
  local src_data_root dst_data_root rel_src_raw rel_dst_raw rel_src rel_dst_trimmed dst_base
  src_data_root="$1"
  rel_src_raw="$2"
  dst_data_root="${3%/}"
  rel_dst_raw="$4"

  # rel_dst (local) may be an absolute/explicit path (/, ./, ../) -> use it verbatim and ignore
  # dst_data_root; otherwise it is joined under dst_data_root as before.
  case "$rel_dst_raw" in
    /*|./*|../*) LOCAL_DST="${rel_dst_raw%/}" ;;
    *)
      rel_dst_trimmed="$(trim_slashes "$rel_dst_raw")"
      dst_base="$(basename "$dst_data_root")"
      # Guard against a redundant leading "<dst_base>/" (e.g. dst_data_root=".../checkpoints"
      # and rel_dst="checkpoints/foo"), which would otherwise double up as ".../checkpoints/checkpoints/foo".
      case "$rel_dst_trimmed" in
        "$dst_base"/*) rel_dst_trimmed="${rel_dst_trimmed#"$dst_base"/}" ;;
      esac
      LOCAL_DST="$dst_data_root/$rel_dst_trimmed"
      ;;
  esac

  # Cloud source ending in "/" (or empty) -> append the local basename (mirror of upload), so
  # local ".../phase1-x" + cloud "reproduce/" pulls "reproduce/phase1-x" from Drive.
  case "$rel_src_raw" in
    ""|*/) rel_src_raw="${rel_src_raw}$(basename "$LOCAL_DST")" ;;
  esac
  rel_src="$(trim_slashes "$rel_src_raw")"
  FOLDER_ID="${FOLDER_ID:-}"

  # Root-scoped sources (folder-ID URL / drive-folder: / bare ID / bare "remote:" with no
  # explicit path) all point at some Drive root, so namespace InternNav's files under
  # DRIVE_PREFIX within it. An explicit "remote:path" override already carries its own path
  # and bypasses this on purpose.
  local prefixed_rel_src
  prefixed_rel_src="$rel_src"
  if [ -n "$DRIVE_PREFIX" ]; then
    prefixed_rel_src="$(trim_slashes "$DRIVE_PREFIX")/$rel_src"
  fi

  case "$src_data_root" in
    http://drive.google.com/*|https://drive.google.com/*)
      if ! FOLDER_ID="$(extract_drive_folder_id "$src_data_root")"; then
        echo "[err] unsupported Google Drive URL: $src_data_root" >&2
        echo "      Expected: https://drive.google.com/drive/folders/<folder-id>" >&2
        exit 2
      fi
      RCLONE_SRC="${REMOTE}:$prefixed_rel_src"
      ;;
    drive-folder:*)
      FOLDER_ID="${src_data_root#drive-folder:}"
      RCLONE_SRC="${REMOTE}:$prefixed_rel_src"
      ;;
    *)
      if is_drive_folder_id "$src_data_root"; then
        FOLDER_ID="$src_data_root"
        RCLONE_SRC="${REMOTE}:$prefixed_rel_src"
      elif [[ "$src_data_root" == *: ]]; then
        RCLONE_SRC="${src_data_root}${prefixed_rel_src}"
      elif [[ "$src_data_root" == *:* ]]; then
        RCLONE_SRC="${src_data_root%/}/$rel_src"
      else
        echo "[err] unsupported src_data_root: $src_data_root" >&2
        echo "      Use a Drive folder URL, folder ID, or rclone root such as gdrive:" >&2
        exit 2
      fi
      ;;
  esac
}

remote_args() {
  if [ -n "${FOLDER_ID:-}" ]; then
    printf '%s\0%s\0' --drive-root-folder-id "$FOLDER_ID"
  fi
}

check_remote_readable() {
  local args=()
  while IFS= read -r -d '' arg; do args+=("$arg"); done < <(remote_args)

  # EXTRA_ARGS is intentionally word-split so callers can pass rclone flags.
  # shellcheck disable=SC2086
  if ! "$RCLONE_BIN" lsf "$RCLONE_SRC" "${args[@]}" ${EXTRA_ARGS:-} >/dev/null; then
    cat >&2 <<EOF_ERR
[err] rclone cannot list source: $RCLONE_SRC${FOLDER_ID:+ (folder id: $FOLDER_ID)}

You probably need one-time Google Drive authentication:
  $0 --auth $REMOTE

Use the current rclone selected by this script. Ubuntu's apt rclone 1.53 is too old for
modern Google OAuth and causes errors like: Invalid number of arguments.
EOF_ERR
    exit 3
  fi
}

list_remote_files() {
  local args=()
  while IFS= read -r -d '' arg; do args+=("$arg"); done < <(remote_args)

  log "listing source: $RCLONE_SRC${FOLDER_ID:+ (folder id: $FOLDER_ID)}"
  # shellcheck disable=SC2086
  "$RCLONE_BIN" lsf "$RCLONE_SRC" -R "${args[@]}" ${EXTRA_ARGS:-}
}

copy_files() {
  local args=()
  while IFS= read -r -d '' arg; do args+=("$arg"); done < <(remote_args)

  mkdir -p "$LOCAL_DST"
  log "copying source: $RCLONE_SRC${FOLDER_ID:+ (folder id: $FOLDER_ID)}"
  log "destination: $LOCAL_DST"

  # EXTRA_ARGS is intentionally word-split so callers can pass rclone flags.
  # shellcheck disable=SC2086
  "$RCLONE_BIN" copy "$RCLONE_SRC" "$LOCAL_DST" \
    "${args[@]}" \
    --ignore-existing \
    --transfers "$TRANSFERS" \
    --checkers "$CHECKERS" \
    --retries 10 \
    --low-level-retries 20 \
    --drive-chunk-size 64M \
    --progress \
    ${EXTRA_ARGS:-}
}

verify_files() {
  local args=()
  while IFS= read -r -d '' arg; do args+=("$arg"); done < <(remote_args)

  log "verifying source against local destination"
  # shellcheck disable=SC2086
  "$RCLONE_BIN" check "$RCLONE_SRC" "$LOCAL_DST" \
    "${args[@]}" \
    --one-way \
    ${EXTRA_ARGS:-}
}

main() {
  local mode="copy"

  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --auth)
        select_rclone
        log "using rclone: $RCLONE_BIN"
        auth_remote "${2:-$REMOTE}"
        exit 0
        ;;
      --check) mode="check"; shift ;;
      --dry-run) mode="dry-run"; shift ;;
      --list) mode="list"; shift ;;
      --remote)
        [ $# -ge 2 ] || { echo "[err] --remote needs a value" >&2; exit 2; }
        REMOTE="$2"; shift 2 ;;
      --src-data-root)
        [ $# -ge 2 ] || { echo "[err] --src-data-root needs a value" >&2; exit 2; }
        SRC_DATA_ROOT="$2"; shift 2 ;;
      --dst-data-root)
        [ $# -ge 2 ] || { echo "[err] --dst-data-root needs a value" >&2; exit 2; }
        DST_DATA_ROOT="$2"; shift 2 ;;
      --)
        shift; break ;;
      -*)
        echo "[err] unknown option: $1" >&2
        usage
        exit 2
        ;;
      *) break ;;
    esac
  done

  case $# in
    2)
      # <local_path> <cloud_rel>, order-independent: the absolute/explicit arg is the local side.
      # Legacy both-relative form stays <cloud_rel> <local_rel>.
      case "$1" in
        /*|./*|../*)
          case "$2" in
            /*|./*|../*) set -- "$SRC_DATA_ROOT" "$1" "$DST_DATA_ROOT" "$2" ;;  # both abs: <src> <dst>
            *)           set -- "$SRC_DATA_ROOT" "$2" "$DST_DATA_ROOT" "$1" ;;  # <local_abs> <cloud_rel>
          esac ;;
        *) set -- "$SRC_DATA_ROOT" "$1" "$DST_DATA_ROOT" "$2" ;;                # <cloud_rel> <local_rel>
      esac
      ;;
    4)
      ;;
    *)
      echo "[err] need <rel_src> <rel_dst> or <src_data_root> <rel_src> <dst_data_root> <rel_dst>" >&2
      echo
      usage
      exit 2
      ;;
  esac

  resolve_paths "$1" "$2" "$3" "$4"

  if [ "$mode" = "dry-run" ]; then
    log "rclone source : $RCLONE_SRC${FOLDER_ID:+ (folder id: $FOLDER_ID)}"
    log "local dest    : $LOCAL_DST"
    exit 0
  fi

  select_rclone
  log "using rclone: $RCLONE_BIN"
  check_remote_readable

  if [ "$mode" = "list" ]; then
    list_remote_files
  elif [ "$mode" = "check" ]; then
    verify_files
  else
    copy_files
    verify_files
    log "done"
  fi
}

main "$@"
