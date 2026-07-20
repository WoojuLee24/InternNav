#!/usr/bin/env bash
# Upload a local checkpoint subdirectory to a Google Drive/rclone data root.
# Adapted from VLN-Challenge/upload_checkpoints_rclone.sh for InternNav's own Drive account.
# Mirror image of download_checkpoints_rclone.sh: same rclone bootstrap/auth/remote/
# DRIVE_PREFIX handling, only the copy direction is reversed (local -> Drive). Intended to
# run on the pod (or wherever SRC_DATA_ROOT lives), e.g. via `kubectl exec -it <pod> -- bash`.
#
# Convenient 2-arg form: give a LOCAL path (absolute — the one you know) and a CLOUD path
# (relative to the fixed Drive folder). Order does not matter: whichever arg is an absolute/
# explicit path (/ ./ ../) is the local side, the other is the cloud side. If the cloud path
# ends in "/" (or is empty), the local basename is appended so the folder name is preserved.
#
# Usage:
#   ./upload_checkpoints_rclone.sh <local_path> <cloud_rel>
#   ./upload_checkpoints_rclone.sh <src_data_root> <rel_src|abs_src> <dst_data_root> <rel_dst>
#   ./upload_checkpoints_rclone.sh --check <local_path> <cloud_rel>
#   ./upload_checkpoints_rclone.sh --auth [remote]
#
# Defaults:
#   SRC_DATA_ROOT=/home/irteam/data-vol2/checkpoints   (local/pod; = $CKPT_OUT)
#   DST_DATA_ROOT=gdrive:  (My Drive root, no folder ID needed)
#   DRIVE_PREFIX=InternNav/checkpoints  (subfolder under My Drive root; prepended to rel_dst)
#
# Arguments:
#   local_path     Local dir to upload. Absolute/explicit (/ ./ ../) -> used verbatim; otherwise
#                  a plain name is joined under src_data_root.
#   cloud_rel      Path under DRIVE_PREFIX to write. Ending in "/" -> append local basename.
#
# Upload an image_base checkpoint into a Drive "image_base/" subfolder:
#   ./upload_checkpoints_rclone.sh \
#     /home/irteam/data-vol2/checkpoints/image_base/base_s1.fpv_s2.fpv_20260709_130547 \
#     image_base/
#   # -> Drive: My Drive/InternNav/checkpoints/image_base/base_s1.fpv_s2.fpv_20260709_130547
#
# First-time auth, if needed:
#   ./upload_checkpoints_rclone.sh --auth gdrive
#   On a headless server choose N, run the printed authorize command on a browser machine,
#   then paste the token back into this machine.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCLONE_VERSION="${RCLONE_VERSION:-v1.74.3}"
RCLONE_BIN="${RCLONE_BIN:-}"
REMOTE="${RCLONE_REMOTE:-gdrive}"
SRC_DATA_ROOT="${SRC_DATA_ROOT:-${CKPT_OUT:-/home/irteam/data-vol2/checkpoints}}"
DST_DATA_ROOT="${DST_DATA_ROOT:-${REMOTE}:}"
DRIVE_PREFIX="${DRIVE_PREFIX:-InternNav/checkpoints}"
TRANSFERS="${TRANSFERS:-2}"
CHECKERS="${CHECKERS:-8}"
FOLDER_ID="${FOLDER_ID:-}"
LOCAL_SRC=""
RCLONE_DST=""

usage() {
  sed -n '2,38p' "$0"
  cat <<EOF_USAGE

Options:
  --auth [remote]       Create/reconnect an rclone Google Drive remote. Default: $REMOTE
  --check               Verify local source against remote destination only.
  --remote <name>       Remote to use for Drive URL/folder-id roots. Default: $REMOTE
  --src-data-root <src> Override source data root (local). Default: $SRC_DATA_ROOT
  --dst-data-root <dst> Override destination data root (Drive). Default: $DST_DATA_ROOT
  -h, --help            Show this help.

Environment:
  SRC_DATA_ROOT    Default source root (local). Default: $SRC_DATA_ROOT
  DST_DATA_ROOT    Default destination root (Drive). Default: $DST_DATA_ROOT
  DRIVE_PREFIX     Subfolder prepended to rel_dst for root-scoped destinations (bare "remote:",
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
  local ns
  ns="$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null || echo "<namespace>")"
  local pod
  pod="$(hostname 2>/dev/null || echo "<pod-name>")"
  cat >&2 <<EOF_AUTH_HINT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STEP 1] 로컬 머신의 새 터미널에서 포트 포워딩 실행:
  kubectl port-forward ${pod} 53682:53682 -n ${ns}

[STEP 2] 아래 URL이 출력되면 로컬 브라우저에서 접속:
  http://127.0.0.1:53682/auth?state=...

[STEP 3] Google 로그인 후 rclone 권한 허용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF_AUTH_HINT
  if remote_exists "$remote"; then
    log "reconnecting existing remote: $remote (interactive OAuth)"
    "$RCLONE_BIN" config reconnect "${remote}:"
  else
    log "creating rclone Google Drive remote: $remote (interactive OAuth)"
    "$RCLONE_BIN" config create "$remote" drive scope=drive
  fi
}

resolve_paths() {
  local src_data_root dst_data_root rel_src_raw rel_dst_raw rel_dst
  src_data_root="${1%/}"
  rel_src_raw="$2"
  dst_data_root="$3"
  rel_dst_raw="$4"

  # rel_src (local) may be an absolute/explicit path (/, ./, ../) -> use it verbatim and ignore
  # src_data_root; otherwise it is joined under src_data_root as before.
  case "$rel_src_raw" in
    /*|./*|../*) LOCAL_SRC="${rel_src_raw%/}" ;;
    *)           LOCAL_SRC="$src_data_root/$(trim_slashes "$rel_src_raw")" ;;
  esac

  # Cloud dest ending in "/" (or empty) -> append the local basename, so the checkpoint folder
  # name is preserved under the given Drive subfolder (rclone copy would otherwise flatten the
  # contents directly into rel_dst). e.g. src ".../phase1-x" + "reproduce/" -> "reproduce/phase1-x".
  case "$rel_dst_raw" in
    ""|*/) rel_dst_raw="${rel_dst_raw}$(basename "$LOCAL_SRC")" ;;
  esac
  rel_dst="$(trim_slashes "$rel_dst_raw")"
  FOLDER_ID="${FOLDER_ID:-}"

  # Root-scoped destinations (folder-ID URL / drive-folder: / bare ID / bare "remote:" with no
  # explicit path) all point at some Drive root, so namespace InternNav's files under
  # DRIVE_PREFIX within it. An explicit "remote:path" override already carries its own path
  # and bypasses this on purpose.
  local prefixed_rel_dst
  prefixed_rel_dst="$rel_dst"
  if [ -n "$DRIVE_PREFIX" ]; then
    prefixed_rel_dst="$(trim_slashes "$DRIVE_PREFIX")/$rel_dst"
  fi

  case "$dst_data_root" in
    http://drive.google.com/*|https://drive.google.com/*)
      if ! FOLDER_ID="$(extract_drive_folder_id "$dst_data_root")"; then
        echo "[err] unsupported Google Drive URL: $dst_data_root" >&2
        echo "      Expected: https://drive.google.com/drive/folders/<folder-id>" >&2
        exit 2
      fi
      RCLONE_DST="${REMOTE}:$prefixed_rel_dst"
      ;;
    drive-folder:*)
      FOLDER_ID="${dst_data_root#drive-folder:}"
      RCLONE_DST="${REMOTE}:$prefixed_rel_dst"
      ;;
    *)
      if is_drive_folder_id "$dst_data_root"; then
        FOLDER_ID="$dst_data_root"
        RCLONE_DST="${REMOTE}:$prefixed_rel_dst"
      elif [[ "$dst_data_root" == *: ]]; then
        RCLONE_DST="${dst_data_root}${prefixed_rel_dst}"
      elif [[ "$dst_data_root" == *:* ]]; then
        RCLONE_DST="${dst_data_root%/}/$rel_dst"
      else
        echo "[err] unsupported dst_data_root: $dst_data_root" >&2
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

check_source_and_remote() {
  local args=()
  while IFS= read -r -d '' arg; do args+=("$arg"); done < <(remote_args)

  if [ ! -d "$LOCAL_SRC" ]; then
    echo "[err] local source directory not found: $LOCAL_SRC" >&2
    exit 2
  fi

  # Listing the Drive root confirms auth + write target reachability before uploading.
  # EXTRA_ARGS is intentionally word-split so callers can pass rclone flags.
  # shellcheck disable=SC2086
  if ! "$RCLONE_BIN" lsf "${REMOTE}:" "${args[@]}" ${EXTRA_ARGS:-} >/dev/null; then
    cat >&2 <<EOF_ERR
[err] rclone cannot access Drive destination root${FOLDER_ID:+ (folder id: $FOLDER_ID)}

You probably need one-time Google Drive authentication:
  $0 --auth $REMOTE

Use the current rclone selected by this script. Ubuntu's apt rclone 1.53 is too old for
modern Google OAuth and causes errors like: Invalid number of arguments.
EOF_ERR
    exit 3
  fi
}

copy_files() {
  local args=()
  while IFS= read -r -d '' arg; do args+=("$arg"); done < <(remote_args)

  log "uploading source: $LOCAL_SRC"
  log "destination: $RCLONE_DST${FOLDER_ID:+ (folder id: $FOLDER_ID)}"

  # EXTRA_ARGS is intentionally word-split so callers can pass rclone flags.
  # No --ignore-existing: let rclone's size+hash comparison skip identical files (resumable + safe).
  # shellcheck disable=SC2086
  "$RCLONE_BIN" copy "$LOCAL_SRC" "$RCLONE_DST" \
    "${args[@]}" \
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

  log "verifying local source against remote destination"
  # --one-way: every local file must exist and match on the remote (extra remote files ignored).
  # shellcheck disable=SC2086
  "$RCLONE_BIN" check "$LOCAL_SRC" "$RCLONE_DST" \
    "${args[@]}" \
    --one-way \
    ${EXTRA_ARGS:-}
}

# Resolve ONLY the Drive destination root (FOLDER_ID) from DST_DATA_ROOT, without needing a
# local/relative path. Used by --ensure-auth and --list.
resolve_dst_root() {
  local dst_data_root="$1"
  case "$dst_data_root" in
    http://drive.google.com/*|https://drive.google.com/*)
      FOLDER_ID="$(extract_drive_folder_id "$dst_data_root")" || {
        echo "[err] unsupported Google Drive URL: $dst_data_root" >&2; exit 2; } ;;
    drive-folder:*) FOLDER_ID="${dst_data_root#drive-folder:}" ;;
    *)
      if is_drive_folder_id "$dst_data_root"; then FOLDER_ID="$dst_data_root"
      elif [[ "$dst_data_root" == *:* ]]; then FOLDER_ID=""   # plain rclone remote root
      else echo "[err] unsupported dst_data_root: $dst_data_root" >&2; exit 2; fi ;;
  esac
}

# Return 0 if the Drive dest root is reachable (auth valid), else 1.
drive_reachable() {
  local args=(); while IFS= read -r -d '' a; do args+=("$a"); done < <(remote_args)
  # shellcheck disable=SC2086
  "$RCLONE_BIN" lsf "${REMOTE}:" "${args[@]}" ${EXTRA_ARGS:-} >/dev/null 2>&1
}

# Verify Drive access; if unreachable, run interactive auth (headless: prints an authorize
# command to run in a browser, then waits for you to paste the token back), then re-verify.
ensure_auth() {
  resolve_dst_root "$DST_DATA_ROOT"
  if drive_reachable; then
    log "Drive access OK (remote: $REMOTE)"
    return 0
  fi
  log "Drive not reachable for remote '$REMOTE' -> starting interactive authentication"
  auth_remote "$REMOTE"
  drive_reachable || { echo "[err] still cannot access Drive after auth (remote: $REMOTE)" >&2; exit 3; }
  log "authentication OK"
}

# List folders under the Drive dest root (optional REL subpath to drill down; defaults to
# DRIVE_PREFIX so the InternNav checkpoints subfolder is what shows up without extra args).
list_dirs() {
  resolve_dst_root "$DST_DATA_ROOT"
  local rel; rel="$(trim_slashes "${1:-$DRIVE_PREFIX}")"
  local args=(); while IFS= read -r -d '' a; do args+=("$a"); done < <(remote_args)
  log "folders under Drive dest root${rel:+/$rel}:"
  # shellcheck disable=SC2086
  "$RCLONE_BIN" lsf "${REMOTE}:${rel}" "${args[@]}" --dirs-only ${EXTRA_ARGS:-}
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
      --ensure-auth)
        select_rclone
        log "using rclone: $RCLONE_BIN"
        ensure_auth
        exit 0
        ;;
      --list)
        select_rclone
        list_dirs "${2:-}"
        exit 0
        ;;
      --check) mode="check"; shift ;;
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
      case "$2" in
        /*|./*|../*)
          case "$1" in
            /*|./*|../*) set -- "$SRC_DATA_ROOT" "$1" "$DST_DATA_ROOT" "$2" ;;  # both abs: <src> <dst>
            *)           set -- "$SRC_DATA_ROOT" "$2" "$DST_DATA_ROOT" "$1" ;;  # <cloud_rel> <local_abs>
          esac ;;
        *) set -- "$SRC_DATA_ROOT" "$1" "$DST_DATA_ROOT" "$2" ;;                # <local_path> <cloud_rel>
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

  select_rclone
  log "using rclone: $RCLONE_BIN"
  # Auto-heal auth: if the Drive dest is unreachable, run interactive auth first (AUTO_AUTH=0 to skip).
  [ "${AUTO_AUTH:-1}" = "1" ] && ensure_auth
  resolve_paths "$1" "$2" "$3" "$4"
  check_source_and_remote

  if [ "$mode" = "check" ]; then
    verify_files
  else
    copy_files
    verify_files
    log "done"
  fi
}

main "$@"
