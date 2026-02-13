#!/usr/bin/env bash
# install.sh — Install betty to ~/.local/bin/
#
# Usage:
#   curl -fsSL https://betty4.sh/install.sh | bash
#   curl -fsSL ... | bash -s -- --github          # install from GitHub
#   curl -fsSL ... | bash -s -- --version 0.10.1   # pin a version
#   curl -fsSL ... | bash -s -- --upgrade          # upgrade existing install
#   curl -fsSL ... | bash -s -- --uninstall        # remove
#
# Environment variable overrides (alternative to flags):
#   BETTY_VERSION=0.10.1   — pin a specific version
#   BETTY_FROM_GITHUB=1    — install from GitHub instead of PyPI

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────
PACKAGE_NAME="betty"
GITHUB_REPO="ai-companion/betty"
BIN_DIR="${HOME}/.local/bin"

# ── Parse flags ───────────────────────────────────────────────────────
FROM_GITHUB="${BETTY_FROM_GITHUB:-0}"
VERSION="${BETTY_VERSION:-}"
ACTION="install"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --github)    FROM_GITHUB=1; shift ;;
        --version)   VERSION="$2"; shift 2 ;;
        --upgrade)   ACTION="upgrade"; shift ;;
        --uninstall) ACTION="uninstall"; shift ;;
        *)           echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────
info()  { printf '\033[1;34m::\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m::\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m::\033[0m %s\n' "$*"; }
err()   { printf '\033[1;31m::\033[0m %s\n' "$*" >&2; }

has() { command -v "$1" >/dev/null 2>&1; }

github_source_https() {
    local ref="${VERSION:-latest}"
    echo "${PACKAGE_NAME} @ git+https://github.com/${GITHUB_REPO}@${ref}"
}

github_source_ssh() {
    local ref="${VERSION:-latest}"
    echo "${PACKAGE_NAME} @ git+ssh://git@github.com/${GITHUB_REPO}@${ref}"
}

pypi_source() {
    if [[ -n "$VERSION" ]]; then
        echo "${PACKAGE_NAME}==${VERSION}"
    else
        echo "${PACKAGE_NAME}"
    fi
}

# ── Uninstall ─────────────────────────────────────────────────────────
do_uninstall() {
    info "Uninstalling ${PACKAGE_NAME}..."
    if has uv && uv tool list 2>/dev/null | grep -q "^${PACKAGE_NAME} "; then
        uv tool uninstall "$PACKAGE_NAME"
        ok "Uninstalled via uv."
    elif has pipx && pipx list 2>/dev/null | grep -q "${PACKAGE_NAME}"; then
        pipx uninstall "$PACKAGE_NAME"
        ok "Uninstalled via pipx."
    else
        err "Could not find ${PACKAGE_NAME} installed via uv or pipx."
        exit 1
    fi
}

# ── Install / Upgrade ────────────────────────────────────────────────

# Try to install a source with a given tool, with HTTPS→SSH fallback for GitHub.
# Usage: try_install <tool> <upgrade_flag>
try_install() {
    local tool="$1"
    local upgrade_flag="$2"

    # uv uses "uv tool install", pipx uses "pipx install"
    local install_cmd
    if [[ "$tool" == "uv" ]]; then
        install_cmd="uv tool install"
    else
        install_cmd="pipx install"
    fi

    if [[ "$FROM_GITHUB" == "1" ]]; then
        local https_source ssh_source
        https_source="$(github_source_https)"
        ssh_source="$(github_source_ssh)"

        info "Trying HTTPS: ${https_source}"
        # shellcheck disable=SC2086
        if $install_cmd $upgrade_flag "$https_source" 2>/dev/null; then
            return 0
        fi

        info "HTTPS failed, trying SSH: ${ssh_source}"
        # shellcheck disable=SC2086
        $install_cmd $upgrade_flag "$ssh_source"
    else
        local source
        source="$(pypi_source)"
        # shellcheck disable=SC2086
        $install_cmd $upgrade_flag "$source"
    fi
}

do_install() {
    local upgrade_flag=""
    if [[ "$ACTION" == "upgrade" ]]; then
        # --reinstall ensures a fresh fetch (important for git sources)
        upgrade_flag="--reinstall"
        info "Upgrading ${PACKAGE_NAME}..."
    else
        info "Installing ${PACKAGE_NAME}..."
    fi

    # Try uv first
    if has uv; then
        info "Using uv..."
        try_install uv "$upgrade_flag"
        ok "Installed via uv."
        return
    fi

    # Try pipx
    if has pipx; then
        info "Using pipx..."
        if [[ "$ACTION" == "upgrade" ]]; then
            pipx reinstall "$PACKAGE_NAME"
        else
            try_install pipx "$upgrade_flag"
        fi
        ok "Installed via pipx."
        return
    fi

    # Bootstrap uv, then install
    info "Neither uv nor pipx found. Bootstrapping uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"

    if ! has uv; then
        err "Failed to bootstrap uv. Please install uv or pipx manually."
        exit 1
    fi

    info "Using uv..."
    try_install uv "$upgrade_flag"
    ok "Installed via uv."
}

# ── PATH check ────────────────────────────────────────────────────────
check_path() {
    case ":${PATH}:" in
        *":${BIN_DIR}:"*) return ;;
    esac

    warn "${BIN_DIR} is not in your PATH."

    local shell_name
    shell_name="$(basename "${SHELL:-/bin/bash}")"

    case "$shell_name" in
        fish)
            warn "Add it with:  fish_add_path ${BIN_DIR}"
            ;;
        zsh)
            warn "Add it with:  echo 'export PATH=\"${BIN_DIR}:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
            ;;
        bash)
            warn "Add it with:  echo 'export PATH=\"${BIN_DIR}:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
            ;;
        *)
            warn "Add ${BIN_DIR} to your shell's PATH."
            ;;
    esac
}

# ── Verify ────────────────────────────────────────────────────────────
verify() {
    if has betty; then
        ok "$(betty --version)"
    else
        check_path
        # Try with full path
        if [[ -x "${BIN_DIR}/betty" ]]; then
            ok "$("${BIN_DIR}/betty" --version)"
        else
            warn "Could not verify installation. You may need to restart your shell."
        fi
    fi
}

# ── Main ──────────────────────────────────────────────────────────────
main() {
    if [[ "$ACTION" == "uninstall" ]]; then
        do_uninstall
    else
        do_install
        verify
    fi
}

main
