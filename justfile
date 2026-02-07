# Tag a specific release version as "latest"
# Usage: just tag-latest v0.8.0
tag-latest release-ver:
    git fetch --tags
    git rev-parse --verify {{release-ver}} >/dev/null 2>&1 || ( \
        echo "Error: release ref '{{release-ver}}' not found after fetching tags." >&2; \
        echo "Recent tags:" >&2; \
        git tag --sort=-creatordate | head -n 10 >&2; \
        exit 1 \
    )
    git tag -d latest 2>/dev/null || true
    git push origin :refs/tags/latest 2>/dev/null || true
    git tag latest {{release-ver}}
    git push origin latest
