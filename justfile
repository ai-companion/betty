# Tag a specific release version as "latest"
# Usage: just tag-latest v0.8.0
tag-latest release-ver:
    git tag -d latest 2>/dev/null || true
    git push origin :refs/tags/latest 2>/dev/null || true
    git tag latest {{release-ver}}
    git push origin latest
