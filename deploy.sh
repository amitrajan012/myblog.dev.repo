#!/bin/sh
# Build the site and publish it to GitHub Pages.
#
#   ./deploy.sh "commit message"   build, commit and push both repos
#   ./deploy.sh -n                 dry run: build and list what would change, push nothing
#
# Two repos are involved:
#   this repo (myblog.dev.repo)        the Hugo source
#   ../amitrajan012.github.io           the built site that GitHub Pages serves
# Override the second with:  PAGES_DIR=/path/to/amitrajan012.github.io ./deploy.sh
set -e

SRC="$(cd "$(dirname "$0")" && pwd)"
PAGES="${PAGES_DIR:-$SRC/../amitrajan012.github.io}"
DRY=0
if [ "$1" = "-n" ] || [ "$1" = "--dry-run" ]; then DRY=1; shift; fi
MSG="${*:-rebuilding site $(date)}"

say() { printf "\033[0;32m%s\033[0m\n" "$*"; }
die() { printf "\033[0;31mError: %s\033[0m\n" "$*" >&2; exit 1; }

command -v hugo >/dev/null 2>&1 || die "hugo is not installed (brew install hugo)"
command -v rsync >/dev/null 2>&1 || die "rsync is not installed"
[ -d "$PAGES/.git" ] || die "GitHub Pages repo not found at $PAGES"
PAGES="$(cd "$PAGES" && pwd)"

# 1. Make sure the Pages repo is current, so the push can't be rejected.
say "Updating $PAGES"
git -C "$PAGES" pull --ff-only --quiet

# 2. Build into a temporary folder. Drafts are NOT published (no -D).
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
say "Building site"
(cd "$SRC" && hugo --minify --cleanDestinationDir --destination "$OUT")

# 3. Sanity checks before touching the live site.
[ -f "$OUT/index.html" ] || die "build has no index.html"
[ -f "$OUT/index.xml" ]  || die "build has no RSS feed"
POSTS=$(find "$OUT/post" -name index.html | wc -l | tr -d ' ')
[ "$POSTS" -gt 100 ] || die "only $POSTS post pages built — refusing to publish"
say "Built $POSTS post pages"

# 4. Copy the build into the Pages repo (keeps its .git, .gitignore and any CNAME).
RSYNC_OPTS="-a --delete --exclude .git --exclude .gitignore --exclude .DS_Store --exclude CNAME"
if [ "$DRY" = 1 ]; then
  say "Dry run — files that would change in $PAGES:"
  # shellcheck disable=SC2086
  rsync $RSYNC_OPTS --dry-run --itemize-changes --checksum "$OUT/" "$PAGES/" | grep -v '^\.' || echo "(none)"
  exit 0
fi
# shellcheck disable=SC2086
rsync $RSYNC_OPTS "$OUT/" "$PAGES/"

# 5. Commit and push the source repo, then the built site.
commit_push() {
  dir="$1"
  git -C "$dir" add -A
  if git -C "$dir" diff --cached --quiet; then
    say "$(basename "$dir"): no uncommitted changes (already committed)"
  else
    git -C "$dir" commit --quiet -m "$MSG"
    say "$(basename "$dir"): committed \"$MSG\""
  fi
  branch="$(git -C "$dir" rev-parse --abbrev-ref HEAD)"
  git -C "$dir" fetch --quiet origin "$branch" 2>/dev/null || true
  ahead="$(git -C "$dir" rev-list --count "origin/$branch..HEAD" 2>/dev/null || echo "?")"
  if [ "$ahead" = "0" ]; then
    say "$(basename "$dir"): nothing to push — already up to date"
  else
    say "$(basename "$dir"): pushing $ahead commit(s):"
    git -C "$dir" log --oneline "origin/$branch..HEAD" 2>/dev/null | sed 's/^/    /'
    git -C "$dir" push --quiet origin "$branch"
  fi
}
commit_push "$SRC"
commit_push "$PAGES"
say "Done. GitHub Pages usually updates within a minute or two."
