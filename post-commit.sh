#!/bin/sh
pwd
# Get the latest commit SHA
commit_sha=$(git rev-parse HEAD)
init_file="spm/__init__.py" #  git hooks run from root of repo
sed -i.bak "s/^GIT_SHA = .*/GIT_SHA = \"$commit_sha\"  # updated by post-commit/" "$init_file" || exit 1
echo "Updated $init_file with GIT_SHA = \"$commit_sha\""