#!/usr/bin/env bash

# Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
#
# This file is part of the Einstein Engine (EinsteinEngine).
#
# EinsteinEngine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EinsteinEngine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  update-copyright-authors.sh [FILE ...]

Behavior:
  - If files are provided, updates only those files.
  - If no files are provided, updates all tracked files from `git ls-files`.
  - Rewrites the first "Copyright (C)" line using:
      * Year/year-range = file creation year through current year.
      * Author list ordered by descending line responsibility from `git blame`.
      * Required suffix: "and other Einstein Engine contributors."
  - Supports hash-comment headers (lines starting with `#`).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

current_year="$(date +%Y)"
updated_count=0
skipped_count=0

gather_files() {
    if [[ "$#" -gt 0 ]]; then
        printf '%s\n' "$@"
    else
        git ls-files
    fi
}

is_tracked() {
    git ls-files --error-unmatch "$1" >/dev/null 2>&1
}

creation_year_for() {
    local file="$1"
    local y
    y="$(git log --follow --diff-filter=A --format=%ad --date=format:%Y -- "$file" | tail -n1)"
    if [[ "$y" =~ ^[0-9]{4}$ ]]; then
        printf '%s\n' "$y"
    else
        printf '%s\n' "$current_year"
    fi
}

years_for() {
    local start_year="$1"
    if [[ "$start_year" =~ ^[0-9]{4}$ ]] && (( start_year < current_year )); then
        printf '%s-%s\n' "$start_year" "$current_year"
    else
        printf '%s\n' "$current_year"
    fi
}

author_suffix_for() {
    local -n author_arr_ref="$1"
    local n="${#author_arr_ref[@]}"
    local joined=""

    if (( n == 0 )); then
        printf '%s\n' ""
        return 0
    fi

    if (( n == 1 )); then
        printf '%s\n' "${author_arr_ref[0]} and other Einstein Engine contributors."
        return 0
    fi

    for author in "${author_arr_ref[@]}"; do
        if [[ -z "$joined" ]]; then
            joined="$author"
        else
            joined="$joined, $author"
        fi
    done

    printf '%s\n' "$joined, and other Einstein Engine contributors."
}

update_one_file() {
    local file="$1"
    local line_no old_line prefix start_year years new_line tmp
    local -a authors

    [[ -f "$file" ]] || return 0
    grep -Iq . "$file" || return 0
    is_tracked "$file" || return 0

    line_no="$(grep -n -m1 'Copyright (C)' "$file" | cut -d: -f1 || true)"
    [[ -n "$line_no" ]] || return 0

    old_line="$(sed -n "${line_no}p" "$file")"
    if [[ ! "$old_line" =~ ^([[:space:]]*#[[:space:]]*)Copyright[[:space:]]\(C\)[[:space:]] ]]; then
        echo "skipped (non-hash header): $file"
        skipped_count=$((skipped_count + 1))
        return 0
    fi
    prefix="${BASH_REMATCH[1]}"

    mapfile -t authors < <(
        git blame --line-porcelain -- "$file" 2>/dev/null \
            | awk '
                /^author / {
                    name = substr($0, 8)
                    if (name != "" && name != "Not Committed Yet") count[name]++
                }
                END {
                    for (name in count) printf "%d\t%s\n", count[name], name
                }
            ' \
            | sort -t $'\t' -k1,1nr -k2,2 \
            | awk -F '\t' '{print $2}'
    )

    if (( ${#authors[@]} == 0 )); then
        echo "skipped (no blame authors): $file"
        skipped_count=$((skipped_count + 1))
        return 0
    fi

    start_year="$(creation_year_for "$file")"
    years="$(years_for "$start_year")"
    new_line="${prefix}Copyright (C) ${years} $(author_suffix_for authors)"

    if [[ "$new_line" != "$old_line" ]]; then
        tmp="$(mktemp)"
        awk -v n="$line_no" -v repl="$new_line" 'NR == n { $0 = repl } { print }' "$file" > "$tmp"
        mv "$tmp" "$file"
        echo "updated: $file"
        updated_count=$((updated_count + 1))
    fi
}

while IFS= read -r file; do
    update_one_file "$file"
done < <(gather_files "$@")

echo "done: updated=$updated_count skipped=$skipped_count"
