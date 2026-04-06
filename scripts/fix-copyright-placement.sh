#!/usr/bin/env bash

# Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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
  fix-copyright-placement.sh [FILE ...]

Behavior:
  - Finds the first occurrence of "Copyright (C)" in each file.
  - If the copyright block is not at the top (allowing optional shebang + blank lines),
    moves the whole comment block to the top.
  - Handles malformed inline insertions like:
      import x as # Copyright (C) ...
    by restoring the code line and moving the comment block.

If no files are provided, all tracked files from `git ls-files` are scanned.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

gather_files() {
    if [[ "$#" -gt 0 ]]; then
        printf '%s\n' "$@"
    else
        git ls-files
    fi
}

fix_one_file() {
    local file="$1"
    [[ -f "$file" ]] || return 0
    grep -Iq . "$file" || return 0
    grep -q "Copyright (C)" "$file" || return 0

    local tmp
    tmp="$(mktemp)"

    awk '
    function trim_header(    k) {
        # Drop trailing empty entries and trailing separator-only comment lines.
        while (hcount > 0 && header[hcount] ~ /^[[:space:]]*$/) hcount--
        while (hcount > 0 && header[hcount] ~ /^#[[:space:]]*$/) hcount--
    }

    {
        line[NR] = $0
        n = NR
        if (cp == 0 && index($0, "Copyright (C)") > 0) {
            cp = NR
        }
    }
    END {
        if (n == 0 || cp == 0) {
            for (i = 1; i <= n; i++) print line[i]
            exit
        }

        shebang = (line[1] ~ /^#!/)

        first = shebang ? 2 : 1
        while (first <= n && line[first] ~ /^[[:space:]]*$/) first++

        cp_is_comment = (line[cp] ~ /^[[:space:]]*#.*Copyright \(C\)/)
        already_top = cp_is_comment && (cp == first)
        if (already_top) {
            for (i = 1; i <= n; i++) print line[i]
            exit
        }

        inline = !cp_is_comment

        hcount = 0
        remove_start = cp
        remove_end = cp - 1

        if (inline) {
            orig = line[cp]
            if (match(orig, /#[[:space:]]*Copyright \(C\).*/)) {
                hdr = substr(orig, RSTART)
                sub(/^[[:space:]]+/, "", hdr)
                header[++hcount] = hdr
            } else {
                for (i = 1; i <= n; i++) print line[i]
                exit
            }

            fixed = orig
            sub(/[[:space:]]+#.*Copyright \(C\).*/, "", fixed)
            line[cp] = fixed

            i = cp + 1
            if (i <= n && line[i] ~ /^[[:space:]]*#/) {
                remove_start = i
                while (i <= n && line[i] ~ /^[[:space:]]*#/) {
                    txt = line[i]
                    sub(/^[[:space:]]+/, "", txt)
                    header[++hcount] = txt
                    i++
                }
                remove_end = i - 1
            }

            # Special case: recover "import ... as <alias>" when alias ended up on
            # the next line after a malformed inline copyright injection.
            next_code = remove_end + 1
            if (next_code <= n &&
                line[cp] ~ /[[:space:]]as[[:space:]]*$/ &&
                line[next_code] ~ /^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*$/) {
                alias = line[next_code]
                sub(/^[[:space:]]+/, "", alias)
                sub(/[[:space:]]+$/, "", alias)
                line[cp] = line[cp] " " alias
                remove_start = (remove_start <= next_code ? remove_start : next_code)
                remove_end = next_code
            }
        } else {
            i = cp
            remove_start = cp
            while (i <= n && line[i] ~ /^[[:space:]]*#/) {
                txt = line[i]
                sub(/^[[:space:]]+/, "", txt)
                header[++hcount] = txt
                i++
            }
            remove_end = i - 1
        }

        trim_header()

        # Emit file with shebang preserved at line 1 (if present), then header.
        if (shebang) {
            print line[1]
            print ""
        }

        for (i = 1; i <= hcount; i++) print header[i]

        # Ensure one blank line after inserted header before code.
        print ""

        start_body = shebang ? 2 : 1
        for (i = start_body; i <= n; i++) {
            if (i >= remove_start && i <= remove_end) continue
            print line[i]
        }
    }' "$file" > "$tmp"

    if ! cmp -s "$file" "$tmp"; then
        mv "$tmp" "$file"
        echo "fixed: $file"
    else
        rm -f "$tmp"
    fi
}

while IFS= read -r f; do
    fix_one_file "$f"
done < <(gather_files "$@")
