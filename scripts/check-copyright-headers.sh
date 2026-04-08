#!/usr/bin/env bash

# Copyright (C) 2024-2026 Max Morris and other Einstein Engine contributors.
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
    cat <<'USAGE'
Usage:
  check-copyright-headers.sh [FILE.py ...]

Checks Python files for a strict top-of-file copyright header format.
If no files are provided, all tracked *.py files are checked.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

gather_files() {
    if [[ "$#" -gt 0 ]]; then
        printf '%s\n' "$@"
    else
        git ls-files '*.py'
    fi
}

check_one_file() {
    local file="$1"
    [[ -f "$file" ]] || return 0

    awk '
    function fail(msg) {
        print FILENAME ": " msg
        bad = 1
        exit 1
    }

    function strip_hash_prefix(s,    t) {
        t = s
        sub(/^[[:space:]]*#[[:space:]]*/, "", t)
        return t
    }

    {
        line[NR] = $0
        n = NR
        if (cp == 0 && index($0, "Copyright (C)") > 0) cp = NR
    }

    END {
        if (n == 0) exit 0

        shebang = (line[1] ~ /^#!/)
        first = shebang ? 2 : 1
        while (first <= n && line[first] ~ /^[[:space:]]*$/) first++

        if (cp == 0) fail("missing copyright header")
        if (cp != first) fail("copyright header is not at top of file")
        if (line[cp] !~ /^[[:space:]]*#.*Copyright \(C\)/) fail("copyright line is not a comment header")

        hcount = 0
        i = cp
        while (i <= n && line[i] ~ /^[[:space:]]*#/) {
            header[++hcount] = strip_hash_prefix(line[i])
            i++
        }

        expected_count = 16
        expected[1]  = "__COPYRIGHT_LINE__"
        expected[2]  = ""
        expected[3]  = "This file is part of the Einstein Engine (EinsteinEngine)."
        expected[4]  = ""
        expected[5]  = "EinsteinEngine is free software: you can redistribute it and/or modify"
        expected[6]  = "it under the terms of the GNU Affero General Public License as published by"
        expected[7]  = "the Free Software Foundation, either version 3 of the License, or"
        expected[8]  = "(at your option) any later version."
        expected[9]  = ""
        expected[10] = "EinsteinEngine is distributed in the hope that it will be useful,"
        expected[11] = "but WITHOUT ANY WARRANTY; without even the implied warranty of"
        expected[12] = "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the"
        expected[13] = "GNU Affero General Public License for more details."
        expected[14] = ""
        expected[15] = "You should have received a copy of the GNU Affero General Public License"
        expected[16] = "along with this program.  If not, see <https://www.gnu.org/licenses/>."

        if (hcount != expected_count) {
            fail("header format mismatch: expected " expected_count " header lines, found " hcount)
        }

        if (header[1] !~ /^Copyright \(C\) [0-9][0-9 ,.-]*[[:space:]]+.+,?[[:space:]]+and other Einstein Engine contributors\.$/) {
            fail("copyright line must match canonical format and end with \"and other Einstein Engine contributors.\" (comma optional)")
        }

        for (i = 2; i <= expected_count; i++) {
            if (header[i] != expected[i]) {
                fail("header line " i " does not match required text")
            }
        }
    }
    ' "$file"
}

errors=()
while IFS= read -r file; do
    [[ "$file" == *.py ]] || continue
    if ! out="$(check_one_file "$file" 2>&1)"; then
        errors+=("$out")
    fi
done < <(gather_files "$@")

if [[ "${#errors[@]}" -gt 0 ]]; then
    printf '%s\n' "${errors[@]}"
    exit 1
fi
