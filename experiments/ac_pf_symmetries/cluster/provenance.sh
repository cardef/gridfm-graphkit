#!/bin/bash
# Print the sha1 of each file a job runs (the log binds it) and keep a content-addressed copy, so a run
# made from an uncommitted working tree can still be traced to the exact code.
set -euo pipefail
P=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/SymmetricalFM/provenance
mkdir -p "$P"
for f in "$@"; do
    h=$(sha1sum "$f" | cut -c1-12)
    dst="$P/$(basename "$f").$h"
    # concurrent array tasks race on the same name; the content is identical by construction
    [ -f "$dst" ] || cp "$f" "$dst" 2>/dev/null || [ -f "$dst" ]
    echo "CODE $f $h"
done
