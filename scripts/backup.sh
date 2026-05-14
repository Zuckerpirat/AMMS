#!/usr/bin/env bash
# Snapshot the SQLite DB out of the running Docker container.
# Run from the repo root on the VPS:
#   ./scripts/backup.sh           writes ./backups/amms-<UTC>.sqlite
#   ./scripts/backup.sh /tmp      writes /tmp/amms-<UTC>.sqlite
set -euo pipefail

dest_dir="${1:-./backups}"
mkdir -p "$dest_dir"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$dest_dir/amms-$ts.sqlite"

container_id="$(docker compose ps -q amms 2>/dev/null)"
if [ -z "$container_id" ]; then
    echo "amms container is not running. Start it first: docker compose up -d"
    exit 1
fi

# .backup is the safe online-backup approach for SQLite.
docker compose exec -T amms sqlite3 /data/amms.sqlite ".backup '/data/amms-backup.sqlite'"
docker compose cp "amms:/data/amms-backup.sqlite" "$out"
docker compose exec -T amms rm -f /data/amms-backup.sqlite
echo "wrote $out"
