#!/bin/bash

set -e

# ── get running job ids ───────────────────────────────────────────────────
# %i gives array-aware IDs (e.g. "12345_3" for array task 3 of job 12345),
# which matches how array logs are named (job_%A_%a.out -> job_12345_3.out).
# Plain %A would just give "12345" for every task and wouldn't match those
# per-task filenames.
RUNNING_IDS=$(squeue -u hcr64 -t RUNNING -h -o "%i")

# ── log locations to check ─────────────────────────────────────────────────
LOG_DIRS=(
    # "/scratch/hcr64"
    "clustering/logs"
    "modelling/logs"
)

# ── find + delete logs whose embedded job id is NOT currently running ──────
DRY_RUN=false   # flip to false once you've checked the output looks right

for dir in "${LOG_DIRS[@]}"; do
    [ -d "$dir" ] || continue

    for f in "$dir"/*.txt "$dir"/*.err "$dir"/*.out; do
        [ -e "$f" ] || continue   # skip if glob matched nothing

        base=$(basename "$f")

        # extract the job id portion from filenames like:
        #   12345.txt / 12345.err                (single run, scratch dir)
        #   job_12345_3.out / job_12345_3.err     (array task, clustering/logs)
        #   model_sweep_12345_3.txt/.err          (array task, modelling/logs)
        job_id=$(echo "$base" | grep -oE '[0-9]+(_[0-9]+)?' | head -n1)

        if [ -z "$job_id" ]; then
            continue
        fi

        # if job_id is NOT in the running list, it's a finished job's log
        if ! grep -qxF "$job_id" <<< "$RUNNING_IDS"; then
            if $DRY_RUN; then
                echo "Would delete: $f  (job_id=$job_id)"
            else
                # echo "Deleting: $f  (job_id=$job_id)"
                rm -f "$f"
            fi
        fi
    done
done