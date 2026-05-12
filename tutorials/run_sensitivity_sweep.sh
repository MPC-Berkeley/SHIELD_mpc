#!/bin/bash
# Sensitivity sweep for SHIELD parameters (ε, ζ, λ)
# One-at-a-time (OAT) sweep: vary one param, fix others at nominal.
# Nominal: eps=0.01, tightening=2.3, l1_lmbd=100
# Each run uses ~20 scenarios (first 20 from the gurobi eval log list).
# Results saved to: nuplan/expert_data/sensitivity/sensitivity_{param}_{value}.pkl.gz

set -e

CONFIG=/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_config_eval.yaml
EVAL=/home/mpc/nuplan-devkit/tutorials/nuplan_closedloop_evaluate.py
OUTDIR=/home/mpc/nuplan-devkit/nuplan/expert_data/sensitivity
mkdir -p "$OUTDIR"

run_sim() {
    local tag="$1"
    cd /home/mpc/nuplan-devkit/tutorials
    NUPLAN_ROOT_DIR=/home/mpc/nuplan-devkit \
    PYTHONPATH=/home/mpc/nuplan-devkit \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    SENSITIVITY_TAG="$tag" \
    SENSITIVITY_OUTDIR="$OUTDIR" \
    conda run -n nuplan python nuplan_closedloop_evaluate.py 2>&1 | tee "/tmp/sensitivity_${tag}.log"
    echo "=== DONE: $tag ==="
}

set_config() {
    local eps="$1" tight="$2" l1="$3" tag="$4"
    sed -i "s/^eps:.*/eps: ${eps}/" "$CONFIG"
    sed -i "s/^tightening:.*/tightening: ${tight}/" "$CONFIG"
    sed -i "s/^l1_lmbd:.*/l1_lmbd: ${l1}/" "$CONFIG"
    sed -i "s/^sensitivity_tag:.*/sensitivity_tag: '${tag}'/" "$CONFIG"
    # Ensure reduced_ls=True, expert_only=False, solver=gurobi, no snapshots
    sed -i "s/^solver:.*/solver: 'gurobi' #['ipopt', 'gurobi']/" "$CONFIG"
    sed -i "s/^expert_only:.*/expert_only: False/" "$CONFIG"
    sed -i "s/^reduced_ls:.*/reduced_ls: True/" "$CONFIG"
    sed -i "s/^save_snapshots:.*/save_snapshots: False/" "$CONFIG"
}

# ── Sweep A: ε (CA constraint screening threshold) ────────────────────────────
echo ""; echo "========== SWEEP A: eps =========="; echo ""
for eps in 0.001 0.005 0.01 0.05 0.10; do
    tag="eps_${eps}"
    echo "--- Config: eps=${eps}, tightening=2.3, l1_lmbd=100 ---"
    set_config "$eps" "2.3" "100" "$tag"
    run_sim "$tag"
done

# Sweep B (ζ tightening) skipped: requires retraining RAID-Net for each value.

# ── Sweep C: λ (L1 penalty weight) ────────────────────────────────────────────
echo ""; echo "========== SWEEP C: l1_lmbd (lambda) =========="; echo ""
for l1 in 10 50 100 150 200; do
    tag="l1_${l1}"
    echo "--- Config: eps=0.01, tightening=2.3, l1_lmbd=${l1} ---"
    set_config "0.01" "2.3" "$l1" "$tag"
    run_sim "$tag"
done

# Restore nominal config
set_config "0.01" "2.3" "100" ""
sed -i "s/^save_snapshots:.*/save_snapshots: True/" "$CONFIG"
echo ""; echo "========== ALL SWEEPS DONE. Results in $OUTDIR =========="; echo ""
