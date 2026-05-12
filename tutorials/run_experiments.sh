#!/bin/bash
set -e

CONFIG=/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_config_eval.yaml
EVAL=/home/mpc/nuplan-devkit/tutorials/nuplan_closedloop_evaluate.py
BUILDER=/home/mpc/nuplan-devkit/nuplan/planning/script/builders/simulation_builder.py

run_sim() {
    cd /home/mpc/nuplan-devkit/tutorials
    NUPLAN_ROOT_DIR=/home/mpc/nuplan-devkit \
    PYTHONPATH=/home/mpc/nuplan-devkit \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    conda run -n nuplan python nuplan_closedloop_evaluate.py
}

# ── Experiment 1: 08ee  Gurobi  expert_only=False ──────────────────────────
echo ""; echo "========== EXP 1: 08ee | Gurobi | expert_only=False =========="; echo ""
sed -i "s/^solver:.*/solver: 'gurobi' #['ipopt', 'gurobi']/" $CONFIG
sed -i "s/^expert_only:.*/expert_only: False/" $CONFIG
# scenario script: 08ee
sed -i "s|^log_list = \[log_list\[7\]\].*|# log_list = [log_list[7]] #for 23b782750976520b|" $EVAL
sed -i "s|^# log_list = \[log_list\[59\]\].*|log_list = [log_list[59]] # for 08ee9351335b5c9a|" $EVAL
sed -i "s|^scenario_types = \['high_magnitude_speed'\].*|# scenario_types = ['high_magnitude_speed'] #for 23b|" $EVAL
sed -i "s|^# scenario_types = \['starting_unprotected_cross_turn'\].*|scenario_types = ['starting_unprotected_cross_turn'] #for 08ee|" $EVAL
sed -i "s|^num_scenarios = 100.*|# num_scenarios = 100 #for 23b|" $EVAL
sed -i "s|^# num_scenarios = 1 .*|num_scenarios = 1 #for 08ee|" $EVAL
sed -i 's|scenario_filter.scenario_tokens=\[23b782750976520b\]|scenario_filter.scenario_tokens=[08ee9351335b5c9a]|' $EVAL
sed -i "s|^    # it += 59|    it += 59|" $EVAL
sed -i "s|^    it += 7|    # it += 7|" $EVAL
run_sim

# ── Experiment 2: 08ee  IPOPT  expert_only=True ────────────────────────────
echo ""; echo "========== EXP 2: 08ee | IPOPT | expert_only=True =========="; echo ""
sed -i "s/^solver:.*/solver: 'ipopt' #['ipopt', 'gurobi']/" $CONFIG
sed -i "s/^expert_only:.*/expert_only: True/" $CONFIG
run_sim

# ── Experiment 3: 23b  IPOPT  expert_only=False ────────────────────────────
echo ""; echo "========== EXP 3: 23b | IPOPT | expert_only=False =========="; echo ""
sed -i "s/^expert_only:.*/expert_only: False/" $CONFIG
sed -i "s|^log_list = \[log_list\[59\]\].*|# log_list = [log_list[59]] #for 08ee|" $EVAL
sed -i "s|^# log_list = \[log_list\[7\]\].*|log_list = [log_list[7]] #for 23b782750976520b|" $EVAL
sed -i "s|^# scenario_types = \['high_magnitude_speed'\].*|scenario_types = ['high_magnitude_speed'] #for 23b|" $EVAL
sed -i "s|^scenario_types = \['starting_unprotected_cross_turn'\].*|# scenario_types = ['starting_unprotected_cross_turn'] #for 08ee|" $EVAL
sed -i "s|^# num_scenarios = 100.*|num_scenarios = 100 #for 23b|" $EVAL
sed -i "s|^num_scenarios = 1 .*|# num_scenarios = 1 #for 08ee|" $EVAL
sed -i 's|scenario_filter.scenario_tokens=\[08ee9351335b5c9a\]|scenario_filter.scenario_tokens=[23b782750976520b]|' $EVAL
sed -i "s|^    it += 59|    # it += 59|" $EVAL
sed -i "s|^    # it += 7|    it += 7|" $EVAL
run_sim

echo ""; echo "========== ALL 3 EXPERIMENTS DONE =========="; echo ""
