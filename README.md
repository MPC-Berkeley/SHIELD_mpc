# SHIELD MPC
<div align="center">
<img src="https://github.com/MPC-Berkeley/SHIELD_mpc/assets/main.png" width="500">
</div>

# Example Simulation Results
<div align="center">
   <table style="border:none;">
        <tr>
            <td style="border: none;" align="center">
                <img src="https://github.com/MPC-Berkeley/SHIELD_mpc/assets/final_eval_08ee9351335b5c9a_29_starting_unprotected_cross_turn_N14_59.gif" width="400" />
                <div>Proposed: SHIELD MPC</div>
            </td>
            <td style="border: none;" align="center">
                <img src="https://github.com/MPC-Berkeley/SHIELD_mpc/assets/final_eval_08ee9351335b5c9a_29_starting_unprotected_cross_turn_N14_59_expert.gif" width="400">
                <div>Baseline: Full MPC</div>
            </td>
        </tr>
    </table>

   <table style="border:none;">
        <tr>
            <td style="border: none;" align="center">
                <img src="https://github.com/MPC-Berkeley/SHIELD_mpc/assets/final_eval_23b782750976520b_0_high_magnitude_speed_N14_7.gif" width="400" />
                <div>Proposed: SHIELD MPC</div>
            </td>
            <td style="border: none;" align="center">
                <img src="https://github.com/MPC-Berkeley/SHIELD_mpc/assets/final_eval_23b782750976520b_0_high_magnitude_speed_N14_7_expert.gif" width="400">
                <div>Baseline: Full MPC</div>
            </td>
        </tr>
    </table>

<img src="https://github.com/MPC-Berkeley/SHIELD_mpc/assets/results_table.png" width="500">

<strong>> x35 Improvement in the total computation time!</strong>

</div>


# Downloading nuPlan mini v1.1 Dataset:
1) Go to https://www.nuscenes.org/nuplan and download nuPlan mini v1.1 dataset.
2) Follow the instructions in [nuplan-devkit](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html) to set up the dataset
 
# Installation:
Conda:
```
conda env create -f environment.yml
conda activate shield
```

UniTraj:
1) Install UniTraj and its dependencies from here: https://github.com/vita-epfl/UniTraj?tab=readme-ov-file
2) Do the data preparation step in UniTraj installation using ScenarioNet


# Training the Classifier:

### Collecting the Classifier Training Data
Create a folder named ```/expert_data``` in which the collected data and evaluation results will be saved.
```
export NUPLAN_ROOT_DIR="<directory of nuplan-devkit>"
```
Run ALL of the scripts from the tutorials directory
```
cd tutorials
```

```
python nuplan_data_collection.py

```

### Preprocessing Training Data
```
python nuplan_process_data.py --filepath <dir to collected data>
```

### Training the Clasifier using the collected data
```
python nuplan_RAIDNET_train_joint.py
```

### Evaluating the Classifer
```
python nuplan_RAIDNET_evaluate_binary_joint.py
```

# nuPlan Evaluation:

### Evaluating the SHIELD algorithm in closedloop
```
python nuplan_closedloop_evaluate.py
```
After each simulation is completed, it saves the video and time snapshots in ```expert_data/video``` and
saves a simulation log to ```expert_data``` folder

### Printing the results of closedloop simulations
```
python nuplan_closedloop_eval_stats.py --filename <dir to the simulation log>
```
