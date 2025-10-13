# Useful imports
import os
from pathlib import Path
import tempfile
from nuplan.planning.script.run_simulation import run_simulation as main_simulation
from nuplan.planning.simulation.planner.smpc_planner import SMPCPlanner, IDMPlanner
import hydra
import yaml

from tutorials.utils.tutorial_utils import construct_simulation_hydra_paths

#SMPC config
with open('/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_config.yaml', 'r') as f:
    smpc_config = yaml.load(f, Loader=yaml.FullLoader)

# Location of paths with all simulation configs
BASE_CONFIG_PATH = os.path.join(os.getenv('NUPLAN_TUTORIAL_PATH', ''), '../nuplan/planning/script')
simulation_hydra_paths = construct_simulation_hydra_paths(BASE_CONFIG_PATH)

# Create a temporary directory to store the simulation artifacts
SAVE_DIR = tempfile.mkdtemp()

# Select simulation parameters
#get file names from a directory
directory_path = '/home/mpc/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/mini/'

#Data directory
log_list = [f.split('.db')[0] for f in sorted(os.listdir(directory_path)) if os.path.isfile(os.path.join(directory_path, f))]

scenario_types=[
#   'on_intersection',
#   'on_traffic_light_intersection',
  'starting_unprotected_cross_turn',
  'starting_unprotected_noncross_turn',
  'starting_u_turn',
  'starting_protected_cross_turn',
  'starting_protected_noncross_turn',
  'changing_lane',
  'starting_left_turn',
  'starting_right_turn',
  'crossed_by_vehicle',
#   'following_lane_with_lead',
#   'following_lane_with_slow_lead',
  'near_multiple_vehicles',
  'traversing_intersection',
  'near_high_speed_vehicle',
  'stopping_with_lead',
  'high_magnitude_speed',
#   'low_magnitude_speed',
  'near_high_speed_vehicle',
  'on_stopline_stop_sign',
  'high_lateral_acceleration',
#   'traversing_traffic_light_intersection'
    ]

print('total log list length:',len(log_list))
log_list = log_list[59:] # 2021.10.06.17.43.07_veh-28_00508_00877 for 08ee9351335b5c9a
scenario_types = ['starting_unprotected_cross_turn'] #for 08ee9351335b5c9a
num_scenarios = 30 #float: fraction, int: number of scenarios to use from the log #for 08ee9351335b5c9a

# log_list = [log_list[7]] #for 23b782750976520b
# scenario_types = ['high_magnitude_speed'] #for 23b782750976520b
# num_scenarios = 0.99 #for 23b782750976520b
# Also, go to simulation_builder.py to turn on the filter for this specific scenario

#General Case
# log_list = log_list[12:]
# num_scenarios = 3

num_scenarios_per_type = 1
for it, log in enumerate(log_list):
    print('#'.center(50, '#'))
    it += 59
    # it += 7
    # it += 12
    if True:
        try:
            print(f'[Iter:{it}] Collecting data from log: ', log)
            EGO_CONTROLLER = 'perfect_tracking_controller'  # [log_play_back_controller, perfect_tracking_controller]
            # OBSERVATION = 'box_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]
            OBSERVATION = 'idm_agents_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]
            DATASET_PARAMS = [
                'scenario_builder=nuplan_mini',  # [nuplan, nuplan_mini] use nuplan mini database (2.5h of 8 autolabeled logs i n Las Vegas)
                f"scenario_filter.log_names=[{str(log)}]",
                f'scenario_filter.scenario_types={scenario_types}', 
                # f'scenario_filter.num_scenarios_per_type={num_scenarios_per_type}',  # use n scenarios per type
                f'scenario_filter.limit_total_scenarios={num_scenarios}',  # use n total scenarios
                'scenario_filter.remove_invalid_goals=true',  
            ]

            # Initialize configuration management system
            hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
            hydra.initialize(config_path=simulation_hydra_paths.config_path)

            # Compose the configuration
            cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=[
                f'group={SAVE_DIR}',
                f'experiment_name=smpc_expert_trajectory',
                f'job_name=data_collection', 
                'experiment=${experiment_name}/${job_name}',
                'worker=sequential',
                f'ego_controller={EGO_CONTROLLER}',
                f'observation={OBSERVATION}',
                f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
                'output_dir=${group}/${experiment}',
                *DATASET_PARAMS,
            ])

            '''
            Initilize the planner
            '''
            # planner = SimplePlanner(horizon_seconds=10.0, sampling_time=0.2, acceleration=[0.0, 0.0])

            if smpc_config['collision_avoidance_method'] == 'obca':
                #OBCA constraints
                ev_noise_std=[0.01,0.01]
                tv_noise_std=[0.05, 0.05]
            else:
                #Affine CA constraints
                ev_noise_std=[0.01,0.01]
                tv_noise_std=[0.2, 0.2]

            print('Initializing the SMPC Planner...')
            planner = SMPCPlanner(ev_noise_std=ev_noise_std, tv_noise_std=tv_noise_std, iter=it, evaluation_mode = True)

            # Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
            main_simulation(cfg, planner)
        except:
            print(f'Error occurred while running NuPlan for log: {log}')
            continue