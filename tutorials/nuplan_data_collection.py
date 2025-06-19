# from tutorials.utils.tutorial_utils import setup_notebook

# setup_notebook()

# (Optional) Increase notebook width for all embedded cells to display properly
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
nuboard = False
# log_list = ['2021.06.09.11.54.15_veh-12_04366_04810']
# log_list = ['2021.06.09.12.39.51_veh-26_05620_06003'] #dual class 0
# log_list = ['2021.05.12.23.36.44_veh-35_01133_01535'] #infeasible
#2021.05.12.23.36.44_veh-35_01133_01535 #Infeasibility from start
# log_list = ['2021.06.07.12.54.00_veh-35_01843_02314']
# log_list = log_list[2:]
# scenario_types=[
#   'behind_long_vehicle',
#   'crossed_by_vehicle',
#   'following_lane_with_lead',
#   'following_lane_with_slow_lead',
#   'following_lane_without_lead',
#   'high_lateral_acceleration',
#   'high_magnitude_jerk',
#   'high_magnitude_speed',
#   'low_magnitude_speed',
#   'medium_magnitude_speed',
#   'near_high_speed_vehicle',
#   'near_multiple_vehicles',
#   'on_intersection',
#   'on_traffic_light_intersection',
#   'starting_high_speed_turn',
#   'starting_protected_cross_turn',
#   'starting_protected_noncross_turn',
#   'starting_right_turn',
#   'starting_left_turn',
#   'changing_lane',
#   'starting_u_turn',
#   'starting_unprotected_cross_turn',
#   'starting_unprotected_noncross_turn',
#   'traversing_intersection',
#   'traversing_traffic_light_intersection',]
scenario_types=[
#   'on_intersection',
#   'on_traffic_light_intersection',
  'starting_protected_cross_turn',
  'starting_protected_noncross_turn',
  'starting_left_turn',
  'changing_lane',
  'starting_unprotected_cross_turn',
#   'starting_unprotected_noncross_turn',
  'starting_right_turn',
#   'traversing_intersection',
#   'traversing_traffic_light_intersection'
    ]
print('total log list length:',len(log_list))
log_list = log_list
for it, log in enumerate(log_list):
    print('#'.center(50, '#'))
    if 60 > it >= 0:
        try:
            print(f'[Iter:{it}] Collecting data from log: ', log)
            EGO_CONTROLLER = 'perfect_tracking_controller'  # [log_play_back_controller, perfect_tracking_controller]
            # OBSERVATION = 'box_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]
            OBSERVATION = 'idm_agents_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]
            DATASET_PARAMS = [
                'scenario_builder=nuplan_mini',  # use nuplan mini database (2.5h of 8 autolabeled logs i n Las Vegas)
                f"scenario_filter.log_names=[{str(log)}]",
                f'scenario_filter.scenario_types={scenario_types}', 
                'scenario_filter.limit_total_scenarios=2',  # use n total scenarios
                'scenario_filter.remove_invalid_goals=true',  # use 1 scenario per log
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
                tv_noise_std=[0.5, 0.5]
            else:
                #Affine CA constraints
                ev_noise_std=[0.01,0.01]
                tv_noise_std=[0.3, 0.3]

            print('Initializing the SMPC Planner...')
            planner = SMPCPlanner(ev_noise_std=ev_noise_std, tv_noise_std=tv_noise_std, iter=it)

            # Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
            main_simulation(cfg, planner)

            if nuboard:
                print('Initializing Nuboard'.center(50, '*'))
                # #Nuboard
                # Get nuBoard simulation file for visualization later on
                simulation_file = [str(file) for file in Path(cfg.output_dir).iterdir() if file.is_file() and file.suffix == '.nuboard']

                # Launch Nuboard
                from tutorials.utils.tutorial_utils import construct_nuboard_hydra_paths

                # Location of paths with all nuBoard configs
                nuboard_hydra_paths = construct_nuboard_hydra_paths(BASE_CONFIG_PATH)

                # Initialize configuration management system
                hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
                hydra.initialize(config_path=nuboard_hydra_paths.config_path)

                # Compose the configuration
                cfg = hydra.compose(config_name=nuboard_hydra_paths.config_name, overrides=[
                    'scenario_builder=nuplan_mini',  # set the database (same as simulation) used to fetch data for visualization
                    f'simulation_path={simulation_file}',  # nuboard file path, if left empty the user can open the file inside nuBoard
                    f'hydra.searchpath=[{nuboard_hydra_paths.common_dir}, {nuboard_hydra_paths.experiment_dir}]',
                ])

                from nuplan.planning.script.run_nuboard import main as main_nuboard

                # Run nuBoard
                main_nuboard(cfg)
        except:
            print(f'Error occurred while running NuPlan for log: {log}')
            continue