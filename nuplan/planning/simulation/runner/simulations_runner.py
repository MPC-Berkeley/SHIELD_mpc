from __future__ import annotations

import logging
import time
import math
from typing import Any, Callable, List
import copy
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.abstract_runner import AbstractRunner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.planner.smpc_planner import SMPCPlanner
logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader
from unitraj.datasets import common_utils
from unitraj.models import build_model
from unitraj.datasets import build_dataset
from unitraj.models.wayformer.wayformer import Wayformer
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
import numpy
import os
import pdb

def for_each(fn: Callable[[Any], Any], items: List[Any]) -> None:
    """
    Call function on every item in items
    :param fn: function to be called fn(item)
    :param items: list of items
    """
    for item in items:
        fn(item)


class SimulationRunner(AbstractRunner):
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulation: Simulation, planner: AbstractPlanner):
        """
        Initialize the simulations manager
        :param simulation: Simulation which will be executed
        :param planner: to be used to compute the desired ego's trajectory
        """
        self._simulation = simulation
        self._planner = planner

    def _initialize(self) -> None:
        """
        Initialize the planner
        """
        # Execute specific callback
        self._simulation.callback.on_initialization_start(self._simulation.setup, self.planner)

        # Initialize Planner
        self.planner.initialize(self._simulation.initialize(sim_mode='closedloop'))

        # Initialize WayformerDataset for Unitraj format (Wayformer)
        #Make sure to update the config file below to match the dataset and the NuPlan scenario split
        cfg = OmegaConf.load('/home/mpc/UniTraj/unitraj/configs/config.yaml')
        model_cfg = OmegaConf.load('/home/mpc/UniTraj/unitraj/configs/method/wayformer.yaml')
        OmegaConf.set_struct(cfg, False)  # Open the struct
        cfg = OmegaConf.merge(cfg, model_cfg)
        cfg.method = model_cfg
        cfg['eval'] = True

        self.wayformer_dataset = build_dataset(cfg, val=True)

        # Initialize Wayformer model
        self.wayformer_model = build_model(cfg)

        # val_loader = DataLoader(
        #     self.wayformer_dataset, batch_size=1, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        #     collate_fn=self.wayformer_dataset.collate_fn)

        # trainer = pl.Trainer(
        #     inference_mode=True,
        #     logger=None,
        #     devices=1,
        #     accelerator="cpu" if cfg.debug else "gpu",
        #     profiler="simple",
        # )
        # pdb.set_trace()
        # pred = trainer.predict(model=self.wayformer_model, dataloaders=val_loader, return_predictions=True, ckpt_path='/home/mpc/UniTraj/unitraj/unitraj_ckpt/test/epoch=933-val/brier_fde=0.60.ckpt')

        # Load the model checkpoint
        self.wayformer_model = Wayformer.load_from_checkpoint('/home/mpc/UniTraj/unitraj/unitraj_ckpt/nuplan_mini/epoch=933-val/brier_fde=0.60.ckpt',config=cfg)
        self.wayformer_model.to('cpu')
        # Execute specific callback
        self._simulation.callback.on_initialization_end(self._simulation.setup, self.planner)

    @property
    def planner(self) -> AbstractPlanner:
        """
        :return: Planner used by the SimulationRunner
        """
        return self._planner

    @property
    def simulation(self) -> Simulation:
        """
        :return: Simulation used by the SimulationRunner
        """
        return self._simulation

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: Get the scenario relative to the simulation.
        """
        return self.simulation.scenario

    def get_wayformer_input(self, planner_input, scenario: AbstractScenario):
        from scenarionet.converter.nuplan.utils import convert_nuplan_scenario
        metadrive_scenario = convert_nuplan_scenario(scenario,version='v1.1',is4inference=True,planner_input=planner_input)
        metadrive_scenario = metadrive_scenario.update_summaries(metadrive_scenario)
        output = self.wayformer_dataset.preprocess(metadrive_scenario,is4inference=True)
        output = self.wayformer_dataset.process(output)
        output = self.wayformer_dataset.postprocess(output,is4inference=True)

        return output
    
    def wayformer_inference(self,planner_input):
        #preprocess, process and postprocess for Unitraj format
        x = self.get_wayformer_input(planner_input,self.simulation.scenario)
        batch_x = self.wayformer_dataset.collate_fn(x)

        # Call the model for inference
        self.wayformer_model.eval()
        with torch.no_grad():
            output, _ = self.wayformer_model.forward(batch_x,is4inference=True)
        pred = output['predicted_trajectory']
        prob = output['predicted_probability']
        use_square_gmm = True
        debug = False
        rho_limit = 0.5
        log_std_range = (-1.609, 0.3)
        # pred: shape [B, c, T, 5] c trajectories for the ego agents with every point being the params of
                                    # Bivariate Gaussian distribution.
        # Bivariate Gaussian Distribution params: [mu_x, mu_y, sigma_x, sigma_y, rho] (rho is the correlation coefficient)                                      
        #post-processing the parameters
        #Assume square gaussian distribution
        log_std1 = torch.clip(pred[:, :, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(pred[:, :, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)
        std2 = torch.exp(log_std2)
        pred[:, :, :, 2] = std1
        pred[:, :, :, 3] = std2
        pred[:, :, :, 4] = torch.clip(pred[:, :, :, 4], min = -rho_limit, max = rho_limit)  #clip the rho values

        if use_square_gmm:
            pred[:, :, :, 3] = pred[:, :, :, 2]
            pred[:, :, :, 4] = torch.zeros_like(pred[:, :, :, 4])  # Set rho to 0 for square Gaussian

        # From center coordinate to traj coordinate
        center_objects_world = batch_x['input_dict']['center_objects_world'] #(N_vh, 10)
        #rotate about the z axis

        pred[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=pred[:, :, :, 0:2].reshape(pred.shape[0], -1, 2),
            angle=center_objects_world[:, 6]
        ).reshape(pred.shape[0], pred.shape[1], pred.shape[2], 2)

        # offset map center
        pred[:,:,:,0:2] += center_objects_world[:, None, None, 0:2]

        state = self.simulation.scenario.get_ego_state_at_iteration(0)
        scenario_center = torch.tensor([state.waypoint.x, state.waypoint.y]).numpy()[None, None, None, :]
        pred[:,:,:,0:2] += scenario_center
        if debug:
            self.plot_gaussian_modes_over_time(pred,planner_input)

        pred_filtered, prob_filtered, tv_params, tv_track_tokens, idx = self.filter_predictions(pred, prob, planner_input, x)
        center_objects_world_filtered = center_objects_world[idx,:]
        tv_psi, anomaly_heading_idx = self.get_heading_wayformer(pred_filtered,center_objects_world_filtered[:,[6]].repeat(1,pred_filtered.shape[1]))
        if anomaly_heading_idx:
            #If anomaly heading detected then set the heading of the corresponding index to the initial heading
            for (i,m,t) in anomaly_heading_idx:
                tv_psi[i,m,t,0] = center_objects_world_filtered[i,6]
                # logger.warning(f"Anomaly heading detected for vehicle index {i} at time {t}. Setting to initial heading.")
        # if planner_input.iteration.index >= 26:
        #     pdb.set_trace()
        return pred_filtered.numpy(), prob_filtered.numpy(), tv_params, tv_psi.numpy(), tv_track_tokens

    def filter_predictions(self, pred, prob, planner_input, wayformerinput):
        """
        Filter the predictions based on the probability and distance from the ego vehicle's current state
        :param pred: predictions
        :param prob: probabilities
        :return: filtered predictions
        """
        ego_state, observation = planner_input.history.current_state
        M = self.planner.config['num_modes'] #Top 3 likely modes
        V = self.planner.config['num_tvs'] #Top 4 vehicles

        N_veh, N_modes, T, D = pred.shape

        pred_M_ind = torch.argsort(prob,dim=1, descending=True)[:,:M]  # Get the top M modes
        index_exp = pred_M_ind.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, D)
        pred_gathered = torch.gather(pred, dim=1, index=index_exp) 

        # Get the ego vehicle's current state
        ego_x, ego_y = ego_state.waypoint.x, ego_state.waypoint.y
        ego_heading = ego_state.waypoint.heading

        # Get the distance from the ego vehicle's current state
        distances = ((pred_gathered[:,:,0,0] - ego_x) ** 2 + (pred_gathered[:, :, 0, 1] - ego_y) ** 2).sqrt()
        avg_distances_over_modes = distances.mean(dim=1)

        # Find index of ego
        ego_index = -1
        for i in range(len(avg_distances_over_modes)):
            if wayformerinput[i]['center_objects_id'] == 'ego':
                ego_index = i
                break

        # Get the indices of the V + 1 (always includes the ego) vehicles closest to the ego vehicle out of N_veh vehicles
        filtered_idx = []
        temp = torch.argsort(avg_distances_over_modes)
        closest_vehicles_indices = temp[temp!=ego_index][:V]
        if closest_vehicles_indices.shape[0] == 0:
            # If no vehicles are found, use dummy vehicles (same heading as ego, x and y really far away)
            # This is a fallback mechanism and should rarely happen
            pred_output = pred.repeat(V,1,1,1)
            pred_output[:,:,:,0] += 1000.0 #x
            pred_output[:,:,:,1] += 1000.0 #y
            prob_output = prob.repeat(V,1)
            tv_params = [(4, 2)] * V #arbitrary length and width
            tv_track_tokens = ['dummy'] * V
            filtered_idx = [ego_index]*V
            print("No vehicles found, using dummy vehicles.")
        else:
            while closest_vehicles_indices.shape[0] < V:
                print(f"Only {closest_vehicles_indices.shape[0]} vehicles found, expected {V}. Using all available vehicles.")
                add_ind = V - closest_vehicles_indices.shape[0]
                closest_vehicles_indices = torch.hstack([closest_vehicles_indices,closest_vehicles_indices[:add_ind]])

            # Compute relative positions and angles
            dx = pred_gathered[:, 0, 0, 0] - ego_x
            dy = pred_gathered[:, 0, 0, 1] - ego_y
            distances = torch.sqrt(dx**2 + dy**2)
            angles = torch.atan2(dy, dx)  # angle from ego to target

            # Normalize angles to [-pi, pi]
            angle_diff = (angles - ego_heading + math.pi) % (2 * math.pi) - math.pi

            # Vehicles within ±80 degrees (~0.5236 rad)
            fov_mask = (angle_diff.abs() <= math.radians(80))

            # Sort both FOV and non-FOV by distance
            sorted_fov = torch.argsort(distances[fov_mask])
            sorted_out_fov = torch.argsort(distances[~fov_mask])

            fov_indices = torch.arange(len(distances))[fov_mask][sorted_fov]
            out_fov_indices = torch.arange(len(distances))[~fov_mask][sorted_out_fov]

            # Combine
            num_fov_veh = len(fov_indices)
            prioritized_indices = torch.cat([fov_indices, out_fov_indices], dim=0)
            # V-1 from top of prioritized list (excluding ego)
            top_fov_vehicles = prioritized_indices[prioritized_indices != ego_index][:V-1]

            # One more from outside FOV starting after num_fov_veh (excluding ego)
            if ego_index in fov_indices:
                non_fov_rest = prioritized_indices[prioritized_indices != ego_index][num_fov_veh-1:]
            else:
                non_fov_rest = prioritized_indices[prioritized_indices != ego_index][num_fov_veh:]
            if len(non_fov_rest) > 0:
                extra_vehicle = non_fov_rest[:1]  # Select just one
                closest_vehicles_indices = torch.cat([top_fov_vehicles, extra_vehicle])
            else:
                closest_vehicles_indices = top_fov_vehicles  # Fallback: use only top FOV

            #(length, width)
            detection_track_tokens = [v.metadata.track_token for v in observation.tracked_objects.tracked_objects]
            tv_params = []
            tv_track_tokens = []
            if 0 < closest_vehicles_indices.shape[0] < V:
                print(f"Only {closest_vehicles_indices.shape[0]} vehicles found, expected {V}. Creating dummy vehicles.")
                #Fill with a dummy vehicle
                closest_vehicles_indices = torch.cat([closest_vehicles_indices, torch.tensor([closest_vehicles_indices[-1]] * (V - closest_vehicles_indices.shape[0]))])
            pred_output = pred_gathered[closest_vehicles_indices]
            prob_output = torch.gather(prob[closest_vehicles_indices], dim = 1, index = pred_M_ind[closest_vehicles_indices])
            
            for i in range(V):
                ind = closest_vehicles_indices[i]
                try:
                    idx4tv_param = detection_track_tokens.index(wayformerinput[ind]['center_objects_id'])
                    length = observation.tracked_objects.tracked_objects[idx4tv_param].box.length
                    width = observation.tracked_objects.tracked_objects[idx4tv_param].box.width
                    # Get the length and width of the vehicle
                    tv_params.append((length, width))
                    tv_track_tokens.append(wayformerinput[ind]['center_objects_id'])
                except:
                    # If the vehicle is not found in the tracked objects, use default values
                    tv_params.append((0, 0))
                    tv_track_tokens.append(wayformerinput[ind]['center_objects_id'])
                    print(f"Vehicle with token {wayformerinput[ind]['center_objects_id']} not found in tracked objects.")
                filtered_idx.append(ind.item())

        return pred_output, prob_output, tv_params, tv_track_tokens, filtered_idx
    
    def get_heading_wayformer(self, pred_filtered, init_psi):
        """
        Get the heading of the vehicle from the planner input
        :param planner_input: planner input
        :return: heading of the vehicle
        """
        # Get the ego vehicle's current state
        n_tv, M, T, _  = pred_filtered.shape
        heading = torch.zeros((n_tv, M, T-1, 1))
        heading[:,:,0,0] = init_psi
        # Get the heading of the vehicle
        anomaly_heading_idx = []
        threshold = 0.4 # Threshold for detecting anomalies in heading change (radians)
        for i in range(n_tv):
            for t in range(1,T-1):            
                #instantaneous heading
                xy_diff = pred_filtered[i, :, t+1, 0:2] - pred_filtered[i, :, t, 0:2]
                torch.atan2(xy_diff[:,1], xy_diff[:,0], out=heading[i, :, t, 0])
                for m in range(xy_diff.shape[0]):
                    if xy_diff[m,1] == 0 or abs(heading[i, m, t, 0]-heading[i, m, t-1, 0])>threshold:
                        anomaly_heading_idx.append((i,m,t))
        return heading, anomaly_heading_idx

    def plot_gaussian_modes_over_time(self,gaussian_tensor, planner_input):
        """
        Plot bivariate Gaussian distributions as ellipses for each vehicle and mode across time.
        
        Args:
            gaussian_tensor (np.ndarray): Shape (N_veh, N_modes, T, 5), where 5 = [mu_x, mu_y, sigma_x, sigma_y, rho]
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Ellipse
        import matplotlib.cm as cm
        N_veh, N_modes, T, _ = gaussian_tensor.shape
        fig, ax = plt.subplots(figsize=(12, 8))

        # Assign each vehicle a unique base color
        base_colors = cm.get_cmap('tab20', N_veh)

        all_x, all_y = [], []

        #plot all vehicles as blue rectangles
        _, observation = planner_input.history.current_state
        for v in observation.tracked_objects.tracked_objects:
            if v.metadata.category_name == 'vehicle':
                xy = [v.center.x,v.center.y]
                l = v.box.length

                #plot a square
                rect = plt.Rectangle((xy[0]-l/2, xy[1]-l/2), l, l, angle=0.0, color='black', alpha=1)
                ax.add_patch(rect)
        for v in range(N_veh):
            base_color = base_colors(v)
            
            for m in range(N_modes):
                alpha = (m + 1) / N_modes  # lighter for lower modes, darker for higher
                for t in range(T):
                    mu_x, mu_y, sigma_x, sigma_y, rho = gaussian_tensor[v, m, t]

                    if sigma_x <= 0 or sigma_y <= 0 or not np.isfinite([mu_x, mu_y, sigma_x, sigma_y, rho]).all():
                        continue  # Skip degenerate or invalid Gaussians

                    # Clamp rho for numerical stability
                    rho = np.clip(rho, -0.999, 0.999)

                    # Build covariance matrix
                    cov = np.array([
                        [sigma_x**2, rho * sigma_x * sigma_y],
                        [rho * sigma_x * sigma_y, sigma_y**2]
                    ])

                    try:
                        lambda_, v_ = np.linalg.eig(cov)
                        lambda_ = np.clip(lambda_, 1e-4, None)  # clip small/negative eigenvalues
                        angle = np.degrees(np.arctan2(*v_[:, 0][::-1]))
                        width, height = 2 * np.sqrt(lambda_)  # 1-sigma ellipse

                        ellipse = Ellipse(
                            xy=(mu_x, mu_y),
                            width=width,
                            height=height,
                            angle=angle,
                            edgecolor='none',
                            facecolor=base_color,
                            alpha=alpha * 0.6
                        )
                        ax.add_patch(ellipse)
                        all_x.append(mu_x)
                        all_y.append(mu_y)
                    except np.linalg.LinAlgError:
                        continue

        if all_x and all_y:
            ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
            ax.set_ylim(min(all_y) - 10, max(all_y) + 10)

        ax.set_title("Bivariate Gaussian Ellipses for All Vehicles and Modes (6s Horizon)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        plt.grid(True)
        plt.show()

    def run(self) -> RunnerReport:
        """
        Run through all simulations. The steps of execution follow:
         - Initialize all planners
         - Step through simulations until there no running simulation
        :return: List of SimulationReports containing the results of each simulation
        """
        start_time = time.perf_counter()

        # Initialize reports for all the simulations that will run
        report = RunnerReport(
            succeeded=True,
            error_message=None,
            start_time=start_time,
            end_time=None,
            planner_report=None,
            scenario_name=self._simulation.scenario.scenario_name,
            planner_name=self.planner.name(),
            log_name=self._simulation.scenario.log_name,
        )

        # Execute specific callback
        self.simulation.callback.on_simulation_start(self.simulation.setup)

        # Initialize all simulations
        self._initialize()
        counter = 0
        viddir = '/home/mpc/nuplan-devkit/nuplan/expert_data/video/N'+str(self.planner.config['N'])+'_' + str(self.planner.config['prediction_method']) + '_' + str(self.planner.config['collision_avoidance_method']) + '/'
        filenames = os.listdir(viddir)
        duplicate_scenario = False
        print(f'Looking for {self.simulation.scenario.token} in {viddir}')
        for filename in filenames:
            if self.simulation.scenario.token in filename:
                print(f"Scenario {self.simulation.scenario.token} already exists. Skipping scenario.")
                duplicate_scenario=True
                break
        duplicate_scenario = False #disable duplicate check
        if self.planner.config['eval_mode']:
            duplicate_scenario = False
        if not duplicate_scenario:
            while self.simulation.is_simulation_running():
                # print(f'Simulation t: {counter}:')
                # Execute specific callback
                self.simulation.callback.on_step_start(self.simulation.setup, self.planner)

                self.planner.set_scenario_id(self.simulation.scenario.token)

                # Perform step
                planner_input = self._simulation.get_planner_input()
                logger.debug("Simulation iterations: %s" % planner_input.iteration.index)

                # Execute specific callback
                self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)

                # Get predictions for planner
                pred, prob, tv_params, tv_psi, tv_track_tokens = self.wayformer_inference(planner_input)
                if isinstance(self.planner, SMPCPlanner):             
                    #Get IDM predictions for planner
                    time_controller_copy = copy.deepcopy(self.simulation._time_controller)
                    if self.planner.config['prediction_method'] == 'idm':
                        if isinstance(self.simulation.setup.observations,IDMAgents):
                            if self.simulation._time_controller.get_iteration().index == 0:
                                ego_traj = list(self.simulation.scenario.get_expert_ego_trajectory())
                                self.planner.ego_traj = ego_traj[:self.planner.config['N']+1]
                                history_buffer = copy.deepcopy(self.simulation._history_buffer)
                                preds = self.simulation._observations.get_idm_predictions(time_controller_copy.get_iteration(), time_controller_copy.next_iteration() if time_controller_copy.next_iteration() is not None else time_controller_copy.get_iteration(), ego_traj[:self.planner.config['N']+1], history_buffer, num_samples=self.planner.config['N'])
                            else:
                                history_buffer = copy.deepcopy(self.simulation._history_buffer)
                                preds = self.simulation._observations.get_idm_predictions(time_controller_copy.get_iteration(), time_controller_copy.next_iteration() if time_controller_copy.next_iteration() is not None else time_controller_copy.get_iteration(), self.planner.get_x_ego(history_buffer), history_buffer, num_samples=self.planner.config['N'])
                            tv_paths_se2 = None

                        else:
                            preds = self.simulation.scenario._get_log_predictions(time_controller_copy.get_iteration(), num_samples=self.planner.config['N'])
                            tv_paths_se2 = self.simulation._scenario._get_agent_paths_from_log()
                    else:
                        #use Wayformer predictions
                        tv_paths_se2 = None
                        scenario_type = self.simulation.scenario.scenario_type
                        preds = (pred, prob, tv_params, tv_psi, tv_track_tokens, scenario_type)
                        if self.simulation._time_controller.get_iteration().index == 0:
                            ego_traj = list(self.simulation.scenario.get_expert_ego_trajectory())
                            self.planner.ego_traj = ego_traj[:self.planner.config['N']+1]
                        tv_paths_se2 = self.simulation._scenario._get_agent_paths_from_log()
                    try:
                        trajectory = self.planner.compute_trajectory(planner_input,preds,tv_paths_se2)
                    except:
                        break
                else:
                    preds = [] #empty preds
                    tv_paths_se2 = None
                    trajectory = self.planner.compute_trajectory(planner_input,preds,tv_paths_se2)
                
                # Propagate simulation based on planner trajectory
                self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)
                self.simulation.propagate(trajectory)

                # Execute specific callback
                self.simulation.callback.on_step_end(self.simulation.setup, self.planner, self.simulation.history.last())
                
                # Store reports for simulations which just finished running
                current_time = time.perf_counter()
                if not self.simulation.is_simulation_running():
                    report.end_time = current_time
                counter += 1

        # Execute specific callback
        if isinstance(self.planner, SMPCPlanner):
            self.planner._callback_end_simulation(logname=self.simulation.scenario.log_name)
        self.simulation.callback.on_simulation_end(self.simulation.setup, self.planner, self.simulation.history)

        planner_report = self.planner.generate_planner_report()
        report.planner_report = planner_report

        return report
