import logging
import math
from typing import List, Tuple, Dict, Union
import numpy as np
from shapely.geometry import LineString, Point, Polygon
import pdb 
from itertools import product
import datetime, cv2
import pickle
import gzip
import faulthandler
import copy
import os
import torch as th
import matplotlib.patches as patches
from tutorials.policies import RAID_NET
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring
from nuplan.planning.simulation.planner.abstract_idm_planner import AbstractIDMPlanner
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.observation.idm.idm_states import IDMAgentState
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent
from nuplan.common.actor_state.agent import Agent
from nuplan.planning.simulation.planner.smpc_predictor import MultiModalPreds as MultiModalPreds
from typing import Optional
import yaml
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.smpc import SMPC
# from nuplan.planning.simulation.planner.smpc_nlp import SMPC

from nuplan.planning.simulation.planner.utils.smpc_utils import flatten, get_preds, make_ca_fun, make_jac_fun, filter_preds
logger = logging.getLogger(__name__)
faulthandler.enable()

class SMPCPlanner(AbstractIDMPlanner):
    """
    The SMPC planner is composed of two parts:
        1. Route planner that constructs a route to the same road block as the goal pose.
        2. SMPC policy controller to control the longitudinal movement of the ego along the planned route.
    """

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        ev_noise_std: List,
        tv_noise_std: List,
        iter: int,
    ):
        """
        Constructor for IDMPlanner
        :param target_velocity: [m/s] Desired velocity in free traffic.
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle.
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front.
        :param accel_max: [m/s^2] maximum acceleration.
        :param decel_max: [m/s^2] maximum deceleration (positive value).
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param occupancy_map_radius: [m] The range around the ego to add objects to be considered.
        """
        with open("/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_config.yaml") as f:
        # with open("/home/hansung/L4SMPC_nuplan/nuplan/planning/simulation/planner/smpc_config_affine.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        super(SMPCPlanner, self).__init__(
            target_velocity=13.0, # (Not used)
            min_gap_to_lead_agent=2.0, #min_gap_to_lead_agent (Not used)
            headway_time=2.0, #headway time (Not used)
            accel_max=self.config['a_max'],
            decel_max=-self.config['a_min'],
            planned_trajectory_samples=self.config['N'],
            planned_trajectory_sample_interval=self.config['dt'],
            occupancy_map_radius=100, #Occupancy_map_radius (Not used)
        )

        self.ev_noise_std = ev_noise_std
        self.tv_noise_std = tv_noise_std
        self.log_iter = iter

        self.u_prev = 0.0 #initialze previous control input(acceleration) to 0 
        self.x_sol = None
        
        self.dual_class = []
        self.expert_action = []
        self.observation = []
        self.preds = []
        self.cl_ego_traj = []
        self.iteration_data = []
        self.ego_planned_trajs = []
        self.figs_w_preds = []
        self.ego_opt_sols_full_state = []
        self.pred_agent_params = []
        self.smpc_params = []
        self.l1_active = []
        
        self._initialized = False
        self.t = 0
        print('SMPC Planner Instantiated')

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._map_api = initialization.map_api
        self._initialize_route_plan(initialization.route_roadblock_ids)
        self._initialized = False

    def get_x_ego(self, history) ->  List[EgoState]:
        if hasattr(self, 'ego_traj'):
            current_time_point = self.ego_traj[-1].time_point
            ego_progress = self._ego_path_linestring.project(Point(*self.ego_traj[-1].center.point.array))
            s, v = ego_progress, self.ego_traj[-1].dynamic_car_state.center_velocity_2d.magnitude()
            s += self.config['dt']*v + 0.5*self.config['dt']**2*0
            v += self.config['dt']*0
            ego_idm_state = IDMAgentState(progress=s, velocity=v)
            current_time_point += TimePoint(int(self._planned_trajectory_sample_interval * 1e6))
            ego_state_tp1 = self._idm_state_to_ego_state(ego_idm_state, current_time_point, self.ego_traj[-1].car_footprint.vehicle_parameters)
            self.ego_traj = self.ego_traj[1:] + [ego_state_tp1]
            return self.ego_traj
        else:
            #Heuristics for now to get the ego trajectory estimation
            ego_state0, _ = history.current_state
            print('Get x_ego:', ego_state0.center.point.x, ego_state0.center.point.y)
            vehicle_parameters = ego_state0.car_footprint.vehicle_parameters
            current_time_point = ego_state0.time_point
            if not self._initialized:
                self._initialize_ego_path(ego_state0)
            a_arr = np.zeros(self.config['N']) #0 acceleration
            ego_traj = []
            ego_progress = self._ego_path_linestring.project(Point(*ego_state0.center.point.array))
            s, v = ego_progress, ego_state0.dynamic_car_state.center_velocity_2d.magnitude()    
            for t in range(self.config['N']+1):     
                if self.config['N']> t > 0:
                    s += self.config['dt']*v + 0.5*self.config['dt']**2*a_arr[t]
                    v += self.config['dt']*a_arr[t]
                elif t == self.config['N']:
                    s += self.config['dt']*v + 0.5*self.config['dt']**2*a_arr[-1]
                    v += self.config['dt']*a_arr[-1]
                ego_idm_state = IDMAgentState(progress=s, velocity=v)
                ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)
                current_time_point += TimePoint(int(self._planned_trajectory_sample_interval * 1e6))
                ego_traj.append(ego_state)
            self.ego_traj = ego_traj
            return self.ego_traj
        
    def get_update_dict(self,current_input: PlannerInput, preds: Union[list,dict], tv_paths_se2: dict) -> dict:
        ego_state, observations = current_input.history.current_state
        routes = [self.ego_route]
        droutes = [self.ego_droute]
        params = {'dt': self.config['dt'], 'N': self.config['N'],'N_TV': self.config['num_tvs'], 'config': self.config}
        ego_progress = self._ego_path_linestring.project(Point(*ego_state.center.point.array))
        x0 = np.array([[ego_progress],[ego_state.dynamic_car_state.center_velocity_2d.magnitude()]])
        if type(preds) == dict:
            prob = preds['prob']
            tv_params = preds['tv_params']
            tv_psi = preds['tv_psi']
            params.update({'prob': prob, 'tv_params': tv_params, 'tv_psi': tv_psi, 'tv_track_tokens': preds['tv_track_tokens']})
            preds = preds['preds']
        z_lin, x_glob, dpos, o_glob, u_tvs, routes, droutes, Qs, tv_psi, tv_params, o0 = get_preds(current_input,
                                                                                                preds,
                                                                                                x0, 
                                                                                                params,
                                                                                                routes,
                                                                                                droutes,
                                                                                                simulation_t=self.t,
                                                                                                u_opt=self.u_opt if hasattr(self, 'u_opt') and self.optimal else None,
                                                                                                ego_traj=self.ego_traj if hasattr(self, 'ego_traj') else None,
                                                                                                ego_p0=self.ego_initial_position,
                                                                                                tv_paths_se2 = tv_paths_se2,
                                                                                                dt=self.config['dt'],is_mm_preds=self.config['is_mm_preds'],
                                                                                                ego_sim_init_state = self.x0 if hasattr(self, 'x0') else None)
        if self.config['prediction_method']=='idm':
            o0 = [np.array([[agent.progress],[agent.velocity]]) for agent in preds[0]] if isinstance(preds[0][0],IDMAgent) else [np.array([[path_to_linestring(tv_paths_se2[agent.metadata.track_token]).project(Point(*agent.center.point.array))],[agent.velocity.magnitude()]]) for agent in preds[0]]
        else:
            pass

        update_dict =   {'x0': x0,
                         'o0': o0,
                        'u_prev': self.u_prev,
                        'z_lin': z_lin,
                        'x_pos': x_glob,
                        'dpos': dpos,
                        'u_tvs': u_tvs,
                        'o_glob': o_glob,
                        'droutes': droutes, 
                        'routes': routes,
                        'preds': preds,
                        'Qs': Qs,
                        'tv_psi': tv_psi,
                        'tv_params': tv_params,
                }
        return update_dict

    def compute_planner_trajectory(self, current_input: PlannerInput, preds = None, tv_paths_se2: Optional[Dict]=None, wayformer_output: Optional[Dict]=None) -> AbstractTrajectory:
        """Inherited, see superclass."""
        print(f'[SMPCPlanner] compute_planner_trajectory: {self.t} iteration')
        # Ego current state
        ego_state, observations = current_input.history.current_state
        detection_track_tokens = [v.metadata.track_token for v in observations.tracked_objects.tracked_objects]
        # print('compute_planner_trajectory:', ego_state.center.point.x, ego_state.center.point.y, ego_state.dynamic_car_state.center_velocity_2d.magnitude())
        if not self._initialized:
            self.x0 = copy.deepcopy(ego_state)
            self._initialize_ego_path(ego_state)
            self.ego_initial_position = ego_state.center.point
            
            N = self.config['N']
            dt = self.config['dt']

            # Lineraized Ego Vehicle Dynamics   
            A = np.array([[1., dt], [0., 1.]])
            B = np.array([0.5*dt**2,dt])

            # Ego route and droute
            s_arr = [point.progress for point in self._ego_path.get_sampled_path()]
            x_arr = [point.x-self.x0.center.point.x for point in self._ego_path.get_sampled_path()] #relative to initial ego position to scale
            y_arr = [point.y-self.x0.center.point.y for point in self._ego_path.get_sampled_path()] #relative to initial ego position to scale
            psi_arr = [point.heading for point in self._ego_path.get_sampled_path()]
            v_arr = [0 for _ in self._ego_path.get_sampled_path()]
            self.ego_route = make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr)
            self.ego_droute = make_jac_fun(self.ego_route)
            self.is_mm_preds = self.config['is_mm_preds']      
            if self.config['eval_mode_category'] == 0:
                # Initialize the Multi-Modal Predictor
                self.mm_predictor = MultiModalPreds(a_lat=self.config['a_lat'],dt=self.config['dt']) 

            if self.config['eval_mode']:
                with open('/home/mpc/nuplan-devkit/tutorials/training_config.yaml') as f:
                    self.raidnet_config = yaml.load(f, Loader=yaml.SafeLoader)
                N = self.config['N']
                with open(f'/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_N{str(N)}_canon_form.pkl', 'rb') as f:
                    canon_prob = pickle.load(f)
                self.canon_prob = canon_prob
                n_modes = [1 for _ in range(self.config['num_tvs'])]
                mode_map = dict(enumerate(product(*[range(n_modes[k]) for k in range(self.config['num_tvs'])])))
                observation_dim = self.config['num_tvs'] * (4*self.config['N'] + 2)
                self.ca_num = len(mode_map)*(self.config['N']-1)*self.config['num_tvs']
                # self.l1_num = sum(n_modes)*(self.config['N']-1)*2 #2 for each position and velocity disturbance feedback w.r.t. the TV
                self.l1_num =  sum(n_modes)*(self.config['N']-1)
                num_layers = self.raidnet_config['num_layers']
                hidden_dim = self.raidnet_config['hidden_dim']
                device = th.device("cuda:0" if th.cuda.is_available() else "cpu") 

                l1_dual_dim = [self.config['N']-1, n_modes, self.config['num_tvs']]
                ca_dual_dim = [self.config['N']-1, len(mode_map), self.config['num_tvs']]
                raidnet_config = {'num_tvs': self.config['num_tvs'], 'num_heads': self.raidnet_config['num_heads'],'dropout_prob':self.raidnet_config['dropout_prob']}
                l1_policy = RAID_NET(raidnet_config,observation_dim, observation_dim, self.l1_num, self.raidnet_config['N']-1, num_layers//2, hidden_dim//2,lambda_dim=self.l1_num, lambda_ubd=self.config['l1_lmbd'],  pred_mode=['l1','binary','binary'])
                ca_policy = RAID_NET(raidnet_config,observation_dim, observation_dim, self.ca_num, self.raidnet_config['N']-1, num_layers//2, hidden_dim//2,lambda_dim=self.ca_num, lambda_ubd=self.config['l1_lmbd'], pred_mode=['ca','binary','binary'])
                self.RAID_NET = [l1_policy,ca_policy]

            # Initialize the SMPC
            offline_mode = True if not self.config['eval_mode'] else False
            self.smpc = SMPC(ev=(A,B),
                    N            =  N,
                    V_MIN        = self.config['v_min'],       #Speed, acceleration constraints
                    V_MAX        = self.config['v_max'], 
                    A_MIN        = self.config['a_min'],
                    A_MAX        =  self.config['a_max'],
                    EV_NOISE_STD    =  self.ev_noise_std,
                    TV_NOISE_STD    = self.tv_noise_std,
                    Q = 1.,       # cost for measuring progress: -Q*s_{t+1}. #was 1.
                    R = 1.,       # cost for penalizing large input rate: (u_{t+1}-u_t).T@R@(u_{t+1}-u_t) #was 1.5
                    ev_length=ego_state.car_footprint.vehicle_parameters.length,
                    offline_mode= offline_mode,
                    solver="ipopt",
                    open_loop = False,
                    eval_mode = False,
                    is_mm_preds=self.config['is_mm_preds'],
                    route = self.ego_route,
                    preds=filter_preds(preds,self.config['num_tvs'],ego_state) if self.config['prediction_method']=='idm' else [[0 for _ in range(self.config['num_tvs'])]],   
                    canon_prob_fn=canon_prob if self.config['eval_mode'] else None,
                    config=self.config,)
            self._initialized = True

        # Update the SMPC parameters
        if not (self.config['eval_mode_category'] == 0):
            if self.config['prediction_method']=='idm':
                update_dict = self.get_update_dict(current_input, filter_preds(preds,self.config['num_tvs'],ego_state),tv_paths_se2)
            else: #wayformer
                pred, prob, tv_params, tv_psi, tv_track_tokens, scenario_type = preds 
                preds_dict = {'preds': pred, 'prob': prob, 'tv_params': tv_params, 'tv_psi': tv_psi, 'tv_track_tokens': tv_track_tokens}
                preds = preds_dict
                update_dict = self.get_update_dict(current_input, preds_dict, tv_paths_se2)
        else:
            if self.config['prediction_method']=='idm':    
                mm_preds = self.mm_predictor.predict(filter_preds(preds,self.config['num_tvs'],ego_state),ego_state,is_mm_preds=self.config['is_mm_preds'])
                update_dict = self.get_update_dict(current_input, mm_preds,tv_paths_se2)
            else:
                #wayformer
                pred, prob, tv_params, tv_psi, tv_track_tokens, scenario_type = preds 
                preds_dict = {'preds': pred, 'prob': prob, 'tv_params': tv_params, 'tv_psi': tv_psi, 'tv_track_tokens': tv_track_tokens}
                mm_preds = preds_dict
                update_dict = self.get_update_dict(current_input, preds_dict, tv_paths_se2)
        leading_vehicle = self.leading_idm_agent(ego_state,observations,current_input)
        leading_vehicle_key =list(leading_vehicle.keys())[0]
        if leading_vehicle is not None and leading_vehicle_key not in preds_dict['tv_track_tokens']:
            leading_agent = leading_vehicle[leading_vehicle_key]
            leading_agent_ind = detection_track_tokens.index(leading_vehicle_key) #check if the leading agent is in the observations
            leading_agent_obs = observations.tracked_objects.tracked_objects[leading_agent_ind]
            #get velocity
            v = leading_agent_obs.velocity.magnitude() #magnitude of the velocity 
            #terminal s of the leading agent (i.e. constant velocity)
            leading_agent.progress += self.smpc.N * self.config['dt'] * v 
            update_dict.update({'leading_vehicle': leading_agent})
        update_dict.update({'leading_vehicle':None,'speed_limit':min(self._policy.target_velocity,ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()+self.config['N']*self.config['a_max']*0.1/2),'ego_sim_initial_state':self.x0,'red_light': self.red_light_leading_idm_agent(ego_state,observations,current_input)})

        if self.config['eval_mode']:
            #update canonical form matrices
            # update_dict.update({'canon_prob':self.canon_prob})
            update_dict.update({'canon_prob':1}) #canon_prob form is provided in the initialization of the SMPC Planner

            #Query RAID-Net 
            if False:
                obs = self.get_observation(ego_state,update_dict['preds'][0])
                l1_duals = self.RAID_NET[0](obs)
                ca_duals = selupdate_dictsf.RAID_NET[1](obs)
            else:
                #test random 0-1 vector
                l1_duals = np.random.randint(2,size=int(self.l1_num))
                ca_duals = np.random.randint(2,size=self.ca_num)

            #update l1 and ca duals
            update_dict.update({'l1_duals':l1_duals, 'ca_duals':ca_duals})
        self.prev_update_dict = update_dict
        pdb.set_trace()
        self.smpc.update(update_dict) 
        # Solve the SMPC
        sol = self.smpc.solve()

        self.optimal = sol['optimal']
        info = {}
        if not self.check_preds(preds):
            print('Empty preds detected')
            raise ValueError
        if self.optimal:
            # Get the optimal DUALS
            if not self.config['eval_mode']: #offline mode
                info.update({"l1_duals":sol["l1_duals"], "ca_duals":sol["ca_duals"]})
                dual_class = 0
                l1_duals_vec = np.fromiter(flatten(info["l1_duals"]),float)
                ca_duals_vec = np.fromiter(flatten(info["ca_duals"]),float)
                #l1 dual active means ||g_1||_inf = 0 or ||g_1||_inf = l1_lmbd
                l1_dual_active = (1-int(np.all(l1_duals_vec<(self.smpc.l1_lmbd-1e-3)*np.ones(l1_duals_vec.shape[0])))) or (1-int(np.all(l1_duals_vec>1e-3*np.ones(l1_duals_vec.shape[0]))))
                ca_duals_active = np.sum(ca_duals_vec>1e-3*np.ones(ca_duals_vec.shape[0]))/ca_duals_vec.shape[0]
                expert_action = np.concatenate((l1_duals_vec,ca_duals_vec))
                if l1_dual_active == 1:
                    if ca_duals_active > 0.05 :
                        dual_class = 3
                    else:
                        dual_class = 1
                elif ca_duals_active > 0.05 :
                    dual_class = 2
                self.dual_class.append(dual_class)
                self.expert_action.append(expert_action)
                self.scenario_type = scenario_type
                print(l1_duals_vec)
                print(dual_class)
                print(ca_duals_active,l1_dual_active)
            if (self.config['eval_mode_category']==0):
                pred = mm_preds
                if self.config['prediction_method']=='idm':
                    preds2save, params2save = self.agent_preds2array(pred)
                else:
                    preds2save, params2save = self.agent_preds2array_wayformer(pred)
                self.pred_agent_params.append(params2save)
                self.preds.append(preds2save)
            else:
                if self.config['prediction_method']=='idm':
                    pred = filter_preds(preds,self.config['num_tvs'],ego_state)
                    preds2save, params2save = self.agent_preds2array(pred)
                else:
                    pred = preds
                    preds2save, params2save = self.agent_preds2array_wayformer(pred)
                self.pred_agent_params.append(params2save)
                self.preds.append(preds2save)
            if self.config['prediction_method']=='idm':
                self.observation.append(self.get_observation(current_input,pred[0])) 
            else:
                self.observation.append(self.get_observation(current_input,pred))
            self.iteration_data.append(observations)
            self.cl_ego_traj.append(ego_state)
            self.ego_planned_trajs.append(self.s2xy(sol['nom_z'][0,1:]))
            self.ego_opt_sols_full_state.append(self.get_ego_full_state())
            self.smpc_params.append(self.smpc.opti.value(self.smpc.params))
            self.l1_active.append(l1_dual_active)
            if leading_vehicle is not None and leading_vehicle_key not in preds_dict['tv_track_tokens']:
                pred.update({'leading_vehicle':leading_agent}) #update the leading agent in the prediction
            pred.update({'leading_vehicle_active':sol['leading_vehicle_active']}) #update the leading agent active status in the prediction

            if not self.config['eval_mode']:
                fig = self.visualize_scene(current_input, pred, 0, info["ca_duals"])
            else:
                fig = self.visualize_scene(current_input, pred, 0)
            self.figs_w_preds.append(fig)
            
            #Save canonical form
            if self.smpc.offline and self.t == 1: #run once
                # canon_prob = self.smpc._get_canon_form_mats() #Output is in dict
                canon_prob_fn = self.smpc._get_canon_form_fns()
                
                with open(f'/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_N{str(self.smpc.N)}_canon_form.pkl', 'wb') as f:
                    pickle.dump(canon_prob_fn, f)
                print(f'[Offline Mode] Canonical form saved')
            elif not self.smpc.offline and self.t == 1: #run once
                canon_prob_fn_precomputed = self.smpc._get_canon_form_fns_precomputed()
                with open(f'/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_N{str(self.smpc.N)}_canon_form_precomputed.pkl', 'wb') as f:
                    pickle.dump(canon_prob_fn_precomputed, f)
                print(f'[Eval Mode] Canonical form saved')   
            # pdb.set_trace()          
            # self.visualize_scene(current_input, pred, 0,info["ca_duals"],visualize=True) 
        else:
            print('No optimal solution found') 
            if leading_vehicle is not None and leading_vehicle_key not in preds_dict['tv_track_tokens']:
                preds_dict.update({'leading_vehicle':leading_agent, 'leading_vehicle_active':False}) #update the leading agent in the prediction
            fig = self.visualize_scene(current_input, preds_dict, 0)
            self.figs_w_preds.append(fig)
            self.scenario_type = scenario_type
            
            # self.visualize_scene(current_input, pred, 0,visualize=True)
            # self.visualize_scene(current_input, pred, 0,info["ca_duals"],visualize=True)
            # self.visualize_observations(ego_state, observations.tracked_objects.tracked_objects)
        
        #Update u_prev
        self.u_prev = sol['u_control'] if self.optimal else 0 #scalar
        self.u_opt = sol['u_opt'] #size N-1

        #Convert smpc solution to NuPlan Trajectory
        self._sol2ego_state(sol['nom_z'],ego_state)
        self.t += 1
        return InterpolatedTrajectory(self.ego_traj) #self.ego_traj is a list of EgoState
    
    def set_scenario_id(self, sc_id):
        self.scenario_id = sc_id

    def get_max_target_vehicle_speeds(self,planner_input):
        # Get the target vehicle speeds from the planner input
        ego_state, observations = planner_input.history.current_state
        target_vehicle_speeds = []
        radius = 50 #radius to consider the target vehicles
        for tracked_object in observations.tracked_objects.tracked_objects:
            #if the tracked object is within certain radius 
            if np.sqrt( (tracked_object.box.center.x - ego_state.center.x)**2 + (tracked_object.box.center.y - ego_state.center.y)**2 ) < radius:
                target_vehicle_speeds.append(tracked_object.velocity.magnitude())
        if len(target_vehicle_speeds) == 0:
            logger.warning("No target vehicles found in the observations.")
            return 1e6 #large number is fine since we use the min
        else:
            return max(target_vehicle_speeds)
        
    def agent_preds2array(self, preds):
        out = np.zeros((len(preds[0]),4,self.smpc.N)) #num_tv x 4 x N
        agent_params = np.zeros((len(preds[0]),2)) #agent l and w
        for t, agents_t_arr in enumerate(preds[:-1]): #exclude the last time step
            for i, agent in enumerate(agents_t_arr):
                if isinstance(agent,IDMAgent):
                    out[i,0,t] = agent.to_se2().x
                    out[i,1,t] = agent.to_se2().y
                    out[i,2,t] = agent.velocity
                    out[i,3,t] = agent.to_se2().heading

                    agent_params[i,0] = agent.length
                    agent_params[i,1] = agent.width
                else:
                    NotImplementedError
        return out, agent_params

    def agent_preds2array_wayformer(self, preds):
        pred = preds['preds']
        tv_params = preds['tv_params']
        tv_psi = preds['tv_psi']
        out = np.zeros((pred.shape[0],pred.shape[1],4,self.smpc.N)) #num_tv x num_mode x 4 x N
        agent_params = np.zeros((pred.shape[0],2)) #agent l and w
        for t in range(self.smpc.N-1): #exclude the last time step
            for i in range(self.smpc.N_TV):
                out[i,:,0,t] = pred[i,:,t,0]
                out[i,:,1,t] = pred[i,:,t,1]
                out[i,:,2,t] = 0 #wayformer doesn't predict velocities explicitly
                out[i,:,3,t] = tv_psi[i,:,t,0]

                agent_params[i,0] = tv_params[i][0]
                agent_params[i,1] = tv_params[i][1]
        return out, agent_params
    
    def get_ego_full_state(self):
        ego_traj_full_state = np.zeros((len(self.ego_traj)-1,4)) #[x,y,v,heading]
        for t in range(1,len(self.ego_traj)): #from planned ego_traj t|t-1 , ... , t+N-1|t-1
            ego_traj_full_state[t-1,0] = self.ego_traj[t].center.point.x
            ego_traj_full_state[t-1,1] = self.ego_traj[t].center.point.y
            ego_traj_full_state[t-1,2] = self.ego_traj[t].dynamic_car_state.center_velocity_2d.magnitude()
            ego_traj_full_state[t-1,3] = self.ego_traj[t].center.heading
        return ego_traj_full_state

    def get_observation(self, current_input, predictions):
        '''
        x0: ego's current states (x,y,v,heading)
        u_prev: previous control input (acceleration)
        o0: TV's current states w.r.t. ego's current states (x,y,v,heading)
        ttc?? some kind of graph encoding of the scene w.r.t. ego vehicle
        '''
        ego_state, observations = current_input.history.current_state
        vh_track_tokens = []
        for vh in observations.tracked_objects.tracked_objects:
            vh_track_tokens.append(vh.metadata.track_token)
        obs = np.zeros((1,4*self.config['num_tvs']+4+1))

        obs[:,:4] = np.array([ego_state.center.point.x,ego_state.center.point.y,ego_state.dynamic_car_state.center_velocity_2d.magnitude(),ego_state.center.heading])
        obs[:,4] = self.u_prev
        for i in range(self.config['num_tvs']):
            if self.config['prediction_method']=='idm':
                obs[:,5+4*i:5+4*(i+1)] = np.array([predictions[i].to_se2().x,predictions[i].to_se2().y,predictions[i].velocity,predictions[i].to_se2().heading]) if isinstance(predictions[i],IDMAgent) else np.array([predictions[i].center.x,predictions[i].center.y,predictions[i].velocity.magnitude(),predictions[i].center.heading]) #TV's current states
                obs[:,5+4*i:5+4*(i+1)] -= obs[:,:4] #relative to ego's current states
            else:
                ind = vh_track_tokens.index(predictions['tv_track_tokens'][i])
                obs[:,5+4*i:5+4*(i+1)] = np.array([observations.tracked_objects.tracked_objects[ind].box.center.x,observations.tracked_objects.tracked_objects[ind].box.center.y,observations.tracked_objects.tracked_objects[ind].velocity.magnitude(),observations.tracked_objects.tracked_objects[ind].box.center.heading]) #TV's current states 
            obs[:,5+4*i:5+4*(i+1)] -= obs[:,:4]
            # if self.is_mm_preds and isinstance(predictions[i],List):
            #     obs[:,5+4*self.config['num_tvs']+i] = 1 if len(predictions[i]) > 1 else 0
            # else:
            #     obs[:,5+4*self.config['num_tvs']+i] = 0
        return obs
    
    def check_preds(self,preds):
        for pred in preds:
            if len(pred) > 0:
                pass
            else:
                return False
        return True
            
    def red_light_leading_idm_agent(self,ego_state,observations,current_input):
        # RED LIGHT
        # Create occupancy map
        occupancy_map, unique_observations = self._construct_occupancy_map(ego_state, observations)
        ego_progress = self._ego_path_linestring.project(Point(*ego_state.center.point.array))
        ego_idm_state = IDMAgentState(progress=ego_progress, velocity=ego_state.dynamic_car_state.center_velocity_2d.x)
        # Traffic light handling
        traffic_light_data = current_input.traffic_light_data
        self._annotate_occupancy_map(traffic_light_data, occupancy_map)
        intersecting_agents = occupancy_map.intersects(self._get_expanded_ego_path(ego_state, ego_idm_state))
        # Check if there are agents intersecting the ego's baseline
        if intersecting_agents.size > 0:
            # Extract closest object
            intersecting_agents.insert(self._ego_token, ego_state.car_footprint.geometry)
            nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(
                self._ego_token
            )
            # Red light at intersection
            if self._red_light_token in nearest_id:
                print('RED LIGHT DETECTED'.center(50, '-'))
                return self._ego_path.get_state_at_progress(ego_progress+relative_distance)
        return None
    
    def leading_idm_agent(self,ego_state,observations,current_input):
        # Create occupancy map
        occupancy_map, unique_observations = self._construct_occupancy_map(ego_state, observations)
        ego_progress = self._ego_path_linestring.project(Point(*ego_state.center.point.array))
        ego_idm_state = IDMAgentState(progress=ego_progress, velocity=ego_state.dynamic_car_state.center_velocity_2d.x)
        # Traffic light handling
        traffic_light_data = current_input.traffic_light_data
        self._annotate_occupancy_map(traffic_light_data, occupancy_map)
        intersecting_agents = occupancy_map.intersects(self._get_expanded_ego_path(ego_state, ego_idm_state))
        # Check if there are agents intersecting the ego's baseline
        if intersecting_agents.size > 0:
            # Extract closest object
            intersecting_agents.insert(self._ego_token, ego_state.car_footprint.geometry)
            nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(
                self._ego_token
            )

            return {nearest_id: self._get_leading_idm_agent(ego_state, unique_observations[nearest_id], relative_distance)}
        
        return None
    
    def s2xy(self,s_arr):
        '''
        Convert s to x,y
        '''
        xy_list = []
        for s in s_arr:
            xy_list.append(np.array(self.ego_route(s)[:2]) + np.array([self.x0.center.point.x,self.x0.center.point.y]) )
        return xy_list

    def _sol2ego_state(self, sol, ego_state0):
        ego_idm_state = IDMAgentState(progress=sol[0,0], velocity=sol[1,0])
        vehicle_parameters = ego_state0.car_footprint.vehicle_parameters
        
        # Initialize planned trajectory with current state
        current_time_point = ego_state0.time_point
        projected_ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)
        ego_traj: List[EgoState] = [projected_ego_state]
        for t in range(1,self.config['N']+1):     
            ego_idm_state = IDMAgentState(progress=sol[0,t], velocity=sol[1,t])
            # Convert IDM state back to EgoState
            current_time_point += TimePoint(int(self._planned_trajectory_sample_interval * 1e6))
            ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)
            ego_traj.append(ego_state)
        self.ego_traj = ego_traj
    
    def visualize_scene(self, current_input, preds, t=0, ca_duals=[], visualize=False) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from shapely.geometry import Point

        ego_state, observations = current_input.history.current_state
        vh_track_tokens = [vh.metadata.track_token for vh in observations.tracked_objects.tracked_objects]
        ego_x, ego_y = ego_state.center.point.x, ego_state.center.point.y
        ego_length = ego_state.car_footprint.vehicle_parameters.length
        ego_width = ego_state.car_footprint.vehicle_parameters.width
        ego_heading = ego_state.center.heading

        fig = plt.figure()
        ax = plt.gca()
        
        # Draw the ego vehicle
        ego_rect = plt.Rectangle(
            (ego_x - ego_length/2, ego_y - ego_width/2),
            ego_length,
            ego_width,
            angle=ego_heading*180/np.pi,
            fill=True,
            color='green',
            rotation_point='center',
            label='Ego Vehicle'
        )
        ax.add_patch(ego_rect)
        self.visualize_road_boundaries(ax, ego_x, ego_y, self._map_api, search_radius=100)
        if 'leading_vehicle' in preds:
            ego_state0, _ = current_input.history.current_state
            vehicle_parameters = ego_state0.car_footprint.vehicle_parameters
            # Initialize planned trajectory with current state
            current_time_point = ego_state0.time_point
            projected_ego_state = self._idm_state_to_ego_state(preds['leading_vehicle'], current_time_point, vehicle_parameters)
            lead_vehicle_active = preds['leading_vehicle_active']
            # Draw ellipsoid for the leading agent
            leading_vehicle_ellipsoid = patches.Ellipse(
                (projected_ego_state.center.point.x, projected_ego_state.center.point.y),
                projected_ego_state.box.length,
                projected_ego_state.box.width,
                angle=projected_ego_state.center.heading*180/np.pi,
                fill=True,
                color='red' if lead_vehicle_active else 'yellow')
            


        # ---- Add Road Lanes Visualization using route roadblocks ----
        if hasattr(self, '_route_roadblocks'):
            ego_point = Point(ego_x, ego_y)
            lane_plotted = False
            for roadblock in self._route_roadblocks:
                for edge in roadblock.interior_edges:
                    # Check if the edge has a baseline path with a discrete_path attribute
                    if hasattr(edge, 'baseline_path') and hasattr(edge.baseline_path, 'discrete_path'):
                        # Optionally, filter lanes by distance to the ego vehicle
                        # Here, we assume discrete_path is a list of points with x, y attributes.
                        pts = edge.baseline_path.discrete_path
                        # For example, plot only if the first point is within 30 meters of the ego vehicle.
                        if pts and Point(pts[0].x, pts[0].y).distance(ego_point) < 200:
                            xs = [pt.x for pt in pts]
                            ys = [pt.y for pt in pts]
                            # Only add the label once
                            label = 'Lane' if not lane_plotted else None
                            plt.plot(xs, ys, color='gray', linestyle='-', linewidth=1, label=label)
                            lane_plotted = True
        else:
            print("No _route_roadblocks attribute available to extract lane geometry.")

        # ---- Add Traffic Light Lines Visualization ----
        try:
            traffic_light_data = current_input.traffic_light_data
            tl_plotted = False
            for tl in traffic_light_data:
                for stop_line in self._map_api._map_objects[SemanticMapLayer.LANE_CONNECTOR][str(tl.lane_connector_id)].stop_lines:
                    xs, ys = stop_line.polygon.exterior.xy
                    # xs, ys = self._map_api._map_objects[SemanticMapLayer.LANE_CONNECTOR][str(tl.lane_connector_id)].baseline_path.linestring.xy
                    color = 'green' if tl.status == TrafficLightStatusType.GREEN else 'red'
                    plt.plot(xs, ys, color=color, linestyle='--', linewidth=2, label=label)
                    tl_plotted = True
        except Exception as e:
            print("Could not retrieve traffic light geometry:", e)

        if ca_duals:
            for t_idx in range(self.smpc.N):
                if self.config['prediction_method']=='idm':
                    for j, agent in enumerate(preds[t_idx]):
                        if isinstance(agent, IDMAgent):
                            x, y = agent.to_se2().x, agent.to_se2().y 
                            length, width = agent.length, agent.width
                            heading = agent.to_se2().heading
                        else:
                            x, y = agent.center.x, agent.center.y
                            length, width = agent.box.length, agent.box.width
                            heading = agent.center.heading
                        if t_idx == 0:
                            rect = plt.Rectangle(
                                (x - length/2, y - width/2), length, width,
                                angle=heading*180/np.pi, fill=True, color='blue', rotation_point='center'
                            )
                            ax.add_patch(rect)
                        else:
                            active_ca_dual = (ca_duals[j][0][t_idx-1][0] > 1e-3)
                            color = 'red' if active_ca_dual else 'yellow'
                            ellipsoid = patches.Ellipse((x, y), length, width, angle=heading*180/np.pi, fill=True, color=color)
                            ax.add_patch(ellipsoid)
                else:
                    #Plot all vehicles in the observation track
                    for i, vh in enumerate(observations.tracked_objects.tracked_objects):
                        x, y = vh.center.x, vh.center.y
                        length, width = vh.box.length, vh.box.width
                        heading = vh.center.heading
                        rect = plt.Rectangle(
                            (x - length/2, y - width/2), length, width,
                            angle=heading*180/np.pi, fill=True, color='blue', rotation_point='center'
                        )
                        ax.add_patch(rect)
                    for j in range(self.config['num_tvs']):
                        if t_idx == 0:
                            pass #vehicles already plotted
                        else:
                            for n in range(self.config['num_modes']):
                                x,y = preds['preds'][j,n,t_idx,0], preds['preds'][j,n,t_idx,1]
                                length, width = preds['tv_params'][j][0], preds['tv_params'][j][1]
                                heading = preds['tv_psi'][j,n,t_idx,0]

                                active_ca_dual = (ca_duals[j][0][t_idx-1][0] > 1e-3)
                                color = 'red' if active_ca_dual else 'yellow'
                                ellipsoid = patches.Ellipse((x, y), length, width, angle=heading*180/np.pi, fill=True, color=color)
                                ax.add_patch(ellipsoid)
        else:
            if self.config['prediction_method']=='idm':
                for j, agent in enumerate(preds[t]):
                    if isinstance(agent, IDMAgent):
                        x, y = agent.to_se2().x, agent.to_se2().y 
                        length, width = agent.length, agent.width
                        heading = agent.to_se2().heading
                    else:
                        x, y = agent.center.x, agent.center.y
                        length, width = agent.box.length, agent.box.width
                        heading = agent.center.heading
                    rect = plt.Rectangle(
                        (x - length/2, y - width/2), length, width,
                        angle=heading*180/np.pi, fill=True, color='blue', rotation_point='center'
                    )
                    ax.add_patch(rect)
            else:
                for j, vh in enumerate(observations.tracked_objects.tracked_objects):
                    x, y = vh.center.x, vh.center.y
                    length, width = vh.box.length, vh.box.width
                    heading = vh.center.heading
                    rect = plt.Rectangle(
                        (x - length/2, y - width/2), length, width,
                        angle=heading*180/np.pi, fill=True, color='blue', rotation_point='center'
                    )
                    ax.add_patch(rect)
                for t_idx in range(self.smpc.N):
                    for j in range(self.config['num_tvs']):
                        if t_idx == 0:
                            pass #vehicles already plotted
                        else:
                            for n in range(self.config['num_modes']):
                                x,y = preds['preds'][j,n,t_idx,0], preds['preds'][j,n,t_idx,1]
                                length, width = preds['tv_params'][j][0], preds['tv_params'][j][1]
                                heading = preds['tv_psi'][j,n,t_idx,0]

                                color = 'yellow'
                                ellipsoid = patches.Ellipse((x, y), length, width, angle=heading*180/np.pi, fill=True, color=color)
                                ax.add_patch(ellipsoid)

        # Plot the planned trajectory, if available
        if hasattr(self, 'ego_traj'):
            for i, state in enumerate(self.ego_traj):
                if i ==0:
                    plt.plot(state.center.point.x, state.center.point.y, 'gs', markersize=1.5,label='Ego Planned Trajectory')
                else:
                    plt.plot(state.center.point.x, state.center.point.y, 'gs', markersize=1.5)
        
        plt.axis('equal')
        plt.legend(loc='upper right')
        # Set axis limits around the ego vehicle
        plt.xlim([ego_x-30, ego_x+30])
        plt.ylim([ego_y-30, ego_y+30])
        if visualize:
            plt.show()
        plt.close(fig)
        return fig
    
    def visualize_road_boundaries(self, ax, ego_x, ego_y, map_api, search_radius=100):
        # Create a point from the ego position.
        ego_point = Point(ego_x, ego_y)
        
        # Retrieve lanes near the ego vehicle.
        map_objects = map_api.get_proximal_map_objects(ego_point, search_radius, [SemanticMapLayer.LANE])
        lanes = map_objects.get(SemanticMapLayer.LANE, [])
        
        # Loop through the lanes and add their polygon as a patch.
        for lane in lanes:
            if hasattr(lane, 'polygon') and lane.polygon:
                xs, ys = lane.polygon.exterior.xy
                road_patch = patches.Polygon(
                    list(zip(xs, ys)),
                    closed=True,
                    facecolor='lightgray',  # Use light gray for the road.
                    edgecolor='none',
                    alpha=0.5  # Adjust transparency if desired.
                )
                ax.add_patch(road_patch)

    def _initialize_ego_path(self, ego_state: EgoState) -> None:
        """
        Initializes the ego path from the ground truth driven trajectory
        :param ego_state: The ego state at the start of the scenario.
        """
        route_plan, _ = self._breadth_first_search(ego_state)
        ego_speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        speed_limit = route_plan[0].speed_limit_mps or self._policy.target_velocity
        print(f"Route plan speed limit: {speed_limit} m/s, Ego speed: {ego_speed} m/s")
        self._policy.target_velocity = speed_limit if speed_limit > ego_speed else ego_speed
        discrete_path = []
        for edge in route_plan:
            discrete_path.extend(edge.baseline_path.discrete_path)
        self._ego_path = create_path_from_se2(discrete_path)
        self._ego_path_linestring = path_to_linestring(discrete_path)

    def _get_starting_edge(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Get the starting edge based on ego state. If a lane graph object does not contain the ego state then
        the closest one is taken instead.
        :param ego_state: Current ego state.
        :return: The starting LaneGraphEdgeMapObject.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert len(self._route_roadblocks) >= 2, "_route_roadblocks should have at least 2 elements!"

        starting_edge = None
        closest_distance = math.inf

        # Check for edges in about first and second roadblocks
        for edge in self._route_roadblocks[0].interior_edges + self._route_roadblocks[1].interior_edges:
            if edge.contains_point(ego_state.center):
                starting_edge = edge
                break

            # In case the ego does not start on a road block
            distance = edge.polygon.distance(ego_state.car_footprint.geometry)
            if distance < closest_distance:
                starting_edge = edge
                closest_distance = distance

        assert starting_edge, "Starting edge for IDM path planning could not be found!"
        return starting_edge

    def _breadth_first_search(self, ego_state: EgoState) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Performs iterative breath first search to find a route to the target roadblock.
        :param ego_state: Current ego state.
        :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end roadblock. If unsuccessful a longest route is given.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert (
            self._candidate_lane_edge_ids is not None
        ), "_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!"

        starting_edge = self._get_starting_edge(ego_state)
        graph_search = BreadthFirstSearch(starting_edge, self._candidate_lane_edge_ids)
        # Target depth needs to be offset by one if the starting edge belongs to the second roadblock in the list
        offset = 1 if starting_edge.get_roadblock_id() == self._route_roadblocks[1].id else 0
        route_plan, path_found = graph_search.search(self._route_roadblocks[-1], len(self._route_roadblocks[offset:]))

        if not path_found:
            logger.warning(
                "IDMPlanner could not find valid path to the target roadblock. Using longest route found instead"
            )

        return route_plan, path_found
    
    def visualize_observations(self,ego_state: EgoState, observations: list):
        """
        Visualize ego state and observations in a 2D top-down view.
        :param ego_state: The current state of the ego vehicle.
        :param observations: List of detected objects (DetectionTrack).
        """
        # Set up the plot
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from nuplan.common.actor_state.agent import Agent

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Plot ego vehicle
        ego_x, ego_y = ego_state.center.point.x, ego_state.center.point.y
        ego_heading = ego_state.center.heading  # In radians
        ego_length, ego_width = ego_state.car_footprint.vehicle_parameters.length, ego_state.car_footprint.vehicle_parameters.width
        ego_box = patches.Rectangle((ego_x - ego_length/2, ego_y - ego_width/2), ego_length, ego_width, 
                                    angle=np.degrees(ego_heading),
                                    edgecolor='green', facecolor='green', alpha=1)
        ax.add_patch(ego_box)

        # Plot each detection
        for obs in observations:
            if isinstance(obs, Agent):
                x, y = obs.box.center.x, obs.box.center.y
                width, length = obs.box.width, obs.box.length
                heading = obs.box.center.heading

                # Add rectangle for detected object
                det_box = patches.Rectangle((x - length/2, y - width/2), length, width, 
                                            angle=np.degrees(heading),
                                            edgecolor='blue', facecolor='blue', alpha=1)
                ax.add_patch(det_box)
        
        ax.set_xlim(ego_x - 30, ego_x + 30)
        ax.set_ylim(ego_y - 30, ego_y + 30)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.title("Ego and Observations Visualization")
        plt.show()

    def _callback_end_simulation(self, logname: str = None) -> None:
        """Callback to be executed at the end of the simulation."""
        #Delete SMPC instance for serialization
        #Store the observation, preds, dual_class, expert_action in a pickle form
        print('End of simulation')
        if self.t >= 5:
            try:
                print('Saving data...')
                if not self.config['eval_mode']:
                    save_dir = ''.join(self.config['save_dir'].split('.pkl')[:-1]) + '_' + self.config['prediction_method']+ '_' + self.config['collision_avoidance_method'] +'.pkl'
                    filepath = save_dir + '.gz'
                    if not os.path.exists(filepath): 
                        with gzip.open(filepath, 'wb') as f:
                            # pickle.dump({'optimal_duals': self.expert_action, 'observation':self.observation, 'dual_class':self.dual_class, 'preds': self.preds}, f, protocol=pickle.HIGHEST_PROTOCOL)
                            # if logname is not None:
                            pickle.dump({'log_iter':[self.log_iter],
                                        'logname': [logname], 
                                        'scenario_id': [self.scenario_id], 
                                        'ego_opt_sol':[self.ego_opt_sols_full_state], 
                                        'ego_cl_traj': [self.cl_ego_traj], 
                                        'ego_planned_trajs':[self.ego_planned_trajs],
                                        'iteration_data': [self.iteration_data], 
                                        'optimal_duals': [self.expert_action],
                                        'dual_class':[self.dual_class],
                                        'l1_active': [self.l1_active],
                                        'preds':[self.preds],
                                        'scenario_type': [self.scenario_type],
                                        'agent_params':[self.pred_agent_params],
                                        'smpc_params':[self.smpc_params]}, 
                                        f, protocol=pickle.HIGHEST_PROTOCOL)
                            # else:
                            #     pickle.dump({'optimal_duals': self.expert_action, 'observation':self.observation, 'dual_class':self.dual_class}, f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        with gzip.open(filepath, 'rb') as f:
                            data = pickle.load(f)
                        data['optimal_duals'].append(self.expert_action)
                        data['scenario_id'].append(self.scenario_id)
                        # data['observation'].append(self.observation)
                        data['iteration_data'].append(self.iteration_data)
                        data['dual_class'].append(self.dual_class)
                        data['ego_cl_traj'].append(self.cl_ego_traj)
                        data['ego_opt_sol'].append(self.ego_opt_sols_full_state)
                        data['ego_planned_trajs'].append(self.ego_planned_trajs) #[s,v]
                        data['preds'].append(self.preds)
                        data['l1_active'].append(self.l1_active)
                        data['agent_params'].append(self.pred_agent_params)
                        data['log_iter'].append(self.log_iter)
                        data['scenario_type'].append(self.scenario_type)
                        data['smpc_params'].append(self.smpc_params)
                        # if logname is not None:
                        data['logname'].append(logname)
                        with gzip.open(filepath, 'wb') as f:
                            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                            f.flush()
                    print('Data saved to', filepath)
                    # Save the list of figures as video
                    # Define the codec and create a VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_save_dir = '/'.join(self.config['video_save_dir'].split('/')[:-1]) +'_' + self.config['prediction_method']+ '_' + self.config['collision_avoidance_method']+'/'

                    # check if the directory exists, if not create it
                    if not os.path.exists(vid_save_dir):
                        os.makedirs(vid_save_dir) 
                        print(f"Created directory: {vid_save_dir}")

                    out = cv2.VideoWriter(vid_save_dir+self.scenario_id+'_N' + str(self.smpc.N) + '_' +str(self.log_iter)+'.mp4', fourcc, 20.0, (640, 480))

                    for fig in self.figs_w_preds:
                        # Convert the figure to an image
                        fig.canvas.draw()
                        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                        # Write the image to the video file
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out.write(img_bgr)

                    # Release the VideoWriter object
                    out.release()
                    print(f"Saved video for scenario {self.scenario_id} at {vid_save_dir} with filename: {self.scenario_id}_N{self.smpc.N}_{self.log_iter}.mp4")
                    # pdb.set_trace()
                else:
                    #in evaluation mode
                    save_dir = ''.join(self.config['save_dir'].split('.pkl')[:-1]) + '_' + self.config['prediction_method']+ '_' + self.config['collision_avoidance_method'] +'.pkl'
                    filepath = save_dir + '.gz'
                    if not os.path.exists(filepath): 
                        with gzip.open(filepath, 'wb') as f:
                            # pickle.dump({'optimal_duals': self.expert_action, 'observation':self.observation, 'dual_class':self.dual_class, 'preds': self.preds}, f, protocol=pickle.HIGHEST_PROTOCOL)
                            # if logname is not None:
                            pickle.dump({'log_iter':[self.log_iter],'logname': [logname], 'scenario_type':[self.scenario_type],'scenario_id': [self.scenario_id], 'ego_opt_sol':[self.ego_opt_sols_full_state], 'ego_cl_traj': [self.cl_ego_traj], 'ego_planned_trajs':[self.ego_planned_trajs],'iteration_data': [self.iteration_data],'preds':[self.preds],'agent_params':[self.pred_agent_params]}, f, protocol=pickle.HIGHEST_PROTOCOL)
                            # else:
                            #     pickle.dump({'optimal_duals': self.expert_action, 'observation':self.observation, 'dual_class':self.dual_class}, f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        with gzip.open(filepath, 'rb') as f:
                            data = pickle.load(f)
                        data['scenario_id'].append(self.scenario_id)
                        # data['observation'].append(self.observation)
                        data['iteration_data'].append(self.iteration_data)
                        data['ego_cl_traj'].append(self.cl_ego_traj)
                        data['ego_opt_sol'].append(self.ego_opt_sols_full_state)
                        data['ego_planned_trajs'].append(self.ego_planned_trajs) #[s,v]
                        data['preds'].append(self.preds)
                        data['agent_params'].append(self.pred_agent_params)
                        data['scenario_type'].append(self.scenario_type)
                        data['log_iter'].append(self.log_iter)
                        # if logname is not None:
                        data['logname'].append(logname)
                        with gzip.open(filepath, 'wb') as f:
                            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                            f.flush()
                        
                        # Save the list of figures as video
                        # Define the codebluec and create a VideoWriter object
                        # pdb.set_trace()
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        vid_save_dir = '/'.join(self.config['video_save_dir'].split('/')[:-1]) +'_' + self.config['prediction_method']+ '_' + self.config['collision_avoidance_method']+'/'
                        # check if the directory exists, if not create it
                        if not os.path.exists(vid_save_dir):
                            os.makedirs(vid_save_dir) 
                            print(f"Created directory: {vid_save_dir}")
                        out = cv2.VideoWriter(vid_save_dir+'eval/eval_'+self.scenario_id+'_N' + str(self.smpc.N) + '_' +str(self.log_iter)+'.mp4', fourcc, 20.0, (640, 480))

                        for fig in self.figs_w_preds:
                            # Convert the figure to an image
                            fig.canvas.draw()
                            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                            # Write the image to the video file
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            out.write(img_bgr)

                        # Release the VideoWriter object
                        out.release()
                        # pdb.set_trace() 
            except:
                pdb.set_trace()    
        else:
            pass
        
        #Delete for memory management and lightweight serialization
        del self.smpc
        self.expert_action = []
        self.observation = []
        self.dual_class = []
        self.iteration_data = []
        self.cl_ego_traj = []
        self.ego_planned_trajs = []
        self.preds = []
        self.pred_agent_params = []
        self.ego_opt_sols_full_state = []
        self.figs_w_preds = []
        self.smpc_params = []
        self.u_prev = 0.0 #initialze previous control input(acceleration) to 0 
        self.x_sol = None
        self.scenario_id = None
        self.t = 0
        self._initialized = False
