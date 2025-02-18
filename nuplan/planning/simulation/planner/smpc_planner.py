import logging
import math
from typing import List, Tuple, Dict
import numpy as np
from shapely.geometry import LineString, Point, Polygon
import pdb 
import datetime, cv2
import pickle
import gzip
import copy
import os
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring
from nuplan.planning.simulation.planner.abstract_idm_planner import AbstractIDMPlanner
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
        # with open("/home/hansung/L4SMPC_nuplan/nuplan/planning/simulation/planner/smpc_config.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        super(SMPCPlanner, self).__init__(
            target_velocity=20.0, # (Not used)
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
        
        self._initialized = False
        self.t = 0

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
    
    def get_update_dict(self,current_input: PlannerInput, preds: List, tv_paths_se2: Dict) -> dict:
        ego_state, observations = current_input.history.current_state
        routes = [self.ego_route]
        droutes = [self.ego_droute]
        params = {'dt': self.config['dt'], 'N': self.config['N'],'N_TV': self.config['num_tvs']}
        ego_progress = self._ego_path_linestring.project(Point(*ego_state.center.point.array))
        x0 = np.array([[ego_progress],[ego_state.dynamic_car_state.center_velocity_2d.magnitude()]])
        z_lin, x_glob, dpos, o_glob, u_tvs, routes, droutes, Qs, tv_psi, tv_params = get_preds(current_input,
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
        #Assume is_mm_preds is True
        update_dict =   {'x0': x0,
                         'o0': [np.array([[agent.progress],[agent.velocity]]) for agent in preds[0]] if isinstance(preds[0][0],IDMAgent) else [np.array([[path_to_linestring(tv_paths_se2[agent.metadata.track_token]).project(Point(*agent.center.point.array))],[agent.velocity.magnitude()]]) for agent in preds[0]],
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

    def compute_planner_trajectory(self, current_input: PlannerInput, preds: Optional[List],tv_paths_se2: Optional[Dict]=None) -> AbstractTrajectory:
        """Inherited, see superclass."""
        # Ego current state
        ego_state, observations = current_input.history.current_state
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

            # Initialize the SMPC
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
                    offline_mode=True,
                    solver="ipopt",
                    open_loop = False,
                    eval_mode = False,
                    is_mm_preds=self.config['is_mm_preds'],
                    route = self.ego_route,
                    preds=filter_preds(preds,self.config['num_tvs'],ego_state))
            self._initialized = True     

        # Update the SMPC parameters
        if not (self.config['eval_mode_category'] == 0):
            update_dict = self.get_update_dict(current_input, filter_preds(preds,self.config['num_tvs'],ego_state),tv_paths_se2)
        else:
            mm_preds = self.mm_predictor.predict(filter_preds(preds,self.config['num_tvs'],ego_state),ego_state,is_mm_preds=self.config['is_mm_preds'])
            update_dict = self.get_update_dict(current_input, mm_preds,tv_paths_se2)
        update_dict.update({'ego_sim_initial_state':self.x0,'red_light': self.red_light_leading_idm_agent(ego_state,observations,current_input)})
        self.prev_update_dict = update_dict
        self.smpc.update(update_dict) 
        # Solve the SMPC
        sol = self.smpc.solve()
        self.optimal = sol['optimal']
        info = {}
        if self.optimal:
            # Get the optimal DUALS
            info.update({"l1_duals":sol["l1_duals"], "ca_duals":sol["ca_duals"]})
            dual_class = 0
            l1_duals_vec = np.fromiter(flatten(info["l1_duals"]),float)
            ca_duals_vec = np.fromiter(flatten(info["ca_duals"]),float)
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
            if not self.check_preds(preds):
                print('Empty preds detected')
                raise ValueError
            if (self.config['eval_mode_category']==0):
                pred = mm_preds
                preds2save, params2save = self.agent_preds2array(pred)
                self.pred_agent_params.append(params2save)
                self.preds.append(preds2save)
            else:
                pred = filter_preds(preds,self.config['num_tvs'],ego_state)
                # self.preds.append(filter_preds(preds,self.config['num_tvs'],ego_state))
                preds2save, params2save = self.agent_preds2array(pred)
                self.pred_agent_params.append(params2save)
                self.preds.append(preds2save)
            self.observation.append(self.get_observation(ego_state,pred[0])) 
            self.iteration_data.append(observations)
            self.cl_ego_traj.append(ego_state)
            self.ego_planned_trajs.append(self.s2xy(sol['nom_z'][0,1:]))
            self.ego_opt_sols_full_state.append(self.get_ego_full_state())
            
            print(ca_duals_vec)
            print(dual_class)
            print(ca_duals_active,l1_dual_active)
            fig = self.visualize_scene(current_input, pred, 0, info["ca_duals"])
            self.figs_w_preds.append(fig)
        else:
            print('No optimal solution found') 
            # pdb.set_trace()
            # self.visualize_scene(current_input, preds, 0,visualize=True)
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

    def get_ego_full_state(self):
        ego_traj_full_state = np.zeros((len(self.ego_traj)-1,4)) #[x,y,v,heading]
        for t in range(1,len(self.ego_traj)): #from planned ego_traj t|t-1 , ... , t+N-1|t-1
            ego_traj_full_state[t-1,0] = self.ego_traj[t].center.point.x
            ego_traj_full_state[t-1,1] = self.ego_traj[t].center.point.y
            ego_traj_full_state[t-1,2] = self.ego_traj[t].dynamic_car_state.center_velocity_2d.magnitude()
            ego_traj_full_state[t-1,3] = self.ego_traj[t].center.heading
        return ego_traj_full_state

    def get_observation(self,ego_state, predictions):
        '''
        x0: ego's current states (x,y,v,heading)
        u_prev: previous control input (acceleration)
        o0: TV's current states w.r.t. ego's current states (x,y,v,heading)
        ttc?? some kind of graph encoding of the scene w.r.t. ego vehicle
        '''
        obs = np.zeros((1,4*self.config['num_tvs']+4+1))
        obs[:,:4] = np.array([ego_state.center.point.x,ego_state.center.point.y,ego_state.dynamic_car_state.center_velocity_2d.magnitude(),ego_state.center.heading])
        obs[:,4] = self.u_prev
        for i in range(self.config['num_tvs']):
            obs[:,5+4*i:5+4*(i+1)] = np.array([predictions[i].to_se2().x,predictions[i].to_se2().y,predictions[i].velocity,predictions[i].to_se2().heading]) if isinstance(predictions[i],IDMAgent) else np.array([predictions[i].center.x,predictions[i].center.y,predictions[i].velocity.magnitude(),predictions[i].center.heading]) #TV's current states
            obs[:,5+4*i:5+4*(i+1)] -= obs[:,:4] #relative to ego's current states
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
    
    def s2xy(self,s_arr):
        '''
        Convert s to x,y
        '''
        xy_list = []
        for s in s_arr:
            xy_list.append(np.array(self.ego_route(s)[:2]) + np.array([self.x0.center.point.x,self.x0.center.point.y]) )
        return xy_list

    def _sol2ego_state(self, sol,ego_state0):
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

        # ---- Add Traffic Light Lines Visualization ----
        try:
            traffic_light_data = current_input.traffic_light_data
            tl_plotted = False
            for tl in traffic_light_data:
                # Adjust the attribute name as necessary—here we check for a 'polygon' attribute.
                if hasattr(tl, 'polygon'):
                    xs, ys = tl.polygon.exterior.xy
                    label = 'Traffic Light' if not tl_plotted else None
                    color = 'green' if tl.status == TrafficLightStatusType.GREEN else 'red'
                    plt.plot(xs, ys, color=color, linestyle='-', linewidth=2, label=label)
                    tl_plotted = True
                # Alternatively, if there is a list of line segments:
                elif hasattr(tl, 'lines'):
                    for line in tl.lines:
                        xs, ys = line.xy
                        color = 'green' if tl.status == TrafficLightStatusType.GREEN else 'red'
                        plt.plot(xs, ys, color=color, linestyle='-', linewidth=2, label='Traffic Light' if not tl_plotted else None)
                        tl_plotted = True
        except Exception as e:
            print("Could not retrieve traffic light geometry:", e)
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
                            plt.plot(xs, ys, color='blue', linestyle='--', linewidth=1, label=label)
                            lane_plotted = True
        else:
            print("No _route_roadblocks attribute available to extract lane geometry.")

        # ---- Add Filled Road Boundary Visualization using route roadblocks ----
        if hasattr(self, '_all_roadblocks'):
            boundary_plotted = False
            ax = plt.gca()  # Get current axis
            for roadblock in self._all_roadblocks:
                # Check if the roadblock has a polygon attribute representing its boundary.
                if hasattr(roadblock, 'polygon'):
                    xs, ys = roadblock.polygon.exterior.xy
                    polygon_points = list(zip(xs, ys))
                    label = 'Road Boundary' if not boundary_plotted else None
                    # Create a filled polygon patch with no edge accent by setting edgecolor to 'none'
                    road_patch = patches.Polygon(
                        polygon_points,
                        closed=True,
                        fill=True,
                        facecolor='gray',
                        edgecolor='none',
                        alpha=0.3,
                        label=label
                    )
                    ax.add_patch(road_patch)
                    boundary_plotted = True
                else:
                    print("Roadblock does not have a polygon attribute.")
        else:
            print("No _all_roadblocks attribute available to extract road boundaries.")
        # ---- Existing Visualization for Target Vehicles ----
        if ca_duals:
            for t_idx in range(self.smpc.N):
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
                        if j == 0:
                            rect = plt.Rectangle(
                                (x - length/2, y - width/2), length, width,
                                angle=heading*180/np.pi, fill=True, color='red', rotation_point='center', label='TV'
                            )
                        else:
                            rect = plt.Rectangle(
                                (x - length/2, y - width/2), length, width,
                                angle=heading*180/np.pi, fill=True, color='red', rotation_point='center'
                            )
                        ax.add_patch(rect)
                    else:
                        active_ca_dual = (ca_duals[j][0][t_idx-1][0] > 1e-3)
                        color = 'yellow' if active_ca_dual else 'red'
                        alpha = 1 if active_ca_dual else 0.3
                        ellipsoid = patches.Ellipse((x, y), length, width, angle=heading*180/np.pi, fill=True, color=color, alpha=alpha)
                        ax.add_patch(ellipsoid)
        else:
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
                    angle=heading*180/np.pi, fill=True, color='red', rotation_point='center'
                )
                ax.add_patch(rect)
        
        # Plot the planned trajectory, if available
        if hasattr(self, 'ego_traj'):
            for i, state in enumerate(self.ego_traj):
                if i ==0:
                    plt.plot(state.center.point.x, state.center.point.y, 'gs', markersize=1.5,label='Ego Planned Trajectory')
                else:
                    plt.plot(state.center.point.x, state.center.point.y, 'gs', markersize=1.5)

        
        plt.axis('equal')
        plt.legend()
        # Set axis limits around the ego vehicle
        plt.xlim([ego_x-30, ego_x+30])
        plt.ylim([ego_y-30, ego_y+30])
        if visualize:
            plt.show()
        plt.close(fig)
        return fig


    def _initialize_ego_path(self, ego_state: EgoState) -> None:
        """
        Initializes the ego path from the ground truth driven trajectory
        :param ego_state: The ego state at the start of the scenario.
        """
        route_plan, _ = self._breadth_first_search(ego_state)
        ego_speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        speed_limit = route_plan[0].speed_limit_mps or self._policy.target_velocity
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
                det_box = patches.Rectangle((x - length / 2, y - width / 2), length, width, 
                                            angle=np.degrees(heading),
                                            edgecolor='red', facecolor='red', alpha=1)
                ax.add_patch(det_box)
        
        ax.set_xlim(ego_x - 30, ego_x + 30)
        ax.set_ylim(ego_y - 30, ego_y + 30)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.title("Ego and Observations Visualization")
        plt.show()

    def _callback_end_simulation(self,logname: str = None) -> None:
        """Callback to be executed at the end of the simulation."""
        #Delete SMPC instance for serialization
        #Store the observation, preds, dual_class, expert_action in a pickle form
        print('End of simulation. Saving data...')
        filepath = self.config['save_dir'] + '.gz'
        if not os.path.exists(filepath): 
            with gzip.open(filepath, 'wb') as f:
                # pickle.dump({'optimal_duals': self.expert_action, 'observation':self.observation, 'dual_class':self.dual_class, 'preds': self.preds}, f, protocol=pickle.HIGHEST_PROTOCOL)
                # if logname is not None:
                pickle.dump({'log_iter':[self.log_iter],'logname': [logname], 'scenario_id': [self.scenario_id], 'ego_opt_sol':[self.ego_opt_sols_full_state], 'ego_cl_traj': [self.cl_ego_traj], 'ego_planned_trajs':[self.ego_planned_trajs],'iteration_data': [self.iteration_data], 'optimal_duals': [self.expert_action], 'dual_class':[self.dual_class],'preds':[self.preds],'agent_params':[self.pred_agent_params]}, f, protocol=pickle.HIGHEST_PROTOCOL)
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
            data['agent_params'].append(self.pred_agent_params)
            data['log_iter'].append(self.log_iter)
            # if logname is not None:
            data['logname'].append(logname)
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save the list of figures as video
            # Define the codec and create a VideoWriter object
            # pdb.set_trace()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.config['video_save_dir']+self.scenario_id+'_'+str(self.log_iter)+'.mp4', fourcc, 20.0, (640, 480))

            for fig in self.figs_w_preds:
                # Convert the figure to an image
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # Write the image to the video file
                out.write(img)

            # Release the VideoWriter object
            out.release()
            # pdb.set_trace()
        print('end of simulation')

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
        self.u_prev = 0.0 #initialze previous control input(acceleration) to 0 
        self.x_sol = None
        self.scenario_id = None
        self.t = 0
        self._initialized = False
