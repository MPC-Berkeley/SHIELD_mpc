import logging
import math
from typing import List, Tuple
import numpy as np
from shapely.geometry import LineString, Point, Polygon
import pdb 
import datetime
import pickle
import gzip
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
            5.0, # (Not used)
            2.0, #min_gap_to_lead_agent (Not used)
            2.0, #headway time (Not used)
            self.config['a_max'],
            self.config['a_min'],
            self.config['N'],
            self.config['dt'],
            100, #Occupancy_map_radius (Not used)
        )

        self.ev_noise_std = ev_noise_std
        self.tv_noise_std = tv_noise_std

        self.u_prev = 0.0 #initialze previous control input(acceleration) to 0 
        self.x_sol = None
        
        self.dual_class = []
        self.expert_action = []
        self.observation = []
        self.preds = []
        self.cl_ego_traj = []

        self._initialized = False

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
    
    def get_update_dict(self,current_input: PlannerInput, preds: List) -> dict:
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
                                                                                                u_opt=self.u_opt if hasattr(self, 'u_opt') and self.optimal else None,
                                                                                                ego_traj=self.ego_traj if hasattr(self, 'ego_traj') else None)
       
        update_dict =   {'x0': x0,
                         'o0': [np.array([[agent.progress],[agent.velocity]]) for agent in preds[0]],
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

    def compute_planner_trajectory(self, current_input: PlannerInput, preds: Optional[List]) -> AbstractTrajectory:
        """Inherited, see superclass."""
        # Ego current state
        ego_state, observations = current_input.history.current_state
        # print('compute_planner_trajectory:', ego_state.center.point.x, ego_state.center.point.y, ego_state.dynamic_car_state.center_velocity_2d.magnitude())
        if not self._initialized:
            self._initialize_ego_path(ego_state)
            
            N = self.config['N']
            dt = self.config['dt']

            # Lineraized Ego Vehicle Dynamics   
            A = np.array([[1., dt], [0., 1.]])
            B = np.array([0.5*dt**2,dt])

            # Ego route and droute
            s_arr = [point.progress for point in self._ego_path.get_sampled_path()]
            x_arr = [point.x for point in self._ego_path.get_sampled_path()]
            y_arr = [point.y for point in self._ego_path.get_sampled_path()]
            psi_arr = [point.heading for point in self._ego_path.get_sampled_path()]
            v_arr = [0 for _ in self._ego_path.get_sampled_path()]

            self.ego_route = make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr)
            self.ego_droute = make_jac_fun(self.ego_route)

            # Initialize the SMPC
            self.smpc = SMPC(ev=(A,B),
                    N            =  N,
                    V_MIN        = self.config['v_min'],       #Speed, acceleration constraints
                    V_MAX        = self.config['v_max'], 
                    A_MIN        = self.config['a_min'],
                    A_MAX        =  self.config['a_max'],
                    TIGHTENING   =  2.6, #2.4
                    EV_NOISE_STD    =  self.ev_noise_std,
                    TV_NOISE_STD    = self.tv_noise_std,
                    Q = 1.,       # cost for measuring progress: -Q*s_{t+1}. #was 1.
                    R = 1.,       # cost for penalizing large input rate: (u_{t+1}-u_t).T@R@(u_{t+1}-u_t) #was 1.5
                    offline_mode=True,
                    solver="ipopt",
                    open_loop = False,
                    eval_mode = False,
                    route = self.ego_route,
                    preds=filter_preds(preds,self.config['num_tvs'],ego_state))
            self._initialized = True     

        # Update the SMPC parameters
        update_dict = self.get_update_dict(current_input, filter_preds(preds,self.config['num_tvs'],ego_state))
        update_dict.update({'red_light': self.red_light_leading_idm_agent(ego_state,observations,current_input)})
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
            self.observation.append(observations)
            self.preds.append(filter_preds(preds,self.config['num_tvs'],ego_state))
            self.cl_ego_traj.append(ego_state)
            # print(ca_duals_vec)
            print(dual_class)
            print(ca_duals_active,l1_dual_active)
        else:
            print('No optimal solution found') 
            pdb.set_trace()
            self.visualize_scene(current_input, preds, 0)
            # self.visualize_observations(ego_state, observations.tracked_objects.tracked_objects)
             
        #Update u_prev
        self.u_prev = sol['u_control'] if self.optimal else 0 #scalar
        self.u_opt = sol['u_opt'] #size N-1

        #Convert smpc solution to NuPlan Trajectory
        self._sol2ego_state(sol['nom_z'],ego_state)
        return InterpolatedTrajectory(self.ego_traj) #self.ego_traj is a list of EgoState

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
                return self._get_red_light_leading_idm_state(relative_distance)
        return None
    
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
    
    def visualize_scene(self, current_input, preds, t=0) -> None:
        ego_state, observations = current_input.history.current_state
        ego_x, ego_y = ego_state.center.point.x,ego_state.center.point.y
        ego_length, ego_width = ego_state.car_footprint.vehicle_parameters.length, ego_state.car_footprint.vehicle_parameters.width
        ego_heading = ego_state.center.heading

        import matplotlib.pyplot as plt
        plt.figure()
        #draw ego as a rectangle
        ego_rect = plt.Rectangle((ego_x-ego_length/2,ego_y-ego_width/2),ego_length,ego_width,angle=ego_heading*180/np.pi,fill=True,color='green',rotation_point='center')

        plt.gca().add_patch(ego_rect)
        plt.xlim([ego_x-30,ego_x+30])
        plt.ylim([ego_y-30,ego_y+30])

        # draw target vehicles
        for j, agent in enumerate(preds[t]):
            x, y = agent.to_se2().x, agent.to_se2().y
            length, width = agent.length, agent.width
            heading = agent.to_se2().heading
            # rect = plt.Rectangle((x,y),length,width,angle=heading*180/np.pi,fill=True,color='red') #center it to (x,y)
            #rectangle with center x,y
            rect = plt.Rectangle((x-length/2,y-width/2),length,width,angle=heading*180/np.pi,fill=True,color='red',rotation_point='center')

            plt.gca().add_patch(rect)
        
        #Plot planned traj
        if hasattr(self, '_ego_path'):
            for point in self._ego_path.get_sampled_path():
                plt.plot(point.x,point.y,'gs',markersize=3)
                
        plt.axis('equal')
        plt.show()

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
        filepath = self.config['save_dir'] + '.gz'
        if not os.path.exists(filepath): 
            with gzip.open(filepath, 'wb') as f:
                # pickle.dump({'optimal_duals': self.expert_action, 'observation':self.observation, 'dual_class':self.dual_class, 'preds': self.preds}, f, protocol=pickle.HIGHEST_PROTOCOL)
                # if logname is not None:
                pickle.dump({'logname': [logname], 'ego_states': self.cl_ego_traj, 'optimal_duals': self.expert_action, 'observation':self.observation, 'dual_class':self.dual_class}, f, protocol=pickle.HIGHEST_PROTOCOL)
                # else:
                #     pickle.dump({'optimal_duals': self.expert_action, 'observation':self.observation, 'dual_class':self.dual_class}, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
            data['optimal_duals'].extend(self.expert_action)
            data['observation'].extend(self.observation)
            data['dual_class'].extend(self.dual_class)
            data['ego_states'].extend(self.cl_ego_traj)
            # data['preds'].extend(self.preds)
            # if logname is not None:
            data['logname'].extend([logname])
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        #Delete for memory management and lightweight serialization
        del self.smpc
        del self.expert_action
        del self.observation
        del self.dual_class
        
        self._initialized = False
