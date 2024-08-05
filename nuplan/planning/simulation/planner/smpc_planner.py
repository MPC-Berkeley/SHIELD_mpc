import logging
import math
from typing import List, Tuple
import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring
from nuplan.planning.simulation.planner.abstract_idm_planner import AbstractIDMPlanner
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from typing import Optional
import yaml
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.smpc import SMPC
from nuplan.planning.simulation.planner.smpc_utils import flatten
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
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        super(SMPCPlanner, self).__init__(
            5.0, # (Not used)
            2.0, #min_gap_to_lead_agent (Not used)
            2.0, #headway time (Not used)
            self.config['a_max'],
            self.config['a_min'],
            self.config['N'],
            self.config['dt'],
            50, #Occupancy_map_radius (Not used)
        )

        self.ev_noise_std = ev_noise_std
        self.tv_noise_std = tv_noise_std


        self._initialized = False

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._map_api = initialization.map_api
        self._initialize_route_plan(initialization.route_roadblock_ids)
        self._initialized = False

    def compute_planner_trajectory(self, current_input: PlannerInput, preds: Optional[List]) -> AbstractTrajectory:
        """Inherited, see superclass."""
        # Ego current state
        ego_state, observations = current_input.history.current_state

        if not self._initialized:
            self._initialize_ego_path(ego_state)
            self._ego_path #ego route
            
            N = self.config['N']
            dt = self.config['dt']

            # Lineraized Ego Vehicle Dynamics
            A = np.array([[1., dt], [0., 1.]])
            B = np.array([0.5*dt**2,dt])

            # Initialize the SMPC
            self.smpc = SMPC(ev=(A,B),
                    N            =  N,
                    V_MIN        = self.config['v_min'],       #Speed, acceleration constraints
                    V_MAX        = self.config['v_max'], 
                    A_MIN        = self.config['a_min'],
                    A_MAX        =  self.config['a_max'],
                    TIGHTENING   =  2.4, #2.6
                    EV_NOISE_STD    =  self.ev_noise_std,
                    TV_NOISE_STD    = self.tv_noise_std,
                    Q = 1.,       # cost for measuring progress: -Q*s_{t+1}. #was 1.
                    R = 1.,       # cost for penalizing large input rate: (u_{t+1}-u_t).T@R@(u_{t+1}-u_t) #was 1.5
                    offline_mode=True,
                    solver="ipopt",
                    open_loop = False,
                    eval_mode = False,
                    preds=preds)
        
            self._initialized = True

        # Update the SMPC parameters
        self.smpc.update(update_dict)
        
        # Solve the SMPC
        sol = self.smpc.solve()
        info = {}
        if sol['optimal']:
            # Get the optimal DUALS
            info.update({"l1_duals":sol["l1_duals"], "ca_duals":sol["ca_duals"]})
            dual_class = 0
            l1_duals_vec = np.fromiter(flatten(info["l1_duals"]),float)
            ca_duals_vec = np.fromiter(flatten(info["ca_duals"]),float)
            l1_dual_active = (1-int(np.all(l1_duals_vec<(self.smpc.l1_lmbd-1e-3)*np.ones(l1_duals_vec.shape[0])))) or (1-int(np.all(l1_duals_vec>1e-3*np.ones(l1_duals_vec.shape[0]))))
            ca_duals_active = np.sum(ca_duals_vec>1e-3*np.ones(ca_duals_vec.shape[0]))/ca_duals_vec.shape[0]

            if l1_dual_active == 1:
                if ca_duals_active > 0.05 :
                    dual_class = 3
                else:
                    dual_class = 1
            elif ca_duals_active > 0.05 :
                dual_class = 2

        #Convert smpc solution to NuPlan Trajectory
        planned_trajectory: List[EgoState]
        for i in range(self.config['N']):
            #TODO: Convert SMPC solution to EgoState
            ego_state = EgoState()
            planned_trajectory.append(ego_state)
        return InterpolatedTrajectory(planned_trajectory)

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
