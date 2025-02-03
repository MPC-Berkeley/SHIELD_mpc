from __future__ import annotations

import logging
import time
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

            if isinstance(self.planner, SMPCPlanner): 
                #Get IDM predictions for planner
                time_controller_copy = copy.deepcopy(self.simulation._time_controller)
                if isinstance(self.simulation.setup.observations,IDMAgents):
                    if self.simulation._time_controller.get_iteration().index == 0:
                        ego_traj = list(self.simulation.scenario.get_expert_ego_trajectory())
                        preds = self.simulation._observations.get_idm_predictions(time_controller_copy.get_iteration(), time_controller_copy.next_iteration() if time_controller_copy.next_iteration() is not None else time_controller_copy.get_iteration(), ego_traj[:self.planner.config['N']+1], self.simulation._history_buffer, num_samples=self.planner.config['N'])
                    else:
                        preds = self.simulation._observations.get_idm_predictions(time_controller_copy.get_iteration(), time_controller_copy.next_iteration() if time_controller_copy.next_iteration() is not None else time_controller_copy.get_iteration(), self.planner.get_x_ego(self.simulation._history_buffer), self.simulation._history_buffer, num_samples=self.planner.config['N'])
                    tv_paths_se2 = None
                else:
                    preds = self.simulation.scenario._get_log_predictions(time_controller_copy.get_iteration(), num_samples=self.planner.config['N'])
                    tv_paths_se2 = self.simulation._scenario._get_agent_paths_from_log()
                # Plan path based on all planner's inputs
                # #TODO: tv_paths_se2 is not used in the planner
                # tv_paths_se2 = None
                trajectory = self.planner.compute_trajectory(planner_input,preds,tv_paths_se2)
            else:
                preds = []
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
