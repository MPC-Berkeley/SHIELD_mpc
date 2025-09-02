"""A
Some miscellaneous utility functions

Functions to edit:
    1. sample_trajectory
"""
import numpy as np
import time
from typing import List, Union, Dict
from torch.autograd import Variable
from collections.abc import Iterable
import numpy as np
from shapely.geometry import LineString, Point, Polygon
import torch as th
import pdb
import time
import os
import copy
import casadi as ca
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent
from nuplan.common.actor_state.agent import Agent
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring


def make_ca_fun(s, x, y, psi, v):
    x_ca= ca.interpolant("f2gx", "linear", [s], x)
    y_ca= ca.interpolant("f2gy", "linear", [s], y)
    psi_ca= ca.interpolant("f2gpsi", "linear", [s], psi)
    v_ca= ca.interpolant("f2gv", "linear", [s], v)
    s_sym=ca.MX.sym("s",1)

    glob_fun=ca.Function("fx",[s_sym], [ca.vertcat(x_ca(s_sym), y_ca(s_sym), psi_ca(s_sym), v_ca(s_sym))])
    return glob_fun

def make_jac_fun(pos_fun):
    s_sym=ca.MX.sym("s",1)
    pos_jac=ca.jacobian(pos_fun(s_sym), s_sym)
    return ca.Function("pos_jac",[s_sym], [pos_jac])      

def get_preds(current_input, preds_list: Union[List[IDMAgent],List[Agent],np.ndarray], x0, params, routes, droutes, simulation_t: int, u_opt = None, ego_traj = None,ego_p0=None,tv_paths_se2=None,dt=0.1,is_mm_preds=False,ego_sim_init_state=None):
    '''
    Getting EV predictions from previous MPC solution.
    This is used for linearizing the collision avoidance constraints
    '''
    try:
        assert ego_p0 is not None, 'Ego initial position must be provided'
        ego_state, observations = current_input.history.current_state
        observation_tokens = []
        for vh in observations.tracked_objects.tracked_objects:
            observation_tokens.append(vh.metadata.track_token)
        #Convert IDM predictions to global coordinates
        o_glob = [np.zeros((2,params['N']+1)) for _ in range(params['N_TV'])]
        o = [np.zeros((2,params['N']+1)) for _ in range(params['N_TV'])]
        tv_psi = [np.zeros((1,params['N']+1)) for _ in range(params['N_TV'])]
        u_tvs=[np.zeros((1,params['N'])) for _ in range(params['N_TV'])]
        tv_lengths = []
        tv_widths = []
        agent_paths = []
        config = params['config']
        if config['prediction_method'] == 'idm':
            for t, agents in enumerate(preds_list):
                for j, agent in enumerate(agents):
                    if isinstance(agent,List):
                        agent = agent[0] #use first mode
                    o_glob[j][:,t] = np.array([agent.to_se2().x - ego_sim_init_state.center.point.x,agent.to_se2().y-ego_sim_init_state.center.point.y]) if isinstance(agent,IDMAgent) else np.array([agent.center.x-ego_sim_init_state.center.point.x,agent.center.y-ego_sim_init_state.center.point.y])  #x,y
                    o[j][:,t] = np.array([agent.progress,agent.velocity]) if isinstance(agent,IDMAgent) else np.array([path_to_linestring(tv_paths_se2[agent.metadata.track_token]).project(Point(*agent.center.point.array)),agent.velocity.magnitude()]) #s,v
                    tv_psi[j][:,t] = agent.to_se2().heading if isinstance(agent,IDMAgent) else agent.center.heading
                    if t < params['N']:
                        try:
                            next_time_agent = preds_list[t+1][j][0] if is_mm_preds else preds_list[t+1][j]
                            u_tvs[j][:,t] = next_time_agent._u_prev if isinstance(agent,IDMAgent) else (next_time_agent.velocity.magnitude()-agent.velocity.magnitude())/dt #u
                        except:
                            print('error: u_tvs in get_preds()')
                            # pdb.set_trace()
                    if t == 0:
                        agent_paths.append(agent._path) if isinstance(agent,IDMAgent) else agent_paths.append(create_path_from_se2(tv_paths_se2[agent.metadata.track_token]))
                        tv_lengths.append(agent.length) if isinstance(agent,IDMAgent) else tv_lengths.append(agent.box.length)
                        tv_widths.append(agent.width) if isinstance(agent,IDMAgent) else tv_widths.append(agent.box.width)
                        assert agent.length if isinstance(agent,IDMAgent) else agent.box.length > 0 and agent.width if isinstance(agent,IDMAgent) else agent.box.width > 0, 'TV length and width must be greater than 0'
        else: #wayformer
            assert params['N'] < preds_list.shape[2], 'N must be less than the number of time steps in the prediction list'
            for t in range(params['N']):
                for j in range(preds_list.shape[0]):
                    o_glob[j][:,t+1] = np.array([preds_list[j,0,t,0] - ego_sim_init_state.center.point.x, preds_list[j,0,t,1]-ego_sim_init_state.center.point.y])
                    query_pt = Point(preds_list[j,0,t,0], preds_list[j,0,t,1])
                    if len(tv_paths_se2[params['tv_track_tokens'][j]]) > 1:
                        o[j][:,t+1] = np.array([path_to_linestring(tv_paths_se2[params['tv_track_tokens'][j]]).project(query_pt),0])
                    else:
                        o[j][:,t+1] = o[j][:,t]
                    tv_psi[j][:,t+1] = params['tv_psi'][j,0,t,0]
                    if t == 0:
                        ind = observation_tokens.index(params['tv_track_tokens'][j])                    
                        o_glob[j][:,t] = np.array([observations.tracked_objects.tracked_objects[ind].box.center.x - ego_sim_init_state.center.point.x, observations.tracked_objects.tracked_objects[ind].box.center.y - ego_sim_init_state.center.point.y])
                        query_pt = Point(observations.tracked_objects.tracked_objects[ind].box.center.x, observations.tracked_objects.tracked_objects[ind].box.center.y)
                        if len(tv_paths_se2[params['tv_track_tokens'][j]]) > 1:
                            o[j][:,t] = np.array([path_to_linestring(tv_paths_se2[params['tv_track_tokens'][j]]).project(query_pt),0])
                        else:
                            o[j][:,t] = np.array([0,0])
                        tv_lengths.append(params['tv_params'][j][0])
                        tv_widths.append(params['tv_params'][j][1])
                        tv_psi[j][0,0] = observations.tracked_objects.tracked_objects[observation_tokens.index(params['tv_track_tokens'][j])].box.center.heading

                if preds_list.shape[0] == 0:
                    for k in range(params['N_TV']):
                        tv_lengths.append(4)
                        tv_widths.append(2)
                elif preds_list.shape[0] != params['N_TV']:
                    for k in range(params['N_TV'] - preds_list.shape[0]):
                        tv_lengths.append(params['tv_params'][0][0])
                        tv_widths.append(params['tv_params'][0][0])
                        tv_psi[k][:,t+1] = params['tv_psi'][0,0,t,0]

        #Convert InterpolatedPath to casadi functions (routes and droutes)
        if config['prediction_method'] == 'idm':
            for path in agent_paths:
                s_arr = [point.progress for point in path.get_sampled_path()]
                x_arr = [point.x - ego_sim_init_state.center.point.x for point in path.get_sampled_path()] #relative to ego initial position to scale the global coordinates
                y_arr = [point.y - ego_sim_init_state.center.point.y for point in path.get_sampled_path()] #relative to ego initial position to scale the global coordinates
                psi_arr = [point.heading for point in path.get_sampled_path()]
                v_arr = [0 for _ in path.get_sampled_path()]
                routes.append(make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr))
                droutes.append(make_jac_fun(routes[-1]))

            o0 = []
            for agent in preds_list[0]:
                if isinstance(agent,List):
                    agent = agent[0]
                o0.append(np.array([[agent.progress],[agent.velocity]])) if isinstance(agent,IDMAgent) else o0.append(np.array([[path_to_linestring(tv_paths_se2[agent.metadata.track_token]).project(Point(*agent.center.point.array))],[agent.velocity.magnitude()]])) #s,v
        else:
            o0 = []
            for j in range(preds_list.shape[0]):
                s_arr = [o[j][0,t] for t in range(params['N'])]
                x_arr = [o_glob[j][0,t] for t in range(params['N'])] #relative to ego initial position to scale the global coordinates
                y_arr = [o_glob[j][1,t] for t in range(params['N'])] #relative to ego initial position to scale the global coordinates
                psi_arr = [tv_psi[j][0,t] for t in range(params['N'])]
                v_arr = [0 for _ in range(params['N'])]
                try:
                    routes.append(make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr))
                    droutes.append(make_jac_fun(routes[-1]))
                except:
                    #find non-increasing s_arr
                    eps = 0.1
                    s_arr = [o[j][0,0]] + [o[j][0,t] if o[j][0,t] > o[j][0,t-1] else o[j][0,t] + abs(o[j][0,t]-o[j][0,t-1]) + eps for t in range(1,params['N'])]
                    ind = 0
                    while not (ind == (len(s_arr) - 1)):
                        ind = s_arr_monotonic(s_arr)
                    routes.append(make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr))
                    droutes.append(make_jac_fun(routes[-1]))
                o0.append(np.array([[o[j][0,0]],[o[j][1,0]]]))
            if preds_list.shape[0] == 0:
                for k in range(params['N_TV']):
                    o0.append(np.array([[o[k][0,0]],[o[k][1,0]]]))
                    #Generate dummy route and droute functions
                    step_size = 1
                    s_arr = [t*step_size for t in range(params['N'])]
                    x_arr = [o_glob[k][0,t] for t in range(params['N'])] #relative to ego initial position to scale the global coordinates
                    y_arr = [o_glob[k][1,t] for t in range(params['N'])] #relative to ego initial position to scale the global coordinates
                    psi_arr = [tv_psi[k][0,t] for t in range(params['N'])]
                    v_arr = [0 for _ in range(params['N'])]
                    routes.append(make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr))
                    droutes.append(make_jac_fun(routes[-1]))
            elif preds_list.shape[0] != params['N_TV']:
                for k in range(params['N_TV'] - preds_list.shape[0]):
                    o0.append(o0[0])
                    routes.append(routes[0])
                    droutes.append(droutes[0])

        #Alternatively, we can acquire linearized prediction of the agents using u_tvs and routes
        #Ego trajectory
        x = x0 + np.zeros((2,params['N']+1))

        #TV trajectory
        if config['prediction_method'] == 'idm':
            o=[o0[i] + np.zeros((2,params['N']+1)) for i in range(params['N_TV'])]
            o_glob = [routes[i+1](o0[i][0,:])[:2].reshape((-1,1)) + np.zeros((2,params['N']+1)) for i in range(params['N_TV'])]
        else:
            pass

        #Initialize parameters
        Qs = [[np.identity(2) for _ in range(params['N'])] for _ in range(params['N_TV'])]
        do_glob = [[ca.DM(2,1) for _ in range(params['N'])] for _ in range(params['N_TV'])]
        x_glob = routes[0](x[0,0])[:2].reshape((-1,1))+np.zeros((2, params['N']+1))
        dx_glob=[ca.DM(2,1) for _ in range(params['N'])]

        #Ego vehicle, TV linearized dynamics
        A = np.array([[1., params['dt']],[0. , 1.]])
        B = np.array([[0.5*params['dt']**2],[params['dt']]])
        
        vehicle_parameters = ego_state.car_footprint.vehicle_parameters
        ev_dims= np.array([(vehicle_parameters.front_length + vehicle_parameters.rear_length)/2, vehicle_parameters.width/2]) #length, width
        Sev = np.diag(ev_dims**(-1.0))
        iSev  = np.linalg.inv(Sev)
        for t in range(params['N']):
            if u_opt is None:
                # print('USING ZERO CONTROL')
                u_opt = np.zeros((1,params['N']))

            a = u_opt[:,t]
            # if ego_traj:
            #     x[:,t+1] = A @ x[:,t] + B @ a
            #     x_glob[:,t+1] = np.array([ego_traj[min(t+1,len(ego_traj)-1)].center.x, ego_traj[min(t+1,len(ego_traj)-1)].center.y])
            #     dx_glob[t] = droutes[0](x[0,t+1])[:2]
            # else:
            x[:,t+1] = A @ x[:,t] + B @ a
            x_glob[:,t+1] = routes[0](x[0,t+1])[:2]
            dx_glob[t] = droutes[0](x[0,t+1])[:2]

            # if ego_traj:
            #     psi = ego_traj[min(t+1,len(ego_traj)-1)].rear_axle.heading #from prev MPC solution
            # else:
            psi = routes[0](x[0,t+1])[2] #from the route function
            Rev = np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
            for i in range(params['N_TV']): #iterate through all surrounding vehicles
                for t in range(params['N']):
                    if config['prediction_method'] == 'idm':
                        o[i][:,t+1] = A @ o[i][:,t] + B @ u_tvs[i][:,t]
                        o_glob[i][:,t+1] = routes[i+1](o[i][0,t+1])[:2] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                        psi = routes[i+1](o[i][0,t+1])[2] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                        Rtv = np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                    else:
                        psi = tv_psi[i][0,t] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                        Rtv = np.array([[np.cos(psi), -np.sin(psi)],[np.sin(psi), np.cos(psi)]])
                    do_glob[i][t] = droutes[i+1](o[i][0,t+1])[:2] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                    
                    # if tv_psi[i] is not None:
                    # Rtv = np.array([[np.cos(tv_psi[i][:,t+1]), np.sin(tv_psi[i][:,t+1])],[-np.sin(tv_psi[i][:,t+1]), np.cos(tv_psi[i][:,t+1])]]).squeeze().T
                    # else:
                    Stv_ = np.diag([tv_lengths[i]/2, tv_widths[i]/2])
                    Stv = np.linalg.inv(Stv_)
                    mat=Rev@iSev@Rtv.T@Stv@Stv@Rtv@iSev@Rev.T 
                    E, V =np.linalg.eigh(mat)
                    # S=np.diag((E**(-0.5)+1.0)**(-2))
                    S=np.diag(1/E)
                    Qs[i][t] = construct_Q_simple(tv_lengths[i], tv_widths[i], ego_radius=ev_dims[0], psi=psi)
                    # Qs[i][t]=Sev@Rev.T@V@S@V.T@Rev@Sev #if t <=4 else (1/(5**2))*np.eye(2) 
        #For multi-modal predictions:
        if is_mm_preds:
            if config['prediction_method'] == 'idm':
                n_modes = [2 for _ in range(2)] + [1 for _ in range(params['N_TV']-2)] #2 lane change mode vehicles from adjacent lanes
                mm_routes = [[copy.deepcopy(routes[i+1]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
            else:
                n_modes =  [config['num_modes'] for _ in range(params['N_TV'])]
                mm_routes = [[copy.deepcopy(routes[i+1]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
            mm_o      = [[copy.deepcopy(o[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
            mm_o_glob = [[copy.deepcopy(o_glob[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
            mm_u_tvs = [[copy.deepcopy(u_tvs[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
            
            mm_droutes = [[copy.deepcopy(do_glob[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
            mm_tv_psi = [[copy.deepcopy(tv_psi[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
            mm_do_glob = [[copy.deepcopy(do_glob[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
            mm_Qs = [[copy.deepcopy(Qs[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]

            if config['prediction_method'] == 'idm':
                #Lane change mode routes for the first two vehicles in preds_list
                routes_mm = []
                droutes_mm = []
                for i in range(2):
                    s_arr = [preds_list[0][i][-1].progress]
                    for t in range(params['N']-1):
                        s_arr.append(s_arr[-1]+dt*preds_list[t][i][-1].velocity+0.5*preds_list[t][i][-1]._u_prev*dt**2)
                    x_arr = [preds_list[t][i][-1].to_se2().x for t in range(params['N'])] #relative to ego initial position to scale the global coordinates
                    y_arr = [preds_list[t][i][-1].to_se2().y for t in range(params['N'])] #relative to ego initial position to scale the global coordinates
                    psi_arr = [preds_list[t][i][-1].to_se2().heading for t in range(params['N'])]
                    v_arr = [0 for _ in range(params['N'])]

                    routes_mm.append(make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr))
                    droutes_mm.append(make_jac_fun(routes_mm[-1]))
            n_tv = params['N_TV'] if config['prediction_method'] == 'idm' else preds_list.shape[0]
            for i in range(n_tv):
                # if n_modes[i] > 1 and len(preds_list[0][i]) > 1: 
                if n_modes[i] > 1:
                    for t in range(params['N']):
                        psi= routes[0](x[0,t+1])[2]
                        Rev=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                        for n in range(1,n_modes[i]):
                            if config['prediction_method'] == 'idm':
                                mm_o[i][n][:,t+1]=A @ mm_o[i][n][:,t] + B @ u_tvs[i][:,t] #assume same u_tvs for all modes (lane change modes)
                                mm_o_glob[i][n][:,t+1]=routes_mm[i](mm_o[i][n][0,t+1])[:2]
                                psi=routes_mm[i](mm_o[i][n][0,t+1])[2]
                            else:
                                mm_o_glob[i][n][:,t+1]=np.array([preds_list[i,n,t,0] - ego_sim_init_state.center.point.x, preds_list[i,n,t,1]-ego_sim_init_state.center.point.y])
                                query_pt = Point(preds_list[i,n,t,0], preds_list[i,n,t,1])
                                if len(tv_paths_se2[params['tv_track_tokens'][i]]) > 1:
                                    mm_o[i][n][:,t+1] = np.array([path_to_linestring(tv_paths_se2[params['tv_track_tokens'][i]]).project(query_pt),0])
                                else:
                                    mm_o[i][n][:,t+1] = mm_o[i][0][:,t]
                                psi = params['tv_psi'][i,n,t,0] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                                mm_tv_psi[i][n][0,t+1] = psi
                            mm_u_tvs[i][n][0,t]=u_tvs[i][:,t]
                            
                            if t==0:
                                try:
                                    if config['prediction_method'] == 'idm':
                                        mm_routes[i][n]=routes_mm[i]   
                                        mm_droutes[i][n][t]=droutes_mm[i](mm_o[i][n][0,t+1])[:2]            
                                    else:
                                        mm_o_glob[i][n][:,0] = np.array([observations.tracked_objects.tracked_objects[observation_tokens.index(params['tv_track_tokens'][i])].box.center.x - ego_sim_init_state.center.point.x ,observations.tracked_objects.tracked_objects[observation_tokens.index(params['tv_track_tokens'][i])].box.center.y - ego_sim_init_state.center.point.y])
                                        if len(tv_paths_se2[params['tv_track_tokens'][i]]) > 1:
                                            mm_o[i][n][:,0] = np.array([path_to_linestring(tv_paths_se2[params['tv_track_tokens'][i]]).project(Point(np.array([observations.tracked_objects.tracked_objects[observation_tokens.index(params['tv_track_tokens'][i])].box.center.x, observations.tracked_objects.tracked_objects[observation_tokens.index(params['tv_track_tokens'][i])].box.center.y]))),0])
                                        else:
                                            mm_o[i][n][:,0] = mm_o[i][0][:,0]
                                        mm_tv_psi[i][n][0,0] = observations.tracked_objects.tracked_objects[observation_tokens.index(params['tv_track_tokens'][i])].box.center.heading  
                                        #generate route functions
                                        if len(tv_paths_se2[params['tv_track_tokens'][i]]) > 1:
                                            s_arr = [mm_o[i][n][0,0]] + [path_to_linestring(tv_paths_se2[params['tv_track_tokens'][i]]).project(Point(preds_list[i,n,k,0], preds_list[i,n,k,1])) for k in range(1,params['N'])]
                                        else:
                                            s_arr = [mm_o[i][n][0,0]] + [mm_o[i][0][0,0]] * (params['N'] - 1)
                                        x_arr = [mm_o_glob[i][n][0,0]] + [preds_list[i,n,k,0] - ego_sim_init_state.center.point.x for k in range(1,params['N'])] #relative to ego initial position to scale the global coordinates
                                        y_arr = [mm_o_glob[i][n][1,0]] + [preds_list[i,n,k,1] - ego_sim_init_state.center.point.x for k in range(1,params['N'])] #relative to ego initial position to scale the global coordinates
                                        psi_arr = [mm_tv_psi[i][n][0,0]] + [params['tv_psi'][i,n,k,0] for k in range(1,params['N'])]
                                        v_arr = [0 for _ in range(params['N'])]
                                        try:
                                            mm_routes[i][n] = make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr)
                                            mm_droutes[i][n][t] = make_jac_fun(mm_routes[i][n])(mm_o[i][n][0,t+1])[:2]   
                                        except:
                                            eps = 0.1
                                            if len(tv_paths_se2[params['tv_track_tokens'][i]]) > 1:
                                                s_arr = [mm_o[i][n][0,0]] + [path_to_linestring(tv_paths_se2[params['tv_track_tokens'][i]]).project(Point(preds_list[i,n,k,0], preds_list[i,n,k,1])) if path_to_linestring(tv_paths_se2[params['tv_track_tokens'][i]]).project(Point(preds_list[i,n,k,0], preds_list[i,n,k,1])) > path_to_linestring(tv_paths_se2[params['tv_track_tokens'][i]]).project(Point(preds_list[i,n,k-1,0], preds_list[i,n,k-1,1])) else path_to_linestring(tv_paths_se2[params['tv_track_tokens'][i]]).project(Point(preds_list[i,n,k,0], preds_list[i,n,k,1])) + abs(path_to_linestring(tv_paths_se2[params['tv_track_tokens'][i]]).project(Point(preds_list[i,n,k,0], preds_list[i,n,k,1])) - path_to_linestring(tv_paths_se2[params['tv_track_tokens'][i]]).project(Point(preds_list[i,n,k-1,0], preds_list[i,n,k-1,1]))) + eps for k in range(1,params['N'])]
                                            else:
                                                s_arr = [mm_o[i][n][0,0]] + [mm_o[i][0][0,0]] * (params['N'] - 1)
                                            # if any s_arr are non-increasing, add eps to the non-increasing elements
                                            ind = 0
                                            while not (ind == (len(s_arr) - 1)):
                                                ind = s_arr_monotonic(s_arr)
                                            mm_routes[i][n] = make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr)
                                            mm_droutes[i][n][t] = make_jac_fun(mm_routes[i][n])(mm_o[i][n][0,t+1])[:2]
                                except:
                                    # pdb.set_trace()
                                    raise ValueError('Error generating mm_routes in get_preds()')

                            Rtv=np.array([[np.cos(psi), -np.sin(psi)],[np.sin(psi), np.cos(psi)]])
                            Stv_ = np.diag([tv_lengths[i]/2, tv_widths[i]/2])
                            Stv = np.linalg.inv(Stv_)
                            mat=Rev@iSev@Rtv.T@Stv@Stv@Rtv@iSev@Rev.T 
                            E, V =np.linalg.eigh(mat)
                            # S=np.diag((E**(-0.5)+1.0)**(-2))
                            S=np.diag(E)
                            mm_Qs[i][n][t] = construct_Q_simple(tv_lengths[i], tv_widths[i], ego_radius=ev_dims[0], psi=psi)

                            # mm_Qs[i][n][t]=Sev@Rev.T@V@S@V.T@Rev@Sev
        else:
            mm_o_glob = [[o_glob[i]] for i in range(params['N_TV'])]
            mm_u_tvs = [[u_tvs[i]] for i in range(params['N_TV'])]
            mm_routes = [[routes[i+1]] for i in range(params['N_TV'])]
            mm_do_glob = [[do_glob[i]] for i in range(params['N_TV'])]
            mm_tv_psi = [[tv_psi[i]] for i in range(params['N_TV'])]
            mm_Qs = [[Qs[i]] for i in range(params['N_TV'])]

        #tv length and width
        tv_params = [[tv_lengths[k], tv_widths[k]] for k in range(params['N_TV'])]
    except:
        pdb.set_trace()
    return x, x_glob, dx_glob, mm_o_glob, mm_u_tvs, mm_routes, mm_do_glob, mm_Qs, mm_tv_psi, tv_params, o0

def s_arr_monotonic(s_arr):
    '''
    Guarantees that s_arr is monotonic with eps 
    '''
    eps = 0.1
    for p in range(1, len(s_arr)):
        if s_arr[p] <= s_arr[p-1]:
            s_arr[p] = s_arr[p-1] + eps
            break
    return p

def construct_Q_simple(tv_length, tv_width, ego_radius, psi):
    """
    Constructs Q matrix for an ellipse defined by TV shape + ego radius buffer.
    Uses direct inverse-square scaling in rotated coordinates.
    """
    import numpy as np

    # Step 1: Semi-axes (radii) with ego buffer
    rx = tv_length / 2 + ego_radius
    ry = tv_width / 2 + ego_radius

    # Step 2: Inverse square matrix (axis-aligned)
    D = np.diag([1 / rx**2, 1 / ry**2])  # Q in TV's body frame

    # Step 3: Rotation matrix (TV heading)
    R = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi),  np.cos(psi)]
    ])

    # Step 4: Rotate into world frame
    Q = R @ D @ R.T
    return Q

def check_agents_in_preds(preds: List[IDMAgent], indices) -> bool:
    '''
    Check if there are n agents in the prediction list
    '''
    agents = []
    for i in indices:
        if i > (len(preds) -1):
            agents.append(i)
    if agents:
        return agents, False
    else:
        return None, True

def filter_preds(preds_list: Union[List[IDMAgent],List[Agent]], n: int, ego_state) -> List[IDMAgent]:
    '''
    Choose n agents from the list of predictions based on distance from ego_state
    '''
    x,y = ego_state.center.x, ego_state.center.y

    #Sort agents based on distance from ego
    if isinstance(preds_list[0][0],IDMAgent):
        dists = [np.sqrt((agent.to_se2().x-x)**2 + (agent.to_se2().y-y)**2) for agent in preds_list[0]] #distance from ego at current time
    elif isinstance(preds_list[0][0],Agent):
        dists = [np.sqrt((agent.center.x-x)**2 + (agent.center.y-y)**2) for agent in preds_list[0]]
    else:
        raise ValueError('Unknown agent type')
    sorted_inds = np.argsort(dists)   
    if n > len(preds_list[0]):
        m = len(preds_list[0])
    else:
        m = n

    # output = [[pred[i] for i in sorted_inds[:m]] for pred in preds_list]
    output = []
    for t, pred in enumerate(preds_list):
        indices, flag = check_agents_in_preds(pred,list(sorted_inds[:m]))
        if flag:
            output.append([*map(pred.__getitem__, sorted_inds[:m])])
        else:
            try:
                add_inds = []
                temp_inds = list(sorted_inds[:m])
                for i in indices:
                    temp_inds.remove(i)
                    #get index of element i in sorted_inds[:m]
                    add_inds.append(list(sorted_inds[:m]).index(i))
                try:
                    temp_list = [pred[m] for m in temp_inds]
                except:
                    print('Error temp_list failed in filter_preds')
                    pdb.set_trace()

                for add_ind in add_inds:
                    i = sorted_inds[add_ind]
                    offset = 1
                    while len(preds_list[t-offset]) - 1 < i:
                        offset += 1
                    temp_list.insert(add_ind,preds_list[t-offset][i])
                # temp_list.insert(add_ind,output[-1][i])
                output.append(temp_list)
            except:
                #backup. append exiting agents
                temp_inds.insert(add_ind,temp_inds[0])
                output.append([*map(pred.__getitem__, temp_inds)])
    # pdb.set_trace()
    # preds_list = [[*map(pred.__getitem__, sorted_inds[:m])] for pred in preds_list] #Choose n closest agents

    return output

def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    dual_classes = np.concatenate([path["dual_classes"] for path in paths])
    return observations, actions, rewards, next_observations, terminals, dual_classes

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x #creates a generator object

def unflatten_duals(x,l1_dual_dim,ca_dual_dim,data2tar = False):
    l1_num=sum(l1_dual_dim[1])*(l1_dual_dim[0])*2
    l1_dual_arr = x[0,:l1_num].flatten().tolist() 
    ca_dual_arr = x[0,l1_num:].flatten().tolist()

    if data2tar:
        l1_dual = [[[ []  for j in range(l1_dual_dim[1][k])] for k in range(l1_dual_dim[2])] for t in range(l1_dual_dim[0])]
        ca_dual = [[[ []  for j in range(ca_dual_dim[1])] for k in range(ca_dual_dim[2])]  for t in range(ca_dual_dim[0])] 

        step=0
        for t in range(l1_dual_dim[0]):
            for k in range(l1_dual_dim[2]):
                for j in range(l1_dual_dim[1][k]):
                    l1_dual[t][k][j]+=[l1_dual_arr[step:step+2]]
                    step+=2
        step=0
        for t in range(ca_dual_dim[0]):
            for k in range(ca_dual_dim[2]):
                for j in range(ca_dual_dim[1]):
                    ca_dual[t][k][j]+=list(ca_dual_arr[step:step+1])
                    step+=1     
    else:
        l1_dual = [[[None for t in range(l1_dual_dim[0])] for j in range(l1_dual_dim[1][k])] for k in range(l1_dual_dim[2])]
        ca_dual = [[[None for t in range(ca_dual_dim[0])] for j in range(ca_dual_dim[1])] for k in range(ca_dual_dim[2])]
    
        step=0
        for k in range(l1_dual_dim[2]):
            for j in range(l1_dual_dim[1][k]):
                for t in range(l1_dual_dim[0]):
                    l1_dual[k][j][t]=l1_dual_arr[step:step+2]
                    step+=2 #l1 gain has 2 elements (position, velocity) see the paper.
        step=0
        for k in range(ca_dual_dim[2]):
            for j in range(ca_dual_dim[1]):
                for t in range(ca_dual_dim[0]):
                    ca_dual[k][j][t]=ca_dual_arr[step:step+1]
                    step+=1

    return l1_dual, ca_dual

def observation_flatten(obs, use_ttc=True):
    if use_ttc:
        return np.concatenate([ obs['x0'],np.array([obs['u_prev'],obs['ev_route']]), obs['ttc'], np.stack(obs['o0'],axis=0).flatten(),obs['mmpreds']]).astype('float32')
    else:
        return np.concatenate([ obs['x0'],np.array([obs['u_prev'],obs['ev_route']]), np.stack(obs['o0'],axis=0).flatten(),obs['mmpreds']]).astype('float32')

def observation_unflatten(obs_flat, n_tv, use_ttc=True):
    if len(obs_flat.shape) > 1: #If batch
        o0_arr = obs_flat[(1+n_tv)*int(use_ttc) + 4 : (1+n_tv)*int(use_ttc) + 4 + n_tv * 2].reshape(-1,2)
        obs_dict={'x0': obs_flat[0:2], 'u_prev': obs_flat[2], 'ev_route': int(obs_flat[3]), 'o0': [o0_arr[ind,:] for ind in range(o0_arr.shape[0])], 'mmpreds':obs_flat[(1+n_tv)*int(use_ttc)+4 + n_tv * 2:] }
        if use_ttc:
            obs_dict.update({'ttc':obs_flat[4:4+1+n_tv] })
        return obs_dict
    else:    
        o0_arr = obs_flat[(1+n_tv)*int(use_ttc)+4:(1+n_tv)*int(use_ttc)+4 + n_tv * 2].reshape(-1,2)
        obs_dict={'x0': obs_flat[0:2], 'u_prev': obs_flat[2], 'ev_route': int(obs_flat[3]), 'o0': [o0_arr[ind,:] for ind in range(o0_arr.shape[0])], 'mmpreds':obs_flat[(1+n_tv)*int(use_ttc)+4 + n_tv * 2:] }
        if use_ttc:
            obs_dict.update({'ttc':obs_flat[4:4+1+n_tv] })
        return obs_dict
    
def to_tensor_var(x, use_cuda=True, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    
class weighted_MSEloss(th.nn.Module):
    '''
    weighted mse loss for prioritising important l1_duals in target
    '''
    def __init__(self, lmbd_bnd):
        super().__init__()
        self.lmbd_bnd = lmbd_bnd

    def __call__(self, input, target):
        '''
        state: th.Tensor
        out: th.Tensor
        '''
        lmbd_bool = ((target < 1e-8) | (target > self.lmbd_bnd - 1e-8)).float()*49 + th.ones_like(target)
        out = th.nn.functional.mse_loss(lmbd_bool*input, lmbd_bool*target)

        return out

