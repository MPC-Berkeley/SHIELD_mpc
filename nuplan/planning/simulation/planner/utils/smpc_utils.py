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

def get_preds(current_input, preds_list: Union[List[IDMAgent],List[Agent]], x0, params, routes, droutes, simulation_t: int, u_opt = None, ego_traj = None,ego_p0=None,tv_paths_se2=None,dt=0.1,is_mm_preds=False):
    '''
    Getting EV predictions from previous MPC solution.
    This is used for linearizing the collision avoidance constraints
    '''
    assert ego_p0 is not None, 'Ego initial position must be provided'
    ego_state, observations = current_input.history.current_state
    
    #Convert IDM predictions to global coordinates
    o_glob = [np.zeros((2,params['N']+1)) for _ in range(params['N_TV'])]
    o = [np.zeros((2,params['N']+1)) for _ in range(params['N_TV'])]
    tv_psi = [np.zeros((1,params['N']+1)) for _ in range(params['N_TV'])]
    u_tvs=[np.zeros((1,params['N'])) for _ in range(params['N_TV'])]
    tv_lengths = []
    tv_widths = []
    agent_paths = []
    for t, agents in enumerate(preds_list):
        for j, agent in enumerate(agents):
            if isinstance(agent,List):
                agent = agent[0] #use first mode
            o_glob[j][:,t] = np.array([agent.to_se2().x,agent.to_se2().y]) if isinstance(agent,IDMAgent) else np.array([agent.center.x,agent.center.y])  #x,y
            o[j][:,t] = np.array([agent.progress,agent.velocity]) if isinstance(agent,IDMAgent) else np.array([path_to_linestring(tv_paths_se2[agent.metadata.track_token]).project(Point(*agent.center.point.array)),agent.velocity.magnitude()]) #s,v
            tv_psi[j][:,t] = agent.to_se2().heading if isinstance(agent,IDMAgent) else agent.center.heading
            if t < params['N']:
                try:
                    next_time_agent = preds_list[t+1][j][0] if is_mm_preds else preds_list[t+1][j]
                    u_tvs[j][:,t] = next_time_agent._u_prev if isinstance(agent,IDMAgent) else (next_time_agent.velocity.magnitude()-agent.velocity.magnitude())/dt #u
                except:
                    print('error: u_tvs in get_preds()')
                    pdb.set_trace()
            if t == 0:
                agent_paths.append(agent._path) if isinstance(agent,IDMAgent) else agent_paths.append(create_path_from_se2(tv_paths_se2[agent.metadata.track_token]))
                tv_lengths.append(agent.length) if isinstance(agent,IDMAgent) else tv_lengths.append(agent.box.length)
                tv_widths.append(agent.width) if isinstance(agent,IDMAgent) else tv_widths.append(agent.box.width)
                assert agent.length if isinstance(agent,IDMAgent) else agent.box.length > 0 and agent.width if isinstance(agent,IDMAgent) else agent.box.width > 0, 'TV length and width must be greater than 0'

    #Convert InterpolatedPath to casadi functions (routes and droutes)
    for path in agent_paths:
        s_arr = [point.progress for point in path.get_sampled_path()]
        x_arr = [point.x for point in path.get_sampled_path()] #relative to ego initial position to scale the global coordinates
        y_arr = [point.y for point in path.get_sampled_path()] #relative to ego initial position to scale the global coordinates
        psi_arr = [point.heading for point in path.get_sampled_path()]
        v_arr = [0 for _ in path.get_sampled_path()]
        routes.append(make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr))
        droutes.append(make_jac_fun(routes[-1]))

    #Alternatively, we can acquire linearized prediction of the agents using u_tvs and routes
    o0 = []
    for agent in preds_list[0]:
        if isinstance(agent,List):
            agent = agent[0]
        o0.append(np.array([[agent.progress],[agent.velocity]])) if isinstance(agent,IDMAgent) else o0.append(np.array([[path_to_linestring(tv_paths_se2[agent.metadata.track_token]).project(Point(*agent.center.point.array))],[agent.velocity.magnitude()]])) #s,v

    #Ego trajectory
    x = x0 + np.zeros((2,params['N']+1))

    #TV trajectory
    o=[o0[i] + np.zeros((2,params['N']+1)) for i in range(params['N_TV'])]
    o_glob = [routes[i+1](o0[i][0,:])[:2].reshape((-1,1)) + np.zeros((2,params['N']+1)) for i in range(params['N_TV'])]

    #Initialize parameters
    Qs = [[np.identity(2) for _ in range(params['N'])] for _ in range(params['N_TV'])]
    do_glob = [[ca.DM(2,1) for _ in range(params['N'])] for _ in range(params['N_TV'])]
    x_glob = routes[0](x[0,0])[:2].reshape((-1,1))+np.zeros((2, params['N']+1))
    dx_glob=[ca.DM(2,1) for _ in range(params['N'])]

    #Ego vehicle, TV linearized dynamics
    A = np.array([[1., params['dt']],[0. , 1.]])
    B = np.array([[0.5*params['dt']**2],[params['dt']]])
    
    vehicle_parameters = ego_state.car_footprint.vehicle_parameters
    ev_dims= np.array([vehicle_parameters.front_length + vehicle_parameters.rear_length, vehicle_parameters.width]) #length, width
    Sev = np.diag(ev_dims**(-1.0))
    iSev  = np.linalg.inv(Sev)
    iSev[-1,-1]+=0.3
    Sev=np.linalg.inv(iSev)
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
                o[i][:,t+1] = A @ o[i][:,t] + B @ u_tvs[i][:,t]
                o_glob[i][:,t+1] = routes[i+1](o[i][0,t+1])[:2] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                do_glob[i][t] = droutes[i+1](o[i][0,t+1])[:2] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                psi = routes[i+1](o[i][0,t+1])[2] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                
                # if tv_psi[i] is not None:
                # Rtv = np.array([[np.cos(tv_psi[i][:,t+1]), np.sin(tv_psi[i][:,t+1])],[-np.sin(tv_psi[i][:,t+1]), np.cos(tv_psi[i][:,t+1])]]).squeeze().T
                # else:
                Rtv = np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze()#.T
                Stv_ = np.diag([tv_lengths[i], tv_widths[i]])
                Stv = np.linalg.inv(Stv_)
                mat=Rev@iSev@Rtv.T@Stv@Stv@Rtv@iSev@Rev.T 
                E, V =np.linalg.eigh(mat)
                S=np.diag((E**(-0.5)+1.0)**(-2))
                Qs[i][t]=Sev@Rev.T@V@S@V.T@Rev@Sev if t <=4 else (1/5**2)*np.eye(2) 

    #For extension to multi-modal prediction. Here we assume only one mode per TV
    if is_mm_preds and isinstance(preds_list[0][0][0],IDMAgent):
        n_modes =  [2 for _ in range(2)] + [1 for _ in range(params['N_TV']-2)] #2 lane change mode vehicles from adjacent lanes
        mm_o      = [[copy.deepcopy(o[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
        mm_o_glob = [[copy.deepcopy(o_glob[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
        mm_u_tvs = [[copy.deepcopy(u_tvs[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
        mm_routes = [[copy.deepcopy(routes[i+1]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
        mm_droutes = [[copy.deepcopy(do_glob[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]

        mm_do_glob = [[copy.deepcopy(do_glob[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]
        mm_Qs = [[copy.deepcopy(Qs[i]) for _ in range(n_modes[i])] for i in range(params['N_TV'])]

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

        for i in range(params['N_TV']):
            if n_modes[i] > 1 and len(preds_list[0][i]) > 1: 
                for t in range(params['N']):
                    psi= routes[0](x[0,t+1])[2]
                    Rev=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                    
                    n = 1 #lane change mode index
                        
                    if t==0:
                        mm_routes[i][n]=routes_mm[i]

                    mm_o[i][n][:,t+1]=A @ mm_o[i][n][:,t] + B @ u_tvs[i][:,t] #assume same u_tvs for all modes (lane change modes)
                    mm_o_glob[i][n][:,t+1]=routes_mm[i](mm_o[i][n][0,t+1])[:2]
                    mm_droutes[i][n][t]=droutes_mm[i](mm_o[i][n][0,t+1])[:2]
                    mm_u_tvs[i][n][0,t]=u_tvs[i][:,t]
                    psi=routes_mm[i](mm_o[i][n][0,t+1])[2]
                    Rtv=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                    Stv_ = np.diag([tv_lengths[i], tv_widths[i]])
                    Stv = np.linalg.inv(Stv_)
                    mat=Rev@iSev@Rtv.T@Stv@Stv@Rtv@iSev@Rev.T 
                    E, V =np.linalg.eigh(mat)
                    S=np.diag((E**(-0.5)+1.0)**(-2))
                    mm_Qs[i][n][t]=Sev@Rev.T@V@S@V.T@Rev@Sev if t <=4 else (1/5**2)*np.eye(2)
    else:
        mm_o_glob = [[o_glob[i]] for i in range(params['N_TV'])]
        mm_u_tvs = [[u_tvs[i]] for i in range(params['N_TV'])]
        mm_routes = [[routes[i+1]] for i in range(params['N_TV'])]
        mm_do_glob = [[do_glob[i]] for i in range(params['N_TV'])]
        mm_Qs = [[Qs[i]] for i in range(params['N_TV'])]
    # pdb.set_trace()

    #tv length and width
    tv_params = [[tv_lengths[k], tv_widths[k]] for k in range(params['N_TV'])]

    return x, x_glob, dx_glob, mm_o_glob, mm_u_tvs, mm_routes, mm_do_glob, mm_Qs, tv_psi, tv_params

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
        l1_dual = [[[ [] for t in range(l1_dual_dim[0])] for j in range(l1_dual_dim[1][k])] for k in range(l1_dual_dim[2])]
        ca_dual = [[[ [] for t in range(ca_dual_dim[0])] for j in range(ca_dual_dim[1])] for k in range(ca_dual_dim[2])]
    

        step=0
        for t in range(l1_dual_dim[0]):
            for k in range(l1_dual_dim[2]):
                for j in range(l1_dual_dim[1][k]):

                    l1_dual[k][j][t]+=[l1_dual_arr[step:step+2]]
                    step+=2
        step=0
        for t in range(ca_dual_dim[0]):
            for k in range(ca_dual_dim[2]):
                for j in range(ca_dual_dim[1]):

                    ca_dual[k][j][t]+=list(ca_dual_arr[step:step+1])
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

