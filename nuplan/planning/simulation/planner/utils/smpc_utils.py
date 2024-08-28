"""A
Some miscellaneous utility functions

Functions to edit:
    1. sample_trajectory
"""
import numpy as np
import time
from typing import List
from torch.autograd import Variable
from collections.abc import Iterable
import numpy as np
import torch as th
import pdb
import time
import os
import copy
import casadi as ca
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent

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
            

def get_preds(current_input, preds_list: List[IDMAgent], x0, params, routes, droutes, u_opt = None, ego_traj = None):
    '''
    Getting EV predictions from previous MPC solution.
    This is used for linearizing the collision avoidance constraints
    '''
    ego_state, observations = current_input.history.current_state
    
    #Convert IDM predictions to global coordinates
    o_glob = [np.zeros((2,params['N']+1)) for _ in range(len(preds_list[0]))]
    o = [np.zeros((2,params['N']+1)) for _ in range(len(preds_list[0]))]
    tv_psi = [np.zeros((1,params['N']+1)) for _ in range(len(preds_list[0]))]
    u_tvs=[np.zeros((1,params['N'])) for _ in range(len(preds_list[0]))]
    tv_lengths = []
    tv_widths = []
    agent_paths = []
    for t, agents in enumerate(preds_list):
        for j, agent in enumerate(agents):
            o_glob[j][:,t] = np.array([agent.to_se2().x,agent.to_se2().y]) #x,y
            o[j][:,t] = np.array([agent.progress,agent.velocity]) #s,v
            tv_psi[j][:,t] = agent.to_se2().heading
            if t < params['N']:
                u_tvs[j][:,t] = agent._u_prev
            if t == 0:
                agent_paths.append(agent._path)
                tv_lengths.append(agent.length)
                tv_widths.append(agent.width)
                assert agent.length > 0 and agent.width > 0, 'TV length and width must be greater than 0'

    #Convert InterpolatedPath to casadi functions (routes and droutes)
    #TODO: Q for Sid: casadi function v needed? 
    for path in agent_paths:
        s_arr = [point.progress for point in path._path]
        x_arr = [point.x for point in path._path]
        y_arr = [point.y for point in path._path]
        psi_arr = [point.heading for point in path._path]
        # v_arr = [point.v for point in agent._path]
        v_arr = [0 for _ in path._path]
        test_s = np.array(s_arr)
        if not np.all((test_s[1:] - test_s[:-1]) > 0):
            print('s_arr not increasing')
            pdb.set_trace()
        routes.append(make_ca_fun(s_arr, x_arr, y_arr, psi_arr, v_arr))
        droutes.append(make_jac_fun(routes[-1]))

    #Alternatively, we can acquire linearized prediction of the agents using u_tvs and routes
    o0 = []
    for agent in current_input.agents.values():
        o0.append(np.array([[agent.progress],[agent.velocity]]))

    #Ego trajectory
    x = x0 + np.zeros((2,params['N']+1))

    #TV trajectory
    o=[o0[i] + np.zeros((2,params['N']+1)) for i in range(len(preds_list[0]))]

    #Initialize parametsr
    Qs = [[np.identity(2) for _ in range(params['N'])] for _ in range(len(preds_list[0]))]
    do_glob = [[ca.DM(2,1) for _ in range(params['N'])] for _ in range(len(preds_list[0]))]
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
            #TODO: Implement IDM control or other heuristics rule when u_opt is None
            # if x0[1] > 0:
            #     u_opt = np.zeros((1,params['N']))
            # else:
            u_opt = np.zeros((1,params['N']))

        a = u_opt[:,t]
        x[:,t+1] = A @ x[:,t] + B @ a
        x_glob[:,[t+1]] = routes[0](x[0,t+1])[:2]
        dx_glob[t] = droutes[0](x[0,t+1])[:2]
        # pdb.set_trace()
        # if ego_traj:
            # psi = ego_traj[t].rear_axle.heading #from prev MPC solution
        psi = routes[0](x[0,t+1])[2] #from the route function
        # Rev = np.array([[np.cos(ego_psi[t+1]), np.sin(ego_psi[t+1])],[-np.sin(ego_psi[t+1]), np.cos(ego_psi[t+1])]]).squeeze().T
        Rev = np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T

        for i in range(len(preds_list[0])): #iterate through all surrounding vehicles
            for t in range(params['N']):
                o[i][:,t+1] = A @ o[i][:,t] + B @ u_tvs[i][:,t]
                o_glob[i][:,[t+1]] = routes[i+1](o[i][0,t+1])[:2] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                do_glob[i][t] = droutes[i+1](o[i][0,t+1])[:2] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                psi = routes[i+1](o[i][0,t+1])[2] #call the tv route function (0=ego, 1=tv1, 2=tv2,...)
                Rtv = np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                # Rtv = np.array([[np.cos(tv_psi[i][:,t+1]), np.sin(tv_psi[i][:,t+1])],[-np.sin(tv_psi[i][:,t+1]), np.cos(tv_psi[i][:,t+1])]]).squeeze().T
                
                Stv_ = np.diag([tv_lengths[i], tv_widths[i]])
                Stv = np.linalg.inv(Stv_)
                mat=Rev@iSev@Rtv.T@Stv@Stv@Rtv@iSev@Rev.T 
                E, V =np.linalg.eigh(mat)
                S=np.diag((E**(-0.5)+1.0)**(-2))
                Qs[i][t]=Sev@Rev.T@V@S@V.T@Rev@Sev if t <=4 else (1/5**2)*np.eye(2) 

    return x, x_glob, dx_glob, o_glob, u_tvs, routes, do_glob, Qs 

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

def obs_normalize(obs, reduced_mode =True):
    '''
    Assume reduced_mode = True
    obs #shape (N, 17)
    "mmpreds" : MultiDiscrete([4,3,5])
    '''
    obs_norm = copy.deepcopy(obs)
    
    def clip(input,min,max):
        if isinstance(input,th.Tensor):
            return th.clip(input,min, max)
        else:
            return NotImplementedError
        
    if reduced_mode:
        #Ego normalization
        obs_norm[:,0] = (obs[:,0] / 110)
        v_max = 10; v_min = -1
        obs_norm[:,1] = (clip(obs[:,1],v_min, v_max) - v_min)/(v_max - v_min) #min-max normalization
        a_max = 2; a_min = -5
        obs_norm[:,2] = (clip(obs[:,2], a_min, a_max) - a_min) / (a_max - a_min) #min-max normalization
        obs_norm[:,3] /= 2 #ego route: Discrete(2)

        #ittc normalization
        obs_norm[:,4:4+4] = (clip(obs_norm[:,4:4+4], 0.05, 10) - 0.05)/ (10 - 0.05)

        #o0 normalization
        obs_norm[:,8] /= th.where(obs[:,8] == -15., th.tensor(15.), th.tensor(110))
        obs_norm[:,10] /= th.where(obs[:,10] == -15., th.tensor(15.), th.tensor(110))
        obs_norm[:,12] /= th.where(obs[:,12] == -15., th.tensor(15.), th.tensor(110))

        obs_norm[:,9] = (clip(obs_norm[:,9],v_min, v_max) - v_min)/(v_max - v_min)
        obs_norm[:,11] = (clip(obs_norm[:,9],v_min, v_max) - v_min)/(v_max - v_min)
        obs_norm[:,13] = (clip(obs_norm[:,9],v_min, v_max) - v_min)/(v_max - v_min)

        #mmpreds normalization
        obs_norm[:,14] /= 4 - 1
        obs_norm[:,15] /= 3 - 1
        obs_norm[:,16] /= 5 - 1

    return obs_norm
    

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
        
def sample_trajectory(env, policy=None, max_path_length=100, use_cuda = False,seed=None,render=False,ani_save_dir=None,expert=True, binary_pred= True,tertiary_l1 = False, normalize_obs=False,dagger_mode=False): 
    """Sample a rollout in the environment from a policy."""
    print('Sampling a trajectory...')
    rollout_done = False
    ob, info = env.reset(seed=seed)
    obs, acs, rewards, next_obs, terminals, solve_times, infeas, collisions, vars_kept, const_kept, NN_query_times, dual_classes= [], [], [], [], [], [], [], [], [], [], [], []
    t_wall_sums, t_proc_sums = [], []
    steps = 0
    only_ca_pred = True
    

    while steps <= max_path_length and not rollout_done:
        # print(f"Steps {steps}".center(80,'-'))
        if policy is None:
            new_ob, reward, done, _, infos = env.step(action=None)
            NN_query_time = 0
        else:
            st = time.time()
            l1_pred = th.zeros((1,sum(env.smpc.N_modes)*(env.smpc.N-1)*2)).to(device="cuda" if use_cuda else "cpu") #Dummy l1 duals required for downstream smpcfr.py
            ca_pred = th.sigmoid(policy(obs_normalize(to_tensor_var(observation_flatten(ob), use_cuda=use_cuda)[None]) if normalize_obs else to_tensor_var(observation_flatten(ob), use_cuda=use_cuda)[None])).round()
            NN_query_time = time.time() - st
            action = th.hstack((l1_pred,ca_pred))
            
            l1_dual, ca_dual = unflatten_duals(action.detach().cpu().numpy(), [env.smpc.N-1, env.smpc.N_modes, env.smpc.N_TV], [env.smpc.N-1, len(env.smpc.mode_map), env.smpc.N_TV] )
            action = [l1_dual, ca_dual]
            new_ob, reward, done, _, infos = env.step(action=action)        
            # print("Step taken ",new_ob["x0"] )

        steps += 1
        rollout_done = done or infos['discard']
        if rollout_done:
            if infos['infeas']:
                infeas.append(infos['infeas'])
            
        if not infos['infeas']:
            l1_duals = np.fromiter(flatten(infos["l1_duals"]),float)
            ca_duals = np.fromiter(flatten(infos["ca_duals"]),float)
            expert_action = np.concatenate((l1_duals,ca_duals))
            action = expert_action
            
            obs.append(observation_flatten(ob))
            acs.append(action)
            rewards.append(reward)
            next_obs.append(observation_flatten(new_ob))
            terminals.append(rollout_done)
            solve_times.append(infos['solve_time'])
            NN_query_times.append(NN_query_time)
            infeas.append(infos['infeas'])
            collisions.append(infos['discard'])
            dual_classes.append(infos["dual_class"])
            if 'vars_kept' in infos.keys():
                vars_kept.append(infos['vars_kept'])
                const_kept.append(infos['const_kept'])

            if env.env_mode == 2:
                t_wall_sums.append(infos['t_wall_sum'])
                t_proc_sums.append(infos['t_proc_sum'])
            else:
                #Append placeholders
                t_wall_sums.append(-1)
                t_proc_sums.append(-1)
        elif infos['infeas'] and not dagger_mode:
            infeas.append(infos['infeas'])
            solve_times.append(None)
            NN_query_times.append(None)
            collisions.append(infos['discard'])
            dual_classes.append(None)
            t_wall_sums.append(None)
            t_proc_sums.append(None)
            vars_kept.append(None)
            const_kept.append(None)            
        ob = new_ob
    print(f'Steps: {steps}')
    if not infos['discard'] or dagger_mode:
        path = {'observation':obs,'reward': np.array(rewards, dtype=np.float32), 'action': np.array(acs, dtype=np.float32),'next_observation': next_obs, "terminal": np.array(terminals, dtype=np.float32), "infeas": infeas, "solve_time":solve_times, 'collision': collisions, 'vars_kept': vars_kept, 'const_kept':const_kept, 'NN_query_time': NN_query_times, "dual_classes":dual_classes, 't_wall_sum': t_wall_sums, 't_proc_sum':t_proc_sums}     #state and expert action  
    else:
        path = None

    if render:
        animation = env.render()
        if expert:
            name = 'expert'
        else:
            name = 'HMPC'
        if os.path.isdir(ani_save_dir):
            pass
        else:
            os.mkdir(ani_save_dir)
        animation.save(ani_save_dir + 'eval_' + name +'.mp4')
    return path

def get_pathlength(path):
    return len(path["reward"])

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, use_cuda = False,seed=None,tertiary_l1 = False,normalize_obs=False,dagger_mode=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length,use_cuda=use_cuda,seed=seed,tertiary_l1 = tertiary_l1,normalize_obs=normalize_obs,dagger_mode=dagger_mode)
        if path is not None: #if not discard
            paths.append(path)
            timesteps_this_batch += get_pathlength(path)
        else:
            seed += 1
        

    return paths, timesteps_this_batch
    
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

