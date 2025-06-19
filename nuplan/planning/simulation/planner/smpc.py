import logging
import math
import numpy as np

import casadi as ca
import pdb
from itertools import product
import time
import copy
from typing import List
from nuplan.planning.simulation.planner.utils.smpc_utils import flatten, unflatten_duals
from scipy.optimize import lsq_linear
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from scipy.linalg import clarkson_woodruff_transform as sketch
from scipy.sparse import block_diag, vstack
from scipy.sparse.linalg import lsmr, svds, lsqr
import scipy.sparse as sp

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import lsq_linear
logger = logging.getLogger(__name__)


class SMPC():

    def __init__(self,
                ev,
                N            =  6,
                V_MIN        = -1.,       #Speed, acceleration constraints
                V_MAX        = 10.0, 
                A_MIN        = -5.0,
                A_MAX        =  2.0,
                TIGHTENING   =  2.6, #2.6, # of std that you want to be robust w.r.t. the TV uncertainty
                EV_NOISE_STD    =  [0.001, 0.001],
                TV_NOISE_STD    =[[0.01, 0.02]]*5,
                Q = 1.,       # cost for measuring progress: -Q*s_{t+1}. #was 1.
                R = 1.,       # cost for penalizing large input rate: (u_{t+1}-u_t).T@R@(u_{t+1}-u_t) #was 1.5
                ev_length = 4.47,
                offline_mode=True,
                solver="ipopt",
                open_loop = False,
                eval_mode = False,
                is_mm_preds: bool = False, 
                route = None,
                preds = List,
                canon_prob_fn=None,
                config=None,
                ):
        # self.routes=routes
        self.ev=ev
        self.N=N
        self.V_MIN=V_MIN
        # self.V_MAX=V_MAX
        self.A_MAX=A_MAX
        self.A_MIN=A_MIN
        self.ev_length = ev_length
        self.canon_prob_fn=canon_prob_fn
        self.config = config
        print('[smpc.py]: Collision Avoidance Method is ',self.config['collision_avoidance_method'])

        self.preds = preds #Predictions of the vehicles List[List[IDMAgent]]. Outer list is of length N+1 and inner list is of length N_TV
        self.N_TV=len(preds[0])
        self.is_mm_preds = is_mm_preds
        if self.is_mm_preds:
            assert self.N_TV > 2
            if self.config['prediction_method']=='idm':
                self.N_modes=[2 for _ in range(2)] + [1 for _ in range(self.N_TV - 2)]
            else:
                self.N_modes=[self.config['num_modes'] for _ in range(self.N_TV)]

        else:
            self.N_modes=[1 for _ in range(self.N_TV)] #Assume, single mode per vehicles

        # Maps a mode, say 10, to the modes of the TVs, like (0,1,1,3,3)
        self.mode_map = dict(enumerate(product(*[range(self.N_modes[k]) for k in range(self.N_TV)])))
        if self.config['collision_avoidance_method'] == 'obca':
            self.tight=2.7
        elif self.config['collision_avoidance_method'] == 'affine':
            self.tight=2.7
        else:
            raise ValueError(f"Unknown collision avoidance method: {self.config['collision_avoidance_method']}")
        self.ev_n_std = EV_NOISE_STD

        self.tv_n_std = [TV_NOISE_STD for _ in range(self.N_TV)]

        self.Q_cost = ca.diag(Q)
        self.R_cost = ca.diag(R)

        self.A=ev[0]
        self.B=ev[1]
        
        self.Atv=self.A 
        self.Btv=self.B
          
        self.vars_kept = None
        self.constr_kept = None
        self.offline=offline_mode
        self.open_loop = open_loop
        self.route = route
        
        p_opts = {'expand': False, 'print_time':0, 'verbose' :False, 'error_on_fail':0}
        # s_opts = {'print_level': 0,'tol':5e-3,'max_wall_time': 120.,'constr_viol_tol':1e-4} 
        s_opts = {'print_level': 0,'tol':1e-4,'max_wall_time': 120.,'constr_viol_tol':1e-4} 
        if eval_mode:
            s_opts.update({'max_wall_time': 15.,'constr_viol_tol':1e-3})

        s_opts_grb = {'OutputFlag': 0, 'PSDTol' : 1e-2,
                       'FeasibilityTol' : 1e-2, 
                       'BarConvTol':1e-2, 
                       'BarQCPConvTol':1e-2,
                       'LogToConsole': 0}
        p_opts_grb = {'expand': True,'error_on_fail':0, 'verbose':False, 'ad_weight':0}

        self.solver=solver
        
        if self.solver=="ipopt":
            self.opti=ca.Opti()
            self.opti.solver("ipopt", p_opts, s_opts)
        else:
            self.opti=ca.Opti("conic")
            self.opti.solver("gurobi", p_opts_grb, s_opts_grb)


        def _flatten2ca(xs):
            if type(xs) == type([]):
                for x in xs:
                    yield from _flatten2ca(x)
            else:
                yield ca.vec(xs)
       
        self.params = []
        
        self.z_curr=self.opti.parameter(2)
        self.u_prev=self.opti.parameter(1)
        self.s0 = self.opti.parameter(1)
        
        self.params+=[self.z_curr, self.u_prev]
        
        self.z_lin=self.opti.parameter(2,self.N+1) #[s,v] of ego
        self.x_pos=self.opti.parameter(2,self.N+1) #[x,y] of ego    
        self.dpos =[self.opti.parameter(2,1) for _ in range(self.N)]

        self.params+=[ca.vec(self.z_lin), ca.vec(self.x_pos), ca.vec(ca.horzcat(*self.dpos))]

        self.l1_lmbd=self.config['l1_lmbd']
        self.V_MAX = self.opti.parameter(1)
        self.z_tv_curr=[self.opti.parameter(2) for _ in range(self.N_TV)] #[s,v] of TV
        self.u_tvs=[[self.opti.parameter(self.N,1) for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
        self.pos_tvs=[[self.opti.parameter(2,self.N+1) for _ in range(self.N_modes[k])] for k in range(self.N_TV)] # [x,y] of TV
        self.dpos_tvs=[[[self.opti.parameter(2,1) for _ in range(self.N)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
        self.Qs=[[[self.opti.parameter(2,2) for _ in range(self.N)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
        self.psi_tvs = [[self.opti.parameter(1,self.N+1) for _ in range(self.N_modes[k])] for k in range(self.N_TV)] # TV heading
        self.tv_params = [self.opti.parameter(2) for _ in range(self.N_TV)] # TV length and width
        self.params+=[self.z_tv_curr, self.u_tvs, self.pos_tvs, self.dpos_tvs, self.Qs, self.psi_tvs, self.tv_params,self.s0]

        if not self.offline:
            self.gain_keep=[[self.opti.parameter(self.N-1,1) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
            self.constr_keep=[[self.opti.parameter(self.N-1,1) for j in range(len(self.mode_map))] for k in range(self.N_TV)]

        # self.params +=[self.gain_keep, self.constr_keep]

        self.params = ca.vertcat(*_flatten2ca(self.params))  
        
        self.policy=self._return_policy_class()
        self._add_constraints_and_cost()
        
        self._update_ev_initial_condition(np.array([0., 2.]), 0.)
        self._update_ev_preds(np.ones((2,self.N+1)), 50*np.ones((2,self.N+1)), [np.ones((2,1))]*self.N)

        self._update_tv_initial_condition([np.array([0., 0.])]*self.N_TV)
        self._update_tv_preds([[np.zeros((self.N,1))]*self.N_modes[k] for k in range(self.N_TV)], [[np.zeros((2,self.N+1))]*self.N_modes[k] for k in range(self.N_TV)], 
                              [[[np.ones((2,1))]*self.N]*self.N_modes[k] for k in range(self.N_TV)], [[[np.eye(2)]*self.N]*self.N_modes[k] for k in range(self.N_TV)])
        self._update_tv_psi([[np.zeros((1,self.N+1))]*self.N_modes[k] for k in range(self.N_TV)])
        self._update_tv_params([[4.47, 2]]*self.N_TV)
        self._update_red_light(None,None)
        self._update_speed_limit(self.config['v_max'])

        if not self.offline: 
            _,_,_,_ = self._test_update_gain_and_constr_keeps()  
        self.solve(first_solve=True)

    def _return_policy_class(self):

        """
        EV Affine disturbance feedback + TV state feedback policies from https://arxiv.org/abs/2109.09792
        """ 
        h0=self.opti.variable(1)
        if self.config['collision_avoidance_method'] == 'obca':
            self.obca_lmbd = [[self.opti.variable(4, self.N-1) for _ in range(self.N_modes[k])] for k in range(self.N_TV)] #Assuming rectangular obstacles
            self.obca_lmbd_redlight = self.opti.variable(4, self.N-1)
        # if self.config['collision_avoidance_method'] == 'affine':
        self.slack = [[[self.opti.variable(1) for _ in range(self.N-1)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
        # else:
        #     self.slack = self.opti.variable(1)
        self.redlight = self.opti.parameter(2,1)
        # Uncomment next line for disturbance feedback when using Gurobi. 
        # Runs slow with Ipopt (default)
        # M=[[[self.opti.variable(1, 2) for n in range(t)] for t in range(self.N)] for j in range(self.N_modes)]
        M=[[ca.DM(1, 2) for n in range(t)] for t in range(self.N)] #set to 
        h=[self.opti.variable(1) for t in range(self.N-1)]
        if self.open_loop:
            K=[[[ ca.DM(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        else:
            if not self.offline:
                K4screening=[[[self.opti.variable(1,2) for t in range(self.N-1)] for j in range(len(self.mode_map))] for k in range(self.N_TV)]
                K=[[[ca.if_else(self.gain_keep[k][j][t], K4screening[k][j][t], ca.MX.zeros(1, 2), True) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                # K=[[[self.opti.variable(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                # h=[[self.opti.variable(1) for t in range(self.N-1)] for j in range(m.prod(self.N_modes))]
            else: 
                K=[[[self.opti.variable(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                # K=[[[K[k][j][t] if t%2 == 0 else K[k][j][t-1] for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                self.gain_l1=[[[self.opti.variable(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        # h=[[self.opti.variable(1) for t in range(self.N-1)] for j in range(m.prod(self.N_modes))]            
        h_stack=ca.vertcat(h0,*[h[t] for t in range(self.N-1)])
        # M_stack=[ca.vertcat(*[ca.horzcat(*[M[j][t][n] for n in range(t)], ca.DM(1,2*(self.N-t))) for t in range(self.N)]) for j in range(self.N_modes)]
        # h_stack=[ca.vertcat(h0,*[h[j][t] for t in range(self.N-1)]) for j in range(m.prod(self.N_modes))]
        M_stack=ca.vertcat(*[ca.horzcat(*[M[t][n] for n in range(t)], ca.DM(1,2*(self.N-t))) for t in range(self.N)])
        K_stack=[[ca.diagcat(ca.DM(1,2),*[K[k][j][t] for t in range(self.N-1)]) for j in range(self.N_modes[k])] for k in range(self.N_TV)] 

        if self.config['collision_avoidance_method'] == 'obca':
            # self.vars_pol = ca.vertcat(h_stack, self.slack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K[k][j][t]) for t in range(self.N-1)], ca.vec(self.obca_lmbd[k][j])) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]))
            self.vars_pol = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K[k][j][t]) for t in range(self.N-1)], ca.vec(self.obca_lmbd[k][j])) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]))
        else:
            #Affine
            self.slack_vec = ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[self.slack[k][j][t] for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)])
            self.vars_pol = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]))
        self.slack_vec = ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[self.slack[k][j][t] for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)])
        if self.offline:
            self.vars_epi = ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(self.gain_l1[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)])
        else:
            if self.config['collision_avoidance_method'] == 'obca':
                self.vars_pol4screening = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K4screening[k][j][t]) for t in range(self.N-1)], ca.vec(self.obca_lmbd[k][j])) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]))
                # self.vars_pol4screening = ca.vertcat(h_stack, self.slack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K4screening[k][j][t]) for t in range(self.N-1)], ca.vec(self.obca_lmbd[k][j])) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]))
            else:
                self.vars_pol4screening = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K4screening[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]))
        self.vars_ws, self.vars_epi_ws  = None, None 
        return h_stack,M_stack,K_stack

        
    def _get_ATV_TV_dynamics(self):
        """
        Constructs system matrices such that for mode j and for TV k,
        O_t=T_tv@o_{t|t}+c_tv+E_tv@N_t
        where
        O_t=[o_{t|t}, o_{t+1|t},...,o_{t+N|t}].T, (TV state predictions)
        N_t=[n_{t|t}, n_{t+1|t},...,n_{t+N-1|t}].T,  (TV process noise sequence)
        o_{i|t}= state prediction of kth vehicle at time step i, given current time t
        """ 

        T_tv=[[ca.DM(2*(self.N+1), 2) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        TB_tv=[[ca.DM(2*(self.N+1), self.N) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        c_tv=[[ca.DM(2*(self.N+1), 1) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        E_tv=[[ca.DM(2*(self.N+1),self.N*2) for j in range(self.N_modes[k])] for k in range(self.N_TV)]

        u_tvs=self.u_tvs

        for k in range(self.N_TV):
            E=ca.diag(self.tv_n_std[k])
            for j in range(self.N_modes[k]):
                for t in range(self.N+1):
                    if t==0:
                        T_tv[k][j][:2,:]=ca.DM.eye(2)
                    else:
                        T_tv[k][j][t*2:(t+1)*2,:]=self.Atv@T_tv[k][j][(t-1)*2:t*2,:]
                        TB_tv[k][j][t*2:(t+1)*2,:]=self.Atv@TB_tv[k][j][(t-1)*2:t*2,:]
                        TB_tv[k][j][t*2:(t+1)*2,t-1:t]=self.Btv
                        E_tv[k][j][t*2:(t+1)*2,:]=self.Atv@E_tv[k][j][(t-1)*2:t*2,:]    
                        E_tv[k][j][t*2:(t+1)*2,(t-1)*2:t*2]=E #*(t/2)#* (t/3 if t < int(3*self.N/2) else int(self.N/2)/3)

                c_tv[k][j]=TB_tv[k][j]@u_tvs[k][j]             

        return T_tv, c_tv, E_tv


    def _get_LTV_EV_dynamics(self):
        """
        Constructs system matrices such for EV,
        X_t=A_pred@x_{t|t}+B_pred@U_t+E_pred@W_t
        where
        X_t=[x_{t|t}, x_{t+1|t},...,x_{t+N|t}].T, (EV state predictions)
        U_t=[u_{t|t}, u_{t+1|t},...,u_{t+N-1|t}].T, (EV control sequence)
        W_t=[w_{t|t}, w_{t+1|t},...,w_{t+N-1|t}].T,  (EV process noise sequence)
        x_{i|t}= state prediction of kth vehicle at time step i, given current time t
        """ 
            
        E=ca.diag(self.ev_n_std)
        
        A_pred=ca.DM(2*(self.N+1), 2)
        B_pred=ca.DM(2*(self.N+1),self.N)
        E_pred=ca.DM(2*(self.N+1),self.N*2)
        
        A_pred[:2,:]=ca.DM.eye(2)
        
        for t in range(1,self.N+1):
                A_pred[t*2:(t+1)*2,:]=self.A@A_pred[(t-1)*2:t*2,:]
                
                B_pred[t*2:(t+1)*2,:]=self.A@B_pred[(t-1)*2:t*2,:]
                B_pred[t*2:(t+1)*2,t-1]=self.B
                
                E_pred[t*2:(t+1)*2,:]=self.A@E_pred[(t-1)*2:t*2,:]
                E_pred[t*2:(t+1)*2,(t-1)*2:t*2]=E
                
        
        return A_pred,B_pred,E_pred
        
    def _add_constraints_and_cost(self):
        """
        Constructs obstacle avoidance, state-input constraints for Stochastic MPC, based on https://arxiv.org/abs/2109.09792
        """        
        [A,B,E]=self._get_LTV_EV_dynamics()
        [T_tv,c_tv,E_tv]=self._get_ATV_TV_dynamics()
        [h,M,K]=self.policy

        self.nom_z_tv=[[T_tv[k][j]@self.z_tv_curr[k]+c_tv[k][j]  for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        
        
        cost = 0
        self.opti.subject_to(self.opti.bounded(self.V_MIN, A[[t*2+1 for t in range(1,self.N+1)],:]@self.z_curr+B[[t*2+1 for t in range(1,self.N+1)],:]@h, self.V_MAX))
        self.opti.subject_to(self.opti.bounded(self.A_MIN, h, self.A_MAX))
        
        nom_z=A@self.z_curr+B@h
        self.nom_z = nom_z
        nom_s=ca.vec(nom_z.reshape((2,-1))[0,:])
        nom_z_diff=ca.vec(ca.diff(nom_z.reshape((2,-1)),1,1))
        # cost+=-2.7*self.Q_cost*ca.sum1(nom_s) +2.*self.Q_cost*nom_z_diff.T@nom_z_diff# penalizes slow progress (was -2.5, 2)
        cost += -0.1*self.Q_cost*ca.sum1(nom_s) + 0.2*self.Q_cost*nom_z_diff.T@nom_z_diff# penalizes slow progress (was -4, 3.5)
        cost += self.R_cost*0.02*ca.diff(ca.vertcat(self.u_prev,h),1,0).T@ca.diff(ca.vertcat(self.u_prev,h),1,0) # penalizes large input rates
        if self.offline:
            self.lin_ineq_l1 = []
            self.ca_ineq = []
            self.l1_constr=[[[ [] for _ in range(self.N-1)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
            self.ca_constr=[[[ [] for _ in range(self.N-1)] for _ in range(len(self.mode_map))] for _ in range(self.N_TV)]
            self.test = [[[ [] for _ in range(self.N-1)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
            self.test2 = [[[ [] for _ in range(self.N-1)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
            self.test3 = [[[ [] for _ in range(self.N-1)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)]

        # OBCA for redlight
        if self.config['collision_avoidance_method'] == 'obca':
            redlight_obca_lmbd = self.obca_lmbd_redlight
            d_min_red = 0
            
            obca_lmbd = self.obca_lmbd
            d_min = 0.8*(self.ev_length/2)
            # d_min = 0.8
            for t in range(1,self.N):
                ego_psi = self.route(nom_s[t] + self.s0)[2]
                Rev = ca.vertcat(
                    ca.horzcat(ca.cos(ego_psi), -ca.sin(ego_psi)),
                    ca.horzcat(ca.sin(ego_psi), ca.cos(ego_psi))
                )
                #Rotation matrix
                R_mk = Rev
                A_m = ca.DM([[1,0],[-1,0],[0,1],[0,-1]]) @ R_mk.T
                tv_nom = self.redlight #(2x1)
                b_m = ca.vertcat(0.5/2, 0.5/2,2/2,2/2) + A_m @ tv_nom #Artibrary length = 0.5 m, width = 4 m to represent a stop line
                pt = self.route(nom_s[t] + self.s0)[:2]
                y = -d_min_red + (A_m @ pt - b_m).T @ redlight_obca_lmbd[:,t-1]
                self.opti.subject_to(0<=y)
                self.opti.subject_to((A_m.T @ redlight_obca_lmbd[:,t-1]).T @(A_m.T @ redlight_obca_lmbd[:,t-1]) <= 1)
                self.opti.subject_to(redlight_obca_lmbd[:,t-1] >= 0)
        elif self.config['collision_avoidance_method'] == 'affine':
            for t in range(1,self.N):
                self.opti.subject_to(self.opti.bounded(0,nom_s[t],self.redlight[0]-1))
        else:  
            raise ValueError(f"Unknown collision avoidance method: {self.config['collision_avoidance_method']}")
        for k in range(self.N_TV):
            for j in range(len(self.mode_map)):
                m=self.mode_map[j][k]
                cost += 0.03*ca.trace(K[k][m]@E_tv[k][m][:2*self.N,:]@E_tv[k][m][:2*self.N,:].T@K[k][m].T)
                for t in range(1, self.N):  # position at time-step 1 not a function of decision variables 
                    if self.config['collision_avoidance_method'] == 'obca':
                        '''
                        OBCA constraints
                        '''
                        # Optimization-based collision avoidance constraints. Ego: Point-mass, Obstacle: Polytope
                                     
                        #first term is the ego vehicle's position noise, second term: is the TV's position
                        tv_nom = self.pos_tvs[k][m][:,t] #relative to the ego's initial position
                        tv_w = ca.horzcat(ca.DM(2,2*self.N),*[self.dpos_tvs[l][m][t-1]@E_tv[l][m][2*t,:] if l==k else ca.DM(2,2*self.N) for l in range(self.N_TV)]) #E_tv: (1x30)

                        # Rotation matrix of the target vehicle
                        Rtv = ca.vertcat(
                            ca.horzcat(ca.cos(self.psi_tvs[k][m][t]), -ca.sin(self.psi_tvs[k][m][t])),
                            ca.horzcat(ca.sin(self.psi_tvs[k][m][t]), ca.cos(self.psi_tvs[k][m][t]))
                        )

                        A_m = ca.DM([[1,0],[-1,0],[0,1],[0,-1]]) @ Rtv.T #A_m: (4x2), Rtv: (2x2), Rotating the polytope
                        b_m = ca.vertcat(self.tv_params[k][0]/2, self.tv_params[k][0]/2,self.tv_params[k][1]/2,self.tv_params[k][1]/2) + A_m @ tv_nom
                        b_m_w = A_m @ tv_w

                        pt = self.route(nom_s[t] + self.s0)[:2] #[x,y] coordinate of the ego vehicle at timestep t(2,1)
                        # pt = self.x_pos[:,t] + self.dpos[t-1]@(A[2*t,:]@self.z_curr+B[2*t,:]@h - self.z_lin[0,t])

                        pt_w = ca.horzcat(self.dpos[t-1]@(B[2*t,:]@M+E[2*t,:]),*[self.dpos[t-1]@B[2*t,:]@K[l][self.mode_map[j][l]]@E_tv[l][self.mode_map[j][l]][:2*self.N,:] for l in range(self.N_TV)])
                        # pt_C = ca.norm_2(self.dpos[t-1]@E[2*t,:])

                        pt_w_z = ca.horzcat(self.dpos[t-1]@(E[2*t,:]),*[self.dpos[t-1]@B[2*t,:]@ca.DM(*K[l][self.mode_map[j][l]].shape)@E_tv[l][self.mode_map[j][l]][:2*self.N,:] for l in range(self.N_TV)])

                        # Tightening
                        # -self.tight*||Am@(ptw-bmw)@obca_lmbd[k][:,t-1]||_2 >= d_min-[Am@pt-bm.T @ obca_lmbd[k][:,t-1]]
                        # z = self.tight**(0.5)*(A_m @ pt_w).T @ obca_lmbd[k][:,t-1] #(180x1)
                        # A_m: (4x2), pt_w: (2x180), b_m_w: (4x30), obca_lmbd: (4x1)
                        # z = 1/ca.sqrt(2)*self.tight*ca.norm_fro(A_m @ pt_w - b_m_w) #worst-case effect of noise on y
                        z = (1/ca.sqrt(2))*ca.sqrt(ca.sumsqr(A_m @ pt_w - b_m_w))

                        self.test[k][m][t-1] = z
                        # pdb.set_trace()
                        # z = self.tight *(A_m @ pt_w - b_m_w).T @ obca_lmbd[k][m][:,t-1] #((N-1)*x1) -- z is quadratic in theta
                        y = -d_min + (A_m @ pt - b_m).T @ obca_lmbd[k][m][:,t-1] #+ 0 + 1e-12*obca_lmbd[k][:,t-1].T@obca_lmbd[k][:,t-1]  #(1x1) Nominal 
                        self.test2[k][m][t-1] = y
                        
            
                        # Ego Frenet-to-Cartesian Jacobian
                        psi_ego = self.route(nom_s[t] + self.s0)[2]
                        J_ego = ca.vertcat(
                            ca.horzcat(ca.cos(psi_ego), 0),
                            ca.horzcat(ca.sin(psi_ego), 0)
                        )
                        Sigma_ev_sv = ca.diag(ca.DM(self.ev_n_std)**2)
                        Sigma_ev_xy = J_ego @ Sigma_ev_sv @ J_ego.T

                        # TV Frenet-to-Cartesian Jacobian
                        psi_tv = self.psi_tvs[k][m][0, t]
                        J_tv = ca.vertcat(
                            ca.horzcat(ca.cos(psi_tv), 0),
                            ca.horzcat(ca.sin(psi_tv), 0)
                        )
                        Sigma_tv_sv = ca.diag(ca.DM(self.tv_n_std[k])**2)
                        Sigma_tv_xy = J_tv @ Sigma_tv_sv @ J_tv.T

                        # Combined uncertainty in Cartesian
                        Q_cov = Sigma_ev_xy + Sigma_tv_xy
                        Q_m = A_m @ Q_cov @ A_m.T

                        # z = self.tight * ca.sqrt(obca_lmbd[k][m][:,t-1].T @ Q_m @ obca_lmbd[k][m][:,t-1] + 1e-4)

                        #Linearize about obca_lmbd = 0.01*ca.DM(4,1)
                        # z_lin = self.tight*(A_m @ pt_w_z - b_m_w).T @ (0.01*ca.DM.ones(*obca_lmbd[k][m][:,t-1].shape)) + self.tight*(A_m.T@(obca_lmbd[k][m][:,t-1] - 0.01*ca.DM.ones(*obca_lmbd[k][m][:,t-1].shape)) ) + self.tight*(A_m @(pt_w - pt_w_z))
                        # z_lin = self.tight * (A_m @ pt_w_z - b_m_w).T @ (0.01*ca.DM.ones(*obca_lmbd[k][m][:,t-1].shape)) + self.tight*((A_m @ (pt_w - pt_w_z)).T@(0.01*ca.DM.ones(*obca_lmbd[k][m][:,t-1].shape))) # This is the linearized z term, which is a function of the previous lambda and the noise term.
                        # y_lin = -d_min + (A_m @ self.route(self.z_lin[0,t] + self.s0)[:2] - b_m).T @ (0.01*ca.DM.ones(*obca_lmbd[k][m][:,t-1].shape)) + (A_m@ (pt - self.route(self.z_lin[0,t] + self.s0)[:2])).T @ (0.01*ca.DM.ones(*obca_lmbd[k][m][:,t-1].shape)) + (A_m@self.route(self.z_lin[0,t] + self.s0)[:2] - b_m).T@(obca_lmbd[k][m][:,t-1] - 0.01*ca.DM.ones(*obca_lmbd[k][m][:,t-1].shape)) #Nominal

                        #Use linearized z and y for the collision avoidance constraint
                        # z = z_lin
                        # y = y_lin
                        # y = -d_min + (A_m @ (pt - tv_nom)-ca.vertcat(self.tv_params[k][0]/2, self.tv_params[k][0]/2,self.tv_params[k][1]/2,self.tv_params[k][1]/2)).T @ obca_lmbd[k][:,t-1] + 0 + 1e-12*obca_lmbd[k][:,t-1].T@obca_lmbd[k][:,t-1]  #(1x1)
                        # self.ca_constr[k][j][t-1]+=[z.T@z<=y**2, 0<=y]
                        # self.ca_constr[k][j][t-1]+=[ca.sqrt(z.T@z + 1e-4)<=y, 0<=y]
                        # self.ca_ineq.append(ca.vertcat(z,y))

                        #slack cost
                        # cost += 1e5*self.slack**2

                        #obca_lambd cost for PD hessian
                        # cost += 0.05*ca.vec(obca_lmbd[k][m][:,t-1]).T@ca.vec(obca_lmbd[k][m][:,t-1])

                    elif self.config['collision_avoidance_method'] == 'affine':
                        # Linearised obstacle avoidance constraints
                        # EV position projection onto obstacle ellipse
                        # oa_ref=self.pos_tvs[k][m][:,t]
                        diff = self.x_pos[:,t] - self.pos_tvs[k][m][:,t]
                        mahalanobis_norm = np.sqrt(diff.T @ self.Qs[k][m][t-1] @ diff)
                        oa_ref = self.pos_tvs[k][m][:,t] + diff / mahalanobis_norm
                        # oa_ref+=(self.x_pos[:,t]-self.pos_tvs[k][m][:,t])/( (self.x_pos[:,t]-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(self.x_pos[:,t]-self.pos_tvs[k][m][:,t]) )**(0.5)
                        # delta = self.x_pos[:, t] - self.pos_tvs[k][m][:, t]
                        # norm = ca.sqrt(ca.mtimes([delta.T, self.Qs[k][m][t-1], delta]))
                        # oa_ref = self.pos_tvs[k][m][:, t] + delta / norm
                        self.test[k][m][t-1] = diff.T @ self.Qs[k][m][t-1] @ diff
                        self.test2[k][m][t-1] = (oa_ref- self.pos_tvs[k][m][:,t]).T @ self.Qs[k][m][t-1] @ (oa_ref - self.pos_tvs[k][m][:,t])
                        self.test3[k][m][t-1] = (oa_ref - self.pos_tvs[k][m][:,t]).T @ self.Qs[k][m][t-1] @ (self.x_pos[:,t] - oa_ref)
                        
                        # oa_ref+=(self.x_pos[:,0]-self.pos_tvs[k][m][:,t])/((self.x_pos[:,0]-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(self.x_pos[:,0]-self.pos_tvs[k][m][:,t]))**(0.5)
                        # Coefficient of random variables in affine chance constraint
                        z=self.tight*((oa_ref-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(ca.horzcat(self.dpos[t-1]@(B[2*t,:]@M+E[2*t,:]),*[self.dpos[t-1]@B[2*t,:]@K[l][self.mode_map[j][l]]@E_tv[l][self.mode_map[j][l]][:2*self.N,:]-int(l==k)*self.dpos_tvs[k][m][t-1]@E_tv[k][m][2*t,:] for l in range(self.N_TV)]))).T
                        z = ca.sqrt(ca.sumsqr(z))
                        # z=self.tight*((oa_ref-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(ca.horzcat(self.dpos[t-1]@(B[2*t,:]@M+E[2*t,:]),*[self.dpos[t-1]@B[2*t,:]@K[l][self.mode_map[j][l]]@E_tv[l][self.mode_map[j][l]][:2*self.N,:] for l in range(self.N_TV)]))).T
                        # pdb.set_trace()
                        # constant term in affine chance constraint self.slack[k][m][t-1]+
                        y=(oa_ref-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(self.x_pos[:,t]-oa_ref+self.dpos[t-1]*(A[2*t,:]@self.z_curr+B[2*t,:]@h-(self.z_lin[0,t] - self.s0)))
                        # cost += 1e5*self.slack**2

                    else:
                        NotImplementedError("Collision avoidance method not implemented")
                    if self.solver=="ipopt":
                        # norm_2(z)<=y
                        if self.offline:
                            if self.config['collision_avoidance_method'] == 'obca':
                                # self.ca_constr[k][j][t-1]+=[ca.sqrt(z.T@z + 1e-4)<=y, 0<=y]
                                self.ca_constr[k][j][t-1]+=[z<=y+self.slack[k][m][t-1],0<=y]
                                #obca related constraints 
                                self.opti.subject_to((A_m.T @ obca_lmbd[k][m][:,t-1]).T @(A_m.T @ obca_lmbd[k][m][:,t-1]) <= 1)
                                self.opti.subject_to(obca_lmbd[k][m][:,t-1] >= 0)
                            elif self.config['collision_avoidance_method'] == 'affine':
                                # self.ca_constr[k][j][t-1]+=[ca.sqrt(z.T@z+1e-4)<=y, 0<=y]
                                self.ca_constr[k][j][t-1]+=[z<=y+self.slack[k][m][t-1],0<=y]
                            else:
                                NotImplementedError("Collision avoidance method not implemented")

                            self.ca_ineq.append(ca.vertcat(z,y)) # #C -> 1
                            
                            #Impose all ca constraints
                            self.opti.subject_to(self.ca_constr[k][j][t-1][0])
                            self.opti.subject_to(self.ca_constr[k][j][t-1][1])
                            
                            if len(self.l1_constr[k][m][t-1])==0:
                                #self.gain_l1 is the capital psi variable in canonical form
                                # self.test[k][m][t-1] +=[K[k][m][t,2*t:2*(t+1)]]
                                self.l1_constr[k][m][t-1]+=[K[k][m][t,2*t:2*(t+1)]<=self.gain_l1[k][m][t-1], -self.gain_l1[k][m][t-1]<=K[k][m][t,2*t:2*(t+1)]]
                                # tightening the l1 gain constraints (set self.gain_li[k][m][t-1] to be 1 dimensional variable)
                                # self.l1_constr[k][m][t-1]+=[ca.max(K[k][m][t,2*t:2*(t+1)])<=self.gain_l1[k][m][t-1], -self.gain_l1[k][m][t-1]<=ca.min(K[k][m][t,2*t:2*(t+1)])]

                                self.opti.subject_to(self.l1_constr[k][m][t-1][0])
                                self.opti.subject_to(self.l1_constr[k][m][t-1][1])

                                self.lin_ineq_l1+=[K[k][m][t,2*t:2*(t+1)] - self.gain_l1[k][m][t-1]] #g1 constraint
                                # self.lin_ineq_l1+=[-K[k][m][t,2*t:2*(t+1)] - self.gain_l1[k][m][t-1]] #g2 constraint
                                cost += self.l1_lmbd*ca.sum1(ca.vec(self.gain_l1[k][m][t-1]))  
                        else:
                            # collision avoidance constraint screening
                            if self.config['collision_avoidance_method'] == 'obca':
                                # obca related constraint screening
                                obca_constr = ca.vertcat(*[y-ca.sqrt(z.T@z + 1e-4), y])
                                obca_dual_constr = ca.vertcat(1 + self.slack - (A_m.T @ obca_lmbd[k][m][:,t-1]).T @(A_m.T @ obca_lmbd[k][m][:,t-1]),obca_lmbd[k][m][:,t-1])
                                obca_switch=ca.if_else(self.constr_keep[k][j][t-1], obca_constr, ca.DM.ones(*obca_constr.shape))
                                obca_switch_dual=ca.if_else(self.constr_keep[k][j][t-1], obca_dual_constr, ca.DM.ones(*obca_dual_constr.shape))
                                self.opti.subject_to(obca_switch>=0)
                                self.opti.subject_to(obca_switch_dual>=0)
                            elif self.config['collision_avoidance_method'] == 'affine':
                                # soc_constr=ca.vertcat(y,y**2-z@z.T)
                                soc_constr=ca.vertcat(y**2-z.T@z, y)
                                soc_switch=ca.if_else(self.constr_keep[k][j][t-1], soc_constr, ca.DM(*soc_constr.shape), True)
                                self.opti.subject_to(soc_switch>=0)
                            else:
                                NotImplementedError("Collision avoidance method not implemented")
                    else:
                        # Use for SOCP solvers: SCS and Gurobi
                        soc_constr=ca.soc(z,y)
                        if self.offline:
                            self.ca_ineq.append(ca.horzcat(z,y))
                            if len(self.l1_constr[k][m][t-1])==0:
                                self.l1_constr[k][m][t-1]+=[K[k][m][t,2*t:2*(t+1)]<=self.gain_l1[k][m][t-1], -self.gain_l1[k][m][t-1]<=K[k][m][t,2*t:2*(t+1)]]
                                self.opti.subject_to(self.l1_constr[k][m][t-1][0])
                                self.opti.subject_to(self.l1_constr[k][m][t-1][1])

                                self.lin_ineq_l1+=[self.l1_constr[k][m][t-1][0]] #only the first constraint: g1
                                cost += self.l1_lmbd*ca.sum1(ca.vec(self.gain_l1[k][m][t-1]))
                                self.opti.subject_to(soc_constr>0)
                        else:
                            soc_switch=ca.if_else(self.constr_keep[k][j][t-1], soc_constr, ca.DM(*soc_constr.shape), True)
                            self.opti.subject_to(soc_switch>0)

                            # # obca related constraint screening
                            # obca_constr = ca.vertcat(1 + self.slack - (A_m.T @ obca_lmbd[k][m][:,t-1]).T @(A_m.T @ obca_lmbd[k][m][:,t-1]),obca_lmbd[k][m][:,t-1])
                            # obca_switch=ca.if_else(self.constr_keep[k][j][t-1], obca_constr, ca.DM(*obca_constr.shape), True)
                            # self.opti.subject_to(obca_switch>0)
        # if self.config['collision_avoidance_method'] == 'affine': 
        cost += 1e4*self.slack_vec.T@self.slack_vec
        self.opti.minimize( cost ) 

        if self.offline:
            #g(x) <= 0
            self.lin_ineq_constr =[]
            self.lin_ineq_constr+=[A[t*2+1,:]@self.z_curr+B[t*2+1,:]@h-self.V_MAX for t in range(1,self.N+1)]
            self.lin_ineq_constr+=[-A[t*2+1,:]@self.z_curr-B[t*2+1,:]@h + self.V_MIN  for t in range(1,self.N+1)]
            self.lin_ineq_constr+=[h[t]-self.A_MAX  for t in range(self.N)]
            self.lin_ineq_constr+=[-h[t] + self.A_MIN for t in range(self.N)]

            self.f_l_i_c = ca.Function("lin_ineq", [self.vars_pol,self.params, self.V_MAX], self.lin_ineq_constr)

            # F\theta < =f
            self.F, self.f = ca.jacobian(ca.vertcat(*self.f_l_i_c(self.vars_pol,self.params,self.V_MAX)),self.vars_pol), ca.vertcat(*self.f_l_i_c(ca.DM(*self.vars_pol.shape),self.params, self.V_MAX))
            self.f_l_i_l1 =ca.Function("l1_ineq", [self.vars_pol,self.vars_epi],self.lin_ineq_l1)
            # L\theta <= psi
            self.L = ca.jacobian(ca.vertcat(*self.f_l_i_l1(self.vars_pol,self.vars_epi)), self.vars_pol)

            # if self.config['collision_avoidance_method'] == 'obca':
            #     self.f_ca_i = ca.Function("ca_ineq", [self.vars_pol, self.params], self.ca_ineq)
            # else:
            self.f_ca_i = ca.Function("ca_ineq", [self.vars_pol, self.params, self.slack_vec], self.ca_ineq)

            # C\theta + c \in K_1 x K_2 x .................
            # if self.config['collision_avoidance_method'] == 'obca':
            #     self.C =  (ca.jacobian(ca_constr, self.vars_pol) for ca_constr in self.f_ca_i(self.vars_pol, self.params))
            #     self.c = self.f_ca_i(ca.DM(*self.vars_pol.shape), self.params)
            # else:
            self.C =  (ca.jacobian(ca_constr, self.vars_pol) for ca_constr in self.f_ca_i(self.vars_pol, self.params,self.slack_vec))
            self.c = self.f_ca_i(ca.DM(*self.vars_pol.shape), self.params, ca.DM.zeros(*self.slack_vec.shape))
            # if self.config['collision_avoidance_method'] == 'obca': 
            #     self.f_cost = ca.Function("cost", [self.vars_pol, self.vars_epi, self.params, self.slack], [cost])
            #     self.Q, self.p = ca.hessian(self.f_cost(self.vars_pol, self.vars_epi,self.params, 0), self.vars_pol)
            #     self.d         = self.f_cost(ca.DM(*self.vars_pol.shape),ca.DM(*self.vars_epi.shape),self.params,0)
            # else:
            self.f_cost = ca.Function("cost", [self.vars_pol, self.vars_epi, self.params, self.slack_vec], [cost])
            self.Q, self.p = ca.hessian(self.f_cost(self.vars_pol, self.vars_epi,self.params, ca.DM.zeros(*self.slack_vec.shape)), self.vars_pol)
            self.d         = self.f_cost(ca.DM(*self.vars_pol.shape),ca.DM(*self.vars_epi.shape),self.params,ca.DM.zeros(*self.slack_vec.shape))
            # else:
            #     self.C =  (ca.jacobian(ca_constr, self.vars_pol) for ca_constr in self.f_ca_i(self.vars_pol, self.params))
            #     self.c = self.f_ca_i(ca.DM(*self.vars_pol.shape), self.params)

            #     self.f_cost = ca.Function("cost", [self.vars_pol, self.vars_epi, self.params], [cost])
            #     self.Q, self.p = ca.hessian(self.f_cost(self.vars_pol, self.vars_epi,self.params), self.vars_pol)
            #     self.d         = self.f_cost(ca.DM(*self.vars_pol.shape),ca.DM(*self.vars_epi.shape),self.params)
            
        else:
            #Precompute functions for constraints and variable screening
            vars_epi = ca.DM.zeros(2*(self.N-1)*self.prod(self.N_modes)*self.N_TV,1)            
            self.F_fn = ca.Function('F_fn', [self.vars_pol4screening, self.params], [ca.jacobian(ca.vertcat(*self.canon_prob_fn['f_l_i_c'](self.vars_pol4screening,self.params,self.V_MAX)),self.vars_pol4screening)])
            self.L_fn = ca.Function('L_fn', [self.vars_pol4screening], [ca.jacobian(ca.vertcat(*self.canon_prob_fn['f_l_i_l1'](self.vars_pol4screening,vars_epi)), self.vars_pol4screening)])
            self.C_fn = ca.Function('C_fn',[self.params],[ca.substitute(ca.jacobian(ca.simplify(ca.vertcat(*self.canon_prob_fn['f_ca_i'](self.vars_pol4screening, self.params))), self.vars_pol4screening), self.vars_pol4screening, ca.DM.zeros(*self.vars_pol4screening.shape))])
            C = self.C_fn(np.ones(self.params.shape))
            self.C_shape = C.shape
            self.nonzero_inds = np.nonzero(np.ravel(C,order='F'))[0]
            # Get the row and column indices of the non-zero elements in the sparse matrix
            self.row_indices, self.col_indices = np.unravel_index(self.nonzero_inds, self.C_shape,order='F')
            nonzero_vec = ca.vec(ca.substitute(ca.jacobian(ca.simplify(ca.vertcat(*self.canon_prob_fn['f_ca_i'](self.vars_pol4screening, self.params))), self.vars_pol4screening), self.vars_pol4screening, ca.DM.zeros(*self.vars_pol4screening.shape)))[self.nonzero_inds]
            self.C_fn_nonzero = ca.Function('C_fn_nonzero',[self.params],[nonzero_vec])
            self.c_fn = ca.Function('c_fn',[self.params],[ca.vertcat(*self.canon_prob_fn['f_ca_i'](ca.DM.zeros(*self.vars_pol4screening.shape), self.params))])
            self.cost_hessian_fn = ca.Function('cost_hessian_fn', [self.vars_pol4screening, self.params], list(ca.hessian(self.canon_prob_fn['f_cost'](self.vars_pol4screening,vars_epi,self.params), self.vars_pol4screening)))

    def _set_canon_form_mats(self):
        #In online mode, vars_epi is not defined. So, set it to zero
        self.F, self.f = self.F_fn(ca.DM(*self.vars_pol4screening.shape),self.opti.value(self.params)), ca.vertcat(*self.canon_prob_fn['f_l_i_c'](ca.DM(*self.vars_pol4screening.shape),self.opti.value(self.params)))

        self.L = self.L_fn(ca.DM(*self.vars_pol4screening.shape))
        # self.C = self.C_fn(self.opti.value(self.params))
        C = self.C_fn_nonzero(self.opti.value(self.params))
        #construct sparse C matrix
        st = time.time()
        self.C = sp.csr_matrix((np.array(C).reshape(-1), (self.row_indices, self.col_indices)), shape=self.C_shape)
        print(f'Constructing C_sparse took {time.time()-st} seconds')
        # self.C = np.ascontiguousarray(self.C_fn(ca.DM.zeros(*self.vars_pol4screening.shape),self.opti.value(self.params)),dtype=np.float64)
        self.c = self.c_fn(self.opti.value(self.params))

        #Here, ca.hessian outputs hessian, J_grad. 
        #J_grad = Q*theta + p. Thus, if we evaluate J_grad with theta = 0, we get p. Note that hessian is not dependent on theta
        self.Q, self.p = self.cost_hessian_fn(ca.DM.zeros(*self.vars_pol4screening.shape),self.opti.value(self.params))
        # vars_epi = ca.DM.zeros(2*(self.N-1)*self.prod(self.N_modes)*self.N_TV,1)
        # self.d = self.canon_prob_fn['f_cost'](ca.DM(*self.vars_pol4screening.shape),vars_epi,self.opti.value(self.params))

    def solve(self,first_solve=False):
        try:      
            # self.plot_obca_obstacles_and_ego_path(self.pos_tvs, self.psi_tvs, self.tv_params )
            st = time.time()
            sol = self.opti.solve()
            solve_time = time.time() - st
            # print(f'Solve Time: {solve_time}')
            # Collect Optimal solution.
            u_control  = sol.value(self.policy[0][0])
            h_opt      = sol.value(self.policy[0]).squeeze()
            u_opt      = sol.value(self.policy[0]).reshape((1,-1))
            M_opt      = sol.value(self.policy[1])
            K_opt      = [[sol.value(self.policy[2][k][j]) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
            nom_z_tv   = [[sol.value(self.nom_z_tv[k][j]) for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            s0         = sol.value(self.s0)
            nom_z      = sol.value(self.nom_z).reshape(-1,2).T
            nom_z[0,:] += s0 
            if self.offline and not first_solve:
                self.vars_ws , self.vars_epi_ws = sol.value(self.vars_pol), sol.value(self.vars_epi)
            if self.offline and self.solver=='ipopt':
                l1_duals=[[[[sol.value(self.opti.dual(self.l1_constr[k][j][t][0]))] for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)]
                #TODO: if g1_inf norm prediction
                # l1_duals=[[[[np.linalg.norm(sol.value(self.opti.dual(self.l1_constr[k][j][t][0])),ord=np.inf)] for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)]
                ca_duals=[[[[sol.value(self.opti.dual(self.ca_constr[k][j][t][0]))] for t in range(self.N-1)] for j in range(len(self.mode_map))] for k in range(self.N_TV)]
            is_opt     = True
            # Debug checking for non-zero gains K
            # eps = 1e-3
            # for k in range(self.N_TV):
            #     for j in range(len(self.mode_map)):
            #         m=self.mode_map[j][k]
            #         for t in range(1,self.N):
            #             if np.max(np.abs(self.opti.value(self.test[k][m][t-1][0]))) > eps:
            #                 print('Non-zero gain found in test')
            #                 pdb.set_trace()

            # eigs = np.linalg.eig(sol.value(self.Q).toarray())
            # largest_eig = np.max(eigs[0])
            # smllest_eig = np.min(eigs[0])
            # print(f'Largest Eigenvalue: {largest_eig}')
            # print(f'Smallest Eigenvalue: {smllest_eig}')
            # print(eigs[0])
            # pdb.set_trace()
        except:
            # self.opti.debug.show_infeasibilities()
            # t=0
            # i=0
            # m=0
            # plot_collision_linearization(
            #             Q=self.opti.debug.value(self.Qs[i][m][t]),
            #             c=self.opti.debug.value(self.pos_tvs[i][m][:, t]),
            #             x=self.opti.debug.value(self.x_pos[:, t])
            # )
            # pdb.set_trace()

            if self.offline:
                self.vars_ws , self.vars_epi_ws = None, None  
            infeas_status = ['Infeasible_Problem_Detected'] if self.solver=="ipopt" else ["INF_OR_UNBD"]
            if self.opti.stats()['return_status'] not in infeas_status:
              # Suboptimal solution (e.g. timed out)
                print('Solver timed out. Yielding a suboptimal solution...')
                u_control=self.opti.debug.value(self.policy[0][0])   
                u_opt = self.opti.debug.value(self.policy[0]).reshape((1,-1))
                s0 = self.opti.debug.value(self.s0)
                nom_z = self.opti.debug.value(self.nom_z).reshape((-1,2)).T
                nom_z[0,:] += s0
            else:
                u_control  = self.u_backup
                u_opt = np.array([self.u_backup]*(self.N-1)).reshape((1,-1))
                accumulated_dynamics = []
                for t in range(self.N+1):
                    temp = 0
                    for offset in range(t):
                        temp += np.linalg.matrix_power(self.A,t-offset)@self.B*self.u_backup if t-offset > 0 else 0
                    accumulated_dynamics.append(temp)
                s0 = self.opti.debug.value(self.s0)
                nom_z = np.hstack([sum(x) for x in zip([np.linalg.matrix_power(self.A,t)@self.x0 for t in range(self.N+1)], accumulated_dynamics)]) #maximum braking
            t_proc_sum = sum(value for key, value in self.opti.stats().items() if key.startswith('t_proc'))
            t_wall_sum = sum(value for key, value in self.opti.stats().items() if key.startswith('t_wall'))
            solve_time = sum(value for key, value in self.opti.stats().items() if key.startswith('t_wall_solver')) if self.solver == 'grb' else t_wall_sum
            is_opt = False
        t_proc_sum = sum(value for key, value in self.opti.stats().items() if key.startswith('t_proc'))
        t_wall_sum = sum(value for key, value in self.opti.stats().items() if key.startswith('t_wall'))
        # solve_time = sum(value for key, value in self.opti.stats().items() if key.startswith('t_wall_solver')) if self.solver == 'grb' else t_wall_sum
        sol_dict = {}
        sol_dict['nom_z']      = nom_z      # nominal state predictions
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['u_opt']      = u_opt      # optimal control sequence
        sol_dict['optimal']    = is_opt      # whether the solution is optimal or not
        if is_opt:
            sol_dict['h_opt']=h_opt
            sol_dict['M_opt']=M_opt
            sol_dict['K_opt']=K_opt
            sol_dict['nom_z_tv']=nom_z_tv

            if self.offline and self.solver == 'ipopt':
                sol_dict['l1_duals']=l1_duals
                sol_dict['ca_duals']=ca_duals
                
                
        sol_dict['solve_time'] = solve_time  # how long the solver took in seconds
        print(f'Optimization Solve Time [{self.solver}]: {solve_time} s')
        sol_dict['t_wall_sum'] = t_wall_sum
        sol_dict['t_proc_sum'] = t_proc_sum
        sol_dict['vars'] = self.vars_kept
        sol_dict['constr'] = self.constr_kept
        #reconstruct self.constr_keep lists
        if not self.offline:
            gain_keep=[[self.opti.value(self.gain_keep[k][j]) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
            constr_keep=[[self.opti.value(self.constr_keep[k][j]) for j in range(len(self.mode_map))] for k in range(self.N_TV)]
            print(f'Gain Keep: {gain_keep}')
            print(f'Constr Keep: {constr_keep}')

        return sol_dict

    def check_update_dict(self,update_dict):
        assert 'x0' in update_dict.keys(), 'Missing EV Initial Condition'
        assert 'preds' in update_dict.keys(), 'Missing TV Predictions'
        assert 'u_prev' in update_dict.keys(), 'Missing EV Previous Control'
        assert 'z_lin' in update_dict.keys(), 'Missing EV Linearised Predictions'
        assert 'x_pos' in update_dict.keys(), 'Missing EV Position Predictions'
        assert 'dpos' in update_dict.keys(), 'Missing EV Process Noise Predictions'
        assert 'u_tvs' in update_dict.keys(), 'Missing TV Controls'
        assert 'o_glob' in update_dict.keys(), 'Missing TV Global Position Predictions'
        assert 'droutes' in update_dict.keys(), 'Missing TV Process Noise Predictions'
        assert 'Qs' in update_dict.keys(), 'Missing TV Process Noise Covariances'
        assert 'o0' in update_dict.keys(), 'Missing TV Initial Condition'
        assert 'tv_psi' in update_dict.keys(), 'Missing TV Heading Predictions'
        assert 'tv_params' in update_dict.keys(), 'Missing TV Parameters'

        if not self.offline:
            assert 'l1_duals' in update_dict.keys(), 'Missing L1 Duals'
            assert 'canon_prob' in update_dict.keys(), 'Missing Canonical Problem'

    def update(self, update_dict):
        self.check_update_dict(update_dict)
        self._update_ev_initial_condition(*[update_dict[key] for key in ['x0', 'u_prev']] )
        self._update_tv_initial_condition(*[update_dict[key] for key in ['o0']] )
        self._update_ev_preds(update_dict['z_lin'], update_dict['x_pos'], update_dict['dpos'])
        self._update_tv_preds(update_dict['u_tvs'], update_dict['o_glob'],
                                update_dict['droutes'], update_dict['Qs'])
        self._update_tv_psi(update_dict['tv_psi'])
        self._update_tv_params(update_dict['tv_params'])
        self._update_red_light(update_dict['red_light'],update_dict['ego_sim_initial_state'])
        self._update_speed_limit(update_dict['speed_limit'])
        if not self.offline:
            if 'l1_duals' in update_dict.keys():
                if 'canon_prob' in update_dict.keys():
                    print('[smpc.py]: Calling set_canon_form_mats,,,')
                    st = time.time()
                    self._set_canon_form_mats()
                    solve_time = time.time()-st
                    print(f'[smpc.py]: set_canon_form_mats took {solve_time} seconds')
                    print('[smpc.py]: Finished set_canon_form_mats,,,')
                print('[smpc.py]: Calling update_gain_and_constr_keeps')
                st = time.time()
                _,_,_,_= self._test_update_gain_and_constr_keeps(*[update_dict[key] for key in ['l1_duals', 'ca_duals']])
                solve_time = time.time()-st
                print(f'[smpc.py]: update_gain_and_constr_keeps took {solve_time} seconds')
                print('[smpc.py]: Finished update_gain_and_constr_keeps,,,')
            else:
                self._test_update_gain_and_constr_keeps()
        # elif self.offline and (self.vars_ws is not None) and (self.vars_epi_ws is not None):
        #     # print('warm starting'.center(80,'#'))
        #     self.opti.set_initial(self.vars_pol,self.vars_ws)
        #     self.opti.set_initial(self.vars_epi,self.vars_epi_ws)

    def _update_speed_limit(self, speed_limit=None):
        if speed_limit is None:
            self.opti.set_value(self.V_MAX, self.config['v_max'])
        else:
            sl = min(speed_limit,self.config['v_max'])
            print(f"[smpc.py] Speed Limit: {speed_limit}, SL: {sl}")
            self.opti.set_value(self.V_MAX, min(speed_limit,self.config['v_max']))

    def _update_red_light(self, red_light_agent = None,ego_sim_init_state=None):
        if self.config['collision_avoidance_method'] == 'obca':
            if red_light_agent is None:
                self.opti.set_value(self.redlight, [1000,1000]) #default red light positon (x,y), really far away from ego
            else:
                red_light = [red_light_agent.x-ego_sim_init_state.center.point.x,red_light_agent.y-ego_sim_init_state.center.point.y] #global x and y of red light agent
                self.opti.set_value(self.redlight, red_light)
        elif self.config['collision_avoidance_method'] == 'affine':
            if red_light_agent is None:
                self.opti.set_value(self.redlight, [1e6,0]) #[s,v]
            else:
                red_light = [red_light_agent.progress,0] #s and v of red light agent
                self.opti.set_value(self.redlight, red_light) #[s,v]

    def _update_ev_initial_condition(self, x0, u_prev):
        self.x0 = x0
        print(f'[smpc.py] Initial Velocity: {x0[1]}')
        self.opti.set_value(self.s0, x0[0])    
        self.opti.set_value(self.z_curr, np.array([0,x0[1]],dtype=np.float64))
        self.opti.set_value(self.u_prev, u_prev)

        self.u_backup=self.A_MIN
                  
    def _update_tv_initial_condition(self, x_tv0):
        for k in range(self.N_TV):
            self.opti.set_value(self.z_tv_curr[k], x_tv0[k])

    def _update_ev_preds(self, z_lin, x_pos, dpos):
        self.opti.set_value(self.z_lin, z_lin)
        self.opti.set_value(self.x_pos, x_pos)
        for  t in range(self.N):
            self.opti.set_value(self.dpos[t],dpos[t])
    
    def _update_tv_preds(self, u_tvs, pos_tvs, dpos_tvs, Qs):
        for k in range(self.N_TV):
            for m in range(self.N_modes[k]):
                self.opti.set_value(self.pos_tvs[k][m], pos_tvs[k][m])
                self.opti.set_value(self.u_tvs[k][m], u_tvs[k][m].reshape((-1,1)))
                for t in range(self.N):
                    self.opti.set_value(self.dpos_tvs[k][m][t],dpos_tvs[k][m][t])
                    self.opti.set_value(self.Qs[k][m][t],Qs[k][m][t])
                   
    def _update_tv_psi(self, psi_tvs):
        for k in range(self.N_TV):
            for m in range(self.N_modes[k]):
                self.opti.set_value(self.psi_tvs[k][m], psi_tvs[k][m])

    def _update_tv_params(self, tv_params):
        for k in range(self.N_TV):
            self.opti.set_value(self.tv_params[k], tv_params[k])

    # def _set_canon_form_mats(self, canon_prob):
    #     self.Q, self.p, self.d = canon_prob["Q"],canon_prob["p"], canon_prob["d"]
    #     self.F, self.f = canon_prob["F"], canon_prob["f"]
    #     self.L = canon_prob["L"]
    #     self.C, self.c = canon_prob["C"], canon_prob["c"]

    def prod(self,val) :
        res = 1
        for ele in val:
            res *= ele
        return res
    

    def _get_canon_form_fns(self):
        '''
        Returns dictionary containing canonical form of the problem (offline mode only)
        '''
        canon_prob_fn ={}
        canon_prob_fn.update({'f_l_i_c':self.f_l_i_c, 'f_l_i_l1': self.f_l_i_l1, 'f_ca_i':self.f_ca_i, 'f_cost':self.f_cost})
        return canon_prob_fn
    
    def _get_canon_form_fns_precomputed(self):
        '''
        Returns dictionary containing canonical form of the problem (offline mode only). Call in
        '''
        canon_prob_fn ={}
        if not self.offline:
            canon_prob_fn.update({'F_fn':self.F_fn, 'L_fn': self.L_fn, 'C_fn_nonzero':self.C_fn_nonzero, 'c_fn':self.c_fn, 'cost_hessian_fn':self.cost_hessian_fn,'row_indices':self.row_indices,'col_indices':self.col_indices,'C_shape':self.C_shape,'vars_pol4screening_shape':self.vars_pol4screening.shape})
        return canon_prob_fn
    
    def _get_canon_form_mats(self):
        '''
        Returns dictionary containing canonical form of the problem 
        '''
        canon_prob ={}
        canon_prob.update({"Q":self.opti.value(self.Q), "p":self.opti.value(self.p), "d":self.opti.value(self.d), 
                           "F":self.opti.value(self.F), "f":self.opti.value(self.f), "L":self.opti.value(self.L), 
                        #    "C":self.opti.value(ca.vertcat(*self.C)), "c":self.opti.value(ca.vertcat(*self.c))}) #self.C and self.c are generator and tuple
                        "C":[self.opti.value(ca.vertcat(C)) for C in self.C], "c":[self.opti.value(ca.vertcat(c)) for c in self.c]}) 
        return canon_prob

    def _test_update_gain_and_constr_keeps(self, l1_duals=None, ca_duals=None):
        if l1_duals is None:
            mu, eta, g1, gap = None, None, None, None
        else:
            # Selection matrix for the "non-zero" mu and g1 duals
            # S1 = np.diag(ca_duals) #if mu_tilde=1, then mu!=0. elif mu_tilde=0, then mu = 0
            mu_dim = 2*self.N * (self.N_TV+1) + 1
            # S1= np.kron(np.diag(ca_duals),np.eye(mu_dim)) #70 x 70 vs 181*70 x 181*70
            self.S1 = sp.kron(sp.diags(ca_duals,format='csr'),sp.eye(mu_dim,format='csr')) #70 x 70 vs 181*70 x 181*70
            #TODO: if g1_inf norm prediction
            S2 = np.kron(np.diag(1-l1_duals),np.eye(2)) #2: disturbance feedback gain w.r.t. each TV's position and velocity
            
            self.S2 = sp.csr_matrix(S2)
            # self.S2 = sp.diags(1-l1_duals,format='csr') #if g1_tilde=1, then g1=0 or g1=lmbd. elif g1_tilde=0, then 0 < g1 < lmbd
            self.rho1 = 0.1
            self.rho2 = 0.1
            #Recover feasible dual solutions
            st = time.time()
            mu, eta, g1 = self.solve_dual_approximation(self.Q, self.L, self.F, self.C, self.p, self.f, self.c, self.S1, self.S2, self.rho1, self.rho2)
            solve_time = time.time() - st
            print(f'Dual Approximation Time: {solve_time}')
            st = time.time()
            gap = self._compute_gap_radius(mu, eta, g1)
            solve_time = time.time() - st
            print(f'Gap Radius Solve Time: {solve_time}')
        st = time.time()
        vars_kept=0 
        constr_kept=0
        ca_constr_ind_counter = 0
        vars_seen=set()
        if l1_duals is not None:
            #unflatten duals
            n_modes = [1 for _ in range(self.N_TV)]
            mode_map = dict(enumerate(product(*[range(n_modes[k]) for k in range(self.N_TV)])))
            l1_dual_dim = [self.N-1, n_modes, self.N_TV]
            ca_dual_dim = [self.N-1, len(mode_map), self.N_TV]
            l1_duals_list, ca_duals_list = unflatten_duals(np.expand_dims(np.concatenate([l1_duals,ca_duals]), axis=0),l1_dual_dim=l1_dual_dim,ca_dual_dim=ca_dual_dim)
            # alternatively, we can use the l1_duals and ca_duals directly
        for k in range(self.N_TV):
            for m in range(len(self.mode_map)):
                j=self.mode_map[m][k]
                for t in range(self.N-1):
                    if l1_duals is not None:
                        try:
                            if not (k,j,t) in vars_seen:
                                #TODO: for multi-modal predictions, the indexing for g1 must be fixed
                                if l1_duals_list[k][j][t][0]:
                                    gain_keep = 1
                                else:
                                    gain_keep = self._safe_screen(g1[len(self.mode_map[m][:k])*2*(self.N-1)+j*2*(self.N-1)+2*t:len(self.mode_map[m][:k])*2*(self.N-1)+j*2*(self.N-1)+2*(t+1)], gap_radius= gap ,dual_type='l1_dual') 

                                # gain_keep = 1- int(np.all(ca.vec(l1_duals[k][j][t][0])==np.zeros(2))) #tertiary simple gain_keep rule
                                # gain_keep = (1-int(np.all(ca.vec(l1_duals[k][j][t][0])<(self.l1_lmbd-1e-8)*np.ones(2)))) or (1-np.all(ca.vec(l1_duals[k][j][t][0])>(1e-8)*np.ones(2)))
                                # gain_keep = 1 #Only do constraint screening
                                vars_kept+=gain_keep*2
                                vars_seen.add((k,j,t))
                            # print(f'k,m,t: {k,m,t} \n  ca_duals dim: {len(ca_duals),len(ca_duals[k]),len(ca_duals[k][m])}')
                            # constr_keep =int(np.linalg.norm(ca_duals[k][m][t][0])>1e-3)
                            #ca dual should be a vector of size mu_dim
                            # if t ==0:
                            #     pdb.set_trace()
                            if ca_duals_list[k][j][t][0]:
                                constr_keep = 1
                            else:
                                constr_keep = self._safe_screen(mu[mu_dim *ca_constr_ind_counter:mu_dim *(ca_constr_ind_counter+1)], gap_radius=gap ,dual_type='ca_dual')
                            ca_constr_ind_counter += 1
                            constr_kept+=constr_keep
                        except:
                            pdb.set_trace()
                    else:
                        gain_keep = 1
                        constr_keep = 1
                        vars_kept+=gain_keep*2
                        constr_kept+=constr_keep
                    self.opti.set_value(self.gain_keep[k][j][t],gain_keep)
                    self.opti.set_value(self.constr_keep[k][m][t],constr_keep)
        if l1_duals is not None:
            self.vars_kept = vars_kept
            self.constr_kept = constr_kept
            print(f"vars:  {vars_kept} out of {g1.shape[0]}, constr: {constr_kept} out of {ca_duals.shape[0]}")     
        solve_time = time.time() - st
        print('Opti Set Value Time: ', solve_time, ' s')
        return mu, eta, g1, gap

    def clarkson_woodruff_transform(self,A, s):
        """
        Applies a Clarkson-Woodruff transform (CountSketch) to matrix A.
        
        Parameters:
        A (ndarray or sparse matrix): An m x n matrix.
        s (int): Target number of rows (s << m).
        
        Returns:
        A_sketch: The sketched matrix of size s x n.
        """
        m, n = A.shape
        # Define a hash function mapping each row of A to one of s rows.
        # Here we simply assign a random index in 0...s-1 for each of m rows.
        h = np.random.randint(low=0, high=s, size=m)
        # Random sign flips: +1 or -1 for each row.
        xi = np.random.choice([-1, 1], size=m)
        
        # Build sparse sketching matrix S (of shape s x m) in CSR format.
        rows = h              # row indices in S where each original row i goes to S[h[i], i]
        cols = np.arange(m)   # original row indices become column indices in S.
        data = xi.astype(float)
        S = sp.csr_matrix((data, (rows, cols)), shape=(s, m))
        
        # Multiply S * A to get the sketched matrix.
        A_sketch = S.dot(A)
        return A_sketch

    def solve_dual_approximation(self,Q, L, F, C, p, f, c, S1, S2, rho1, rho2):
        """
        Solves regularized dual problem via least squares with constraint projection
        Returns feasible (mu, eta, g1) approximation
        """
        # Precompute dimensions
        st = time.time()
        n_mu = C.shape[0]
        n_eta = F.shape[0]
        n_g1 = L.shape[0]
        # Construct least squares problem 
        self.c = sp.csr_matrix(c) 
        self.p = sp.csr_matrix(p)
        self.L = sp.csr_matrix(L)
        self.F = sp.csr_matrix(F)
        self.Q_inv = sp.csr_matrix(np.linalg.inv(Q))
        self.S1_sq = S1.T @ S1
        self.S2_sq = S2.T @ S2

        solve_time = time.time() - st
        print(f'[Dual Approximation] Precompute Dimensions Time: {solve_time} s')
        # A_grad_mu = sp.hstack([self.C @ self.Q_inv @ self.C.T + 2*rho1 * S1.T @ S1, -self.C @ self.Q_inv @ self.F.T, -2 * self.C @ self.Q_inv @ self.L.T])
        # A_grad_eta = sp.hstack([-self.F @ self.Q_inv @ self.C.T, self.F @ self.Q_inv @ self.F.T, 2 * self.F @ self.Q_inv @ self.L.T])
        # A_grad_g1 = sp.hstack([-2 * self.L @ self.Q_inv @ self.C.T, 2 * self.L @ self.Q_inv @ self.F.T, 4 * self.L @ self.Q_inv @ self.L.T + 2*rho2 * S2.T @ S2])
        # A = sp.vstack([A_grad_mu, A_grad_eta, A_grad_g1])

        # b_grad_mu = self.c + self.C @ self.Q_inv @(self.p - self.l1_lmbd*self.L.T @np.ones((n_g1,1)))
        # b_grad_eta = -f.reshape(-1,1) - self.F @ self.Q_inv @(self.p- self.l1_lmbd*self.L.T @np.ones((n_g1,1)))
        # b_grad_g1 = -2*self.L @ self.Q_inv @(self.p- self.l1_lmbd*self.L.T @np.ones((n_g1,1))) + self.l1_lmbd*rho2*S2.T @ np.ones((n_g1,1))
        # b = sp.vstack([b_grad_mu, b_grad_eta, b_grad_g1]).toarray().ravel()
        
        # Precompute common products:
        self.C_Qinv = self.C @ self.Q_inv
        self.F_Qinv = self.F @ self.Q_inv
        self.L_Qinv = self.L @ self.Q_inv

        # Precompute block products:
        A11 = self.C_Qinv @ self.C.T + 2 * rho1 * self.S1_sq
        A12 = -self.C_Qinv @ self.F.T
        A13 = -2 * self.C_Qinv @ self.L.T

        A21 = -self.F_Qinv @ self.C.T
        A22 = self.F_Qinv @ self.F.T
        A23 = 2 * self.F_Qinv @ self.L.T

        A31 = -2 * self.L_Qinv @ self.C.T
        A32 = 2 * self.L_Qinv @ self.F.T
        A33 = 4 * self.L_Qinv @ self.L.T + 2 * rho2 * self.S2_sq
        # Assemble the full A matrix using sp.bmat:
        row1 = sp.hstack([A11, A12, A13], format='csr')
        row2 = sp.hstack([A21, A22, A23], format='csr')
        row3 = sp.hstack([A31, A32, A33], format='csr')
        A = sp.vstack([row1, row2, row3], format='csr')
        # For b, cache the constant ones vector:
        ones_ng1 = np.ones((n_g1, 1))
        self.pmvec = (self.p - self.l1_lmbd * self.L.T @ ones_ng1)
        # self.pmvec = (self.p - 1 * self.L.T @ ones_ng1)
        b1 = self.c + self.C_Qinv @ self.pmvec
        b2 = -f.reshape((-1, 1)) - self.F_Qinv @ self.pmvec
        b3 = -2 * self.L_Qinv @ self.pmvec + self.l1_lmbd * rho2 * S2.T @ ones_ng1
        # b3 = -2 * self.L_Qinv @ self.pmvec + 1 * rho2 * S2.T @ ones_ng1
        # Assemble b in one call:
        b = sp.vstack([b1, b2, b3]).toarray()

        solve_time = time.time() - st
        print(f'[Dual Approximation] Least Squares Formulation Time: {solve_time} s')
        # Solve least squares with iterative method
        reduce_w_random_rows=False
        if reduce_w_random_rows:
            n_rows = 8000
            A = self.clarkson_woodruff_transform(A,s=n_rows)
            b = self.clarkson_woodruff_transform(b,s=n_rows)
        st = time.time()
        pdb.set_trace()
        x = lsqr(A, b, atol=1e-6, btol=1e-6,iter_lim=1000)[0]
        solve_time = time.time() - st
        print(f'[Dual Approximation] Least Squares Solve Time: {solve_time} s')

        # Split variables and project to constraints
        mu = np.maximum(x[:n_mu], 0)
        eta = np.maximum(x[n_mu:n_mu+n_eta], 0)
        g1 = np.clip(x[n_mu+n_eta:], 0, self.l1_lmbd)
        # g1 = np.clip(x[n_mu+n_eta:], -1, 1)  # Ensure ||g||_inf <= 1

        # # #Constrained least squares
        # ub= np.concatenate([np.ones((n_mu+n_eta))*np.inf , self.l1_lmbd*np.ones(n_g1)])
        # st = time.time()
        # x_star = lsq_linear(A, b.ravel(), tol=1e-6,bounds=(np.zeros(n_mu+n_eta+n_g1),ub), method='trf')
        # x = x_star.x
        # solve_time = time.time() - st
        # print(f'[Dual Approximation] Constrained Least Squares Solve Time: {solve_time} s')

        # mu = x[:n_mu] # mu corresponds to the first n_mu entries in x
        # eta = x[n_mu:n_mu+n_eta] # eta corresponds to the next n_eta entries in x
        # g1 = x[n_mu+n_eta:] # g1 corresponds to the last n_g1 entries in x
        return mu, eta, g1

    def _compute_gap_radius(self, f_mu, f_nu, f_g):
        """
        gap_radius = ||(dual - Proj(dual-grad_dual))|| * (1+sigma)/eta where
        eta = 1/largest_eigval(Q), sigma = 1/smallest_eigval(Q),
        grad_d_obj = [C; -F; -2L] * inv(Q) * (C' f_mu - F' f_nu - 2L' f_g_adjusted) + [c; f; 0],
        with f_g_adjusted = 2*f_g - l1_lmbd*ones.
        """
        # Compute eigenvalues of Q once (if Q does not change frequently you could precompute these)

        st = time.time()
        Q = sp.csr_matrix(self.Q)
        self.largest_eig = spla.eigsh(Q, k=1, which='LM', v0=self.largest_eig*np.ones(Q.shape[0]) if hasattr(self,'largest_eig') else None, return_eigenvectors=False)[0]
        self.smallest_eig = spla.eigsh(Q, k=1, v0=self.smallest_eig*np.ones(Q.shape[0])  if hasattr(self,'smallest_eig') else None, sigma=0, which='LM', return_eigenvectors=False)[0] #Solve for smallest eigenvalue using shift-invert  [https://docs.scipy.org/doc/scipy/tutorial/arpack.html]
        solve_time = time.time() - st

        print(f'Eigenvalues: {self.largest_eig, self.smallest_eig}')    
        print(f'Eigenvalue Computation Time: {solve_time} s')
        eta, sigma = self.largest_eig**(-1), self.smallest_eig**(-1)
        
        # Precompute the right-hand side product. Note: reshape inputs as column vectors.
        st = time.time()
        ones_fg = np.ones((f_g.shape[0], 1))
        temp = (-self.C.T @ f_mu.reshape((-1, 1))
                + self.p
                + self.F.T @ f_nu.reshape((-1, 1))
                + self.L.T @ (2 * f_g.reshape((-1, 1)) - self.l1_lmbd * ones_fg))
        
        # Use your precomputed blocks to form the stacked multiplication:
        stacked_prod = sp.vstack([-self.C_Qinv, self.F_Qinv, 2 * self.L_Qinv])
        grad_d = stacked_prod @ temp
        
        # Add the constant offset
        grad_d += sp.vstack([-self.c, self.f.reshape((-1, 1)), np.zeros(f_g.shape).reshape((-1, 1))])

        # Form the full dual vector and compute the projected version:
        duals = np.concatenate((f_mu, f_nu, f_g)).reshape((-1, 1))
        proj_dual = duals - grad_d
        start_idx = f_mu.shape[0] + f_nu.shape[0]
        proj_dual[:start_idx] = np.maximum(proj_dual[:start_idx], 0)
        solve_time = time.time() - st
        print(f'Gradient Computation Time: {solve_time} s')
        # For the last block (corresponding to f_g), clip between 0 and l1_lmbd.
        proj_dual[start_idx:] = np.clip(proj_dual[start_idx:], 0, self.l1_lmbd)
        # proj_dual[start_idx:] = np.clip(proj_dual[start_idx:], -1, 1)  # Ensure ||g||_inf <= 1
        # Compute gap
        st = time.time()
        gap = np.linalg.norm(duals - proj_dual) * (1 + sigma) / eta
        # gap = np.linalg.norm(grad_d) * self.largest_eig
        solve_time = time.time() - st
        print(f'Norm time: {solve_time} s')
        print(gap)
        return gap
    
    def _safe_screen(self,dual, gap_radius, dual_type = "ca_dual"):
            keep = 1
            TOL = 1e-2
            if dual_type == "ca_dual":
                # Do ca_dual sensitivity-based screen (l2 norm of mu)
                # if np.linalg.norm(dual,ord=2) <= min(TOL, gap_radius):
                #     keep = 0
                if np.linalg.norm(dual, ord=2) + gap_radius < TOL:
                    keep = 0
            else:
                # Do l1_dual strong duality screen (infinity norm of g1)
                if np.linalg.norm(dual,ord=np.inf) + gap_radius < 1 and np.linalg.norm(dual,ord=np.inf) - gap_radius > 0:
                    keep = 0
            return keep #output 1 or 0

    def plot_obca_obstacles_and_ego_path(self, pos_tvs, psi_tvs, tv_params, buffer_radius=1.5, t=0):
        """
        Plots OBCA rectangular obstacle polytopes and ego path with buffer.

        :param pos_tvs: list of [2, N+1] TV position arrays (one per vehicle)
        :param psi_tvs: list of [1, N+1] TV headings (radians)
        :param tv_params: list of (length, width) tuples for TVs
        :param buffer_radius: radius of ego buffer (approximate safety zone)
        :param t: timestep to visualize
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot TV rectangles
        for k in range(len(pos_tvs)):
            for j in range(len(self.mode_map)):
                m=self.mode_map[j][k]
                center = self.opti.value(pos_tvs[k][m][:,t])
                psi = self.opti.value(psi_tvs[k][m][0, t])
                length, width = self.opti.value(tv_params[k])

                # Compute rectangle corners
                R = np.array([[np.cos(psi), -np.sin(psi)],
                            [np.sin(psi),  np.cos(psi)]])
                rect_pts = np.array([
                    [ length/2,  width/2],
                    [-length/2,  width/2],
                    [-length/2, -width/2],
                    [ length/2, -width/2],
                ]).T
                corners = R @ rect_pts + center.reshape(2, 1)

                polygon = plt.Polygon(corners.T, closed=True, edgecolor='black', facecolor='red', alpha=0.5, label='TV' if k == 0 else "")
                ax.add_patch(polygon)
                ax.plot(center[0], center[1], 'ko')  # TV center

        # # Plot ego path
        # ax.plot(ego_path[0, :], ego_path[1, :], 'g.-', label='Ego Path')
        # ax.plot(ego_path[0, t], ego_path[1, t], 'go', markersize=10, label='Ego @ t')

        # Add ego buffer at each step
        # for tt in range(ego_path.shape[1]):
        #     circ = patches.Circle((ego_path[0, tt], ego_path[1, tt]), buffer_radius,
        #                         edgecolor='green', facecolor='none', linestyle='--', alpha=0.4)
        #     ax.add_patch(circ)

        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title(f"OBCA Constraints Visualization at t={t}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

def plot_collision_linearization(Q, c, x):
    # Compute linearization point on the ellipse boundary
    diff = x - c
    mahalanobis_norm = np.sqrt(diff.T @ Q @ diff)
    oa_ref = c + diff / mahalanobis_norm

    # Generate unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack((np.cos(theta), np.sin(theta)))

    # Ellipse: (x - c)^T Q (x - c) = 1 ⇒ shape is inv(Q)
    Qinv = np.linalg.inv(Q)
    eigvals, eigvecs = np.linalg.eigh(Qinv)
    axes = np.sqrt(eigvals)  # semi-axes lengths
    ellipse = eigvecs @ np.diag(axes) @ circle + c[:, None]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ellipse[0], ellipse[1], 'b-', label='Ellipse')
    plt.plot(c[0], c[1], 'bo', label='TV Center (c)')
    plt.plot(oa_ref[0], oa_ref[1], 'go', label='oa_ref (linearization pt)')
    plt.plot(x[0], x[1], 'ro', label='Ego Position')

    # Draw normal vector
    plt.quiver(c[0], c[1], oa_ref[0] - c[0], oa_ref[1] - c[1],
               angles='xy', scale_units='xy', scale=1, color='g', width=0.005, label='Normal (oa_ref - c)')

    # Draw deviation vector
    plt.quiver(oa_ref[0], oa_ref[1], x[0] - oa_ref[0], x[1] - oa_ref[1],
               angles='xy', scale_units='xy', scale=1, color='r', width=0.005, label='Deviation (x - oa_ref)')

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Collision Avoidance Linearization Visualization')
    plt.legend()
    plt.show()

    # Print debug info
    print("Check (oa_ref - c)^T Q (oa_ref - c) =", (oa_ref - c).T @ Q @ (oa_ref - c))

