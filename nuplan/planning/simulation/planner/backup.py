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
import scipy.sparse.linalg as spla from scipy.linalg 
import clarkson_woodruff_transform as sketch 
from scipy.sparse import block_diag, vstack 
from scipy.sparse.linalg import lsmr, svds, lsqr
import scipy.sparse as sp 
import torch from nuplan.planning.simulation.planner.dualGD 
import DualApproxGD import matplotlib.pyplot as plt 
import matplotlib.patches as patches 

logger = logging.getLogger(__name__) 
class SMPC(): 
    def __init__(self, 
                ev, 
                N = 15, 
                V_MIN = -1., #Speed, acceleration constraints 
                V_MAX = 10.0, 
                A_MIN = -5.0, 
                A_MAX = 2.0, 
                TIGHTENING = 2.5, # of std that you want to be robust w.r.t. the TV uncertainty
                EV_NOISE_STD = [0.001, 0.001], 
                TV_NOISE_STD =[[0.01, 0.02]]*5, 
                Q = [1.,1.], # cost for measuring progress: -Q*s_{t+1}. #was 1. 
                R = 1., # cost for penalizing large input rate: (u_{t+1}-u_t).T@R@(u_{t+1}-u_t) #was 1.5 
                ev_length = 4.47, 
                offline_mode=True, 
                solver="ipopt",
                open_loop = False, 
                eval_mode = False, 
                is_mm_preds: bool = False, 
                route = None, 
                preds = List, 
                canon_prob_fn=None, 
                config=None, ): 
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
        print('[smpc.py]: Collision Avoidance Method is',self.config['collision_avoidance_method']) 
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
            self.tight=2.1
        elif self.config['collision_avoidance_method'] == 'affine': 
            self.tight=2.3 
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
        if not self.offline: 
            self.time_safety_screening = np.nan 
            self.time_set_canon_form_mats = np.nan 
            self.time_least_squares_formulation = np.nan 
            self.time_least_squares_solve = np.nan 
            self.p_opts = {'expand': False, 'print_time':0, 'verbose' :False, 'error_on_fail':0} 
            self.s_opts = {'print_level': 0,'tol':1e-4,'max_wall_time': 120.,'constr_viol_tol':1e-4}
            # if eval_mode: # s_opts.update({'max_wall_time': 15.,'constr_viol_tol':1e-4}) 
            s_opts_grb = {'OutputFlag': 0, 'PSDTol' : 1e-3, 'FeasibilityTol' : 1e-3, 'BarConvTol':1e-3, 'BarQCPConvTol':1e-3, 'LogToConsole': 0}
            p_opts_grb = {'expand': False,'error_on_fail':0, 'verbose':False, 'ad_weight':0} 
            self.solver=solver 
            if self.solver=="ipopt": 
                self.opti=ca.Opti() 
                self.opti.solver("ipopt", self.p_opts, self.s_opts)
            elif self.solver=="gurobi": 
                self.opti=ca.Opti("conic") 
                self.opti.solver("gurobi", p_opts_grb, s_opts_grb) 
            elif self.solver=="mosek": 
                self.opti=ca.Opti("conic") 
                self.opti.solver("mosek", self.p_opts, self.s_opts) 
            elif self.solver=="scs": 
                self.opti=ca.Opti("conic") 
                self.opti.solver("scs", self.p_opts, self.s_opts) 
            else: 
                raise ValueError(f"Unknown solver: {self.solver}") 
            def _flatten2ca(xs): 
                if type(xs) == type([]): 
                    for x in xs: yield from _flatten2ca(x) 
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
            self.dpos_tvs=[[[self.opti.parameter(2,1) for _ in range(self.N)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)] self.Qs=[[[self.opti.parameter(2,2) for _ in range(self.N)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)] 
            self.psi_tvs = [[self.opti.parameter(1,self.N+1) for _ in range(self.N_modes[k])] for k in range(self.N_TV)] # TV heading 
            self.tv_params = [self.opti.parameter(2) for _ in range(self.N_TV)] # TV length and width 
            self.params+=[self.z_tv_curr, self.u_tvs, self.pos_tvs, self.dpos_tvs, self.Qs, self.psi_tvs, self.tv_params,self.s0] 
            
            if not self.offline: 
                #Parameters for constraint and gain screening 
                self.gain_keep=[[self.opti.parameter(self.N-1,1) for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                self.constr_keep=[[self.opti.parameter(self.N-1,1) for m in range(len(self.mode_map))] for k in range(self.N_TV)] 
                # self.constr_keep=[[np.zeros((self.N-1)) for m in range(len(self.mode_map))] for k in range(self.N_TV)] 
            self.params = ca.vertcat(*_flatten2ca(self.params)) 
            self.policy=self._return_policy_class() 
            self._add_constraints_and_cost() 
            self._update_ev_initial_condition(np.array([0., 2.]), 0.) 
            self._update_ev_preds(np.ones((2,self.N+1)), 50*np.ones((2,self.N+1)), [np.ones((2,1))]*self.N) 
            self._update_tv_initial_condition([np.array([0., 0.])]*self.N_TV) 
            self._update_tv_preds([[np.zeros((self.N,1))]*self.N_modes[k] for k in range(self.N_TV)], [[np.zeros((2,self.N+1))]*self.N_modes[k] for k in range(self.N_TV)], 
                                  [[[np.ones((2,1))]*self.N]*self.N_modes[k] for k in range(self.N_TV)], [[[np.eye(2)]*self.N]*self.N_modes[k] for k in range(self.N_TV)]) 
            self._update_tv_psi([[np.zeros((1,self.N+1))]*self.N_modes[k] for k in range(self.N_TV)]) 
            self._update_tv_params([[4.47, 2]]*self.N_TV) self._update_red_light(None,None) 
            self._update_speed_limit(self.config['v_max']) 
            self._update_leading_vehicle_params(None) 
            if not self.offline: 
                _,_,_,_ = self.update_gain_and_constr_keeps() 
            self.solve(first_solve=True) 
            # if not self.offline: 
            # # self.opti_copy = self.opti.copy() 
            # #Used in online mode for faster replanning 
            # # self.opti_copy.solver("ipopt", self.p_opts, self.s_opts) 
            # 
    def _return_policy_class(self): 
        """ 
        EV Affine disturbance feedback + TV state feedback policies from https://arxiv.org/abs/2109.09792 
        """
        if self.config['collision_avoidance_method'] == 'obca': 
            self.obca_lmbd = [[self.opti.variable(4, self.N-1) for _ in range(self.N_modes[k])] for k in range(self.N_TV)] #Assuming rectangular obstacles 
            self.obca_lmbd_redlight = self.opti.variable(4, self.N-1) 
            self.slack = [[[self.opti.variable(1) for _ in range(self.N-1)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)] 
            self.slack_vec = ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[self.slack[k][j][t] for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]) 
            
            #Parameters for red light and lead vehicle collision avoidance 
            self.redlight = self.opti.parameter(2,1) 
            self.lead_vehicle_s = self.opti.parameter(1) 
            
            M=[[ca.DM(1, 2) for n in range(t)] for t in range(self.N)] #Not Used 
            h0=self.opti.variable(1) #Initial nominal input 
            h=[self.opti.variable(1) for t in range(self.N-1)] #nominal input sequence 
            
            if self.open_loop: 
                K=[[[ ca.DM(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            else: 
                if not self.offline: #evaluation mode (online) 
                    K4screening=[[[self.opti.variable(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                    # K=[[[ca.if_else(self.gain_keep[k][j][t], K4screening[k][j][t], ca.MX.zeros(1, 2)) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                    K=[[[self.gain_keep[k][j][t]*K4screening[k][j][t] for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                    # K = K4screening 
                    self.gain_l1=[[[ca.if_else(self.gain_keep[k][j][t], self.opti.variable(1,2), ca.MX.zeros(1, 2)) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                else: 
                    K=[[[self.opti.variable(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                    self.gain_l1=[[[self.opti.variable(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            h_stack=ca.vertcat(h0,*[h[t] for t in range(self.N-1)]) 
            M_stack=ca.vertcat(*[ca.horzcat(*[M[t][n] for n in range(t)], ca.DM(1,2*(self.N-t))) for t in range(self.N)])
            K_stack=[[ca.diagcat(ca.DM(1,2),*[K[k][j][t] for t in range(self.N-1)]) for j in range(self.N_modes[k])] for k in range(self.N_TV)] #Gains for the first time step is set to zero 
            
            if self.config['collision_avoidance_method'] == 'obca': 
                self.vars_pol = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K[k][j][t]) for t in range(self.N-1)], ca.vec(self.obca_lmbd[k][j])) for j in range(self.N_modes[k])]) for k in range(self.N_TV)])) 
            else:
                #Affine 
                self.vars_pol = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)])) 
                
            if self.offline: 
                self.vars_epi = ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(self.gain_l1[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]) 
            else: 
                if self.config['collision_avoidance_method'] == 'obca':
                    self.vars_pol4screening = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K4screening[k][j][t]) for t in range(self.N-1)], ca.vec(self.obca_lmbd[k][j])) for j in range(self.N_modes[k])]) for k in range(self.N_TV)])) 
                else: 
                    self.vars_pol4screening = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K4screening[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)])) 
               
            #Variables for warmstart 
            self.vars_ws, self.vars_epi_ws = None, None 
        return h_stack,M_stack,K_stack
    
    def _get_ATV_TV_dynamics(self): 
        """ 
        Constructs system matrices such that for mode j and for TV k, 
        O_t=T_tv@o_{t|t}+c_tv+E_tv@N_t 
        where
        O_t=[o_{t|t}, o_{t+1|t},...,o_{t+N|t}].T, (TV state predictions) 
        N_t=[n_{t|t}, n_{t+1|t},...,n_{t+N-1|t}].T, (TV process noise sequence) 
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
                        # E_tv[k][j][t*2:(t+1)*2,:] = self.Atv@E_tv[k][j][(t-1)*2:t*2,:] 
                        E_tv[k][j][t*2:(t+1)*2,:] = ca.repmat(E, 1, self.N) #No dynamics propagation for the noise in WayFormer 
                        E_tv[k][j][t*2:(t+1)*2,(t-1)*2:t*2]=E 
                        
                c_tv[k][j]=TB_tv[k][j]@u_tvs[k][j]
                
        return T_tv, c_tv, E_tv 
    
    
    def _get_LTV_EV_dynamics(self): 
        """ 
        Constructs system matrices such for EV, 
        X_t=A_pred@x_{t|t}+B_pred@U_t+E_pred@W_t 
        where 
        X_t=[x_{t|t}, x_{t+1|t},...,x_{t+N|t}].T, (EV state predictions) 
        U_t=[u_{t|t}, u_{t+1|t},...,u_{t+N-1|t}].T, (EV control sequence) 
        W_t=[w_{t|t}, w_{t+1|t},...,w_{t+N-1|t}].T, (EV process noise sequence) 
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
        self.nom_z_tv=[[T_tv[k][j]@self.z_tv_curr[k]+c_tv[k][j] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
        
        #Cost initialization 
        cost = 0 
        # #State and input constraints 
        # self.opti.subject_to(self.opti.bounded(self.V_MIN, A[[t*2+1 for t in range(1,self.N+1)],:]@self.z_curr+B[[t*2+1 for t in range(1,self.N+1)],:]@h, self.V_MAX)) 
        self.opti.subject_to(self.V_MIN<=A[[t*2+1 for t in range(1,self.N+1)],:]@self.z_curr+B[[t*2+1 for t in range(1,self.N+1)],:]@h) 
        self.opti.subject_to(A[[t*2+1 for t in range(1,self.N+1)],:]@self.z_curr+B[[t*2+1 for t in range(1,self.N+1)],:]@h <= self.V_MAX + self.slack[0][0][0]) 
        self.opti.subject_to(self.opti.bounded(self.A_MIN, h, self.A_MAX))
        
        #Propagate nominal dynamics 
        nom_z=A@self.z_curr+B@h 
        self.nom_z = nom_z 
        nom_s=ca.vec(nom_z.reshape((2,-1))[0,:]) 
        nom_z_diff=ca.vec(ca.diff(nom_z.reshape((2,-1)),1,1)) 
        
        #collision avoidance with the lead vehicle (added because wayformer doesn't detect objects that doesn't move by some threshold) 
        self.lead_vehicle_constr = nom_s[-1]<= self.lead_vehicle_s-self.s0 - self.ev_length*2 - 1 +self.slack[0][0][0] #s_{N|t} <= s_{lead|t} - 2*ev_length - 1 + slack 
        self.opti.subject_to(self.lead_vehicle_constr) #s_{N|t} <= s_{lead|t} 
        # cost+=-2.7*self.Q_cost*ca.sum1(nom_s) +2.*self.Q_cost*nom_z_diff.T@nom_z_diff# penalizes slow progress (was -2.5, 2) 
        self.multiplier = 0.1 
        cost += -0.1*self.multiplier*10*self.Q_cost[0,0]*ca.sum1(nom_s) + self.multiplier*5.56*nom_z_diff.T@ca.kron(ca.DM.eye(self.N),self.Q_cost)@nom_z_diff# penalizes slow progress (was -4, 3.5) 
        cost += self.multiplier*0.01*self.R_cost*ca.diff(ca.vertcat(self.u_prev,h),1,0).T@ca.diff(ca.vertcat(self.u_prev,h),1,0) # penalizes large input rates  

        if self.offline: 
            self.lin_ineq_l1 = [] 
            self.ca_ineq = [] 
            self.l1_constr = [[[ [] for _ in range(self.N-1)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)] 
            self.ca_constr = [[[ [] for _ in range(self.N-1)] for _ in range(len(self.mode_map))] for _ in range(self.N_TV)] 
        else: 
            self.soc_constr_online = [[[ [] for _ in range(self.N-1)] for _ in range(len(self.mode_map))] for _ in range(self.N_TV)] 
            self.l1_constr = [[[ [] for _ in range(self.N-1)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)] 
            
            '''
            Red light related constraints 
            ''' 
            if self.config['collision_avoidance_method'] == 'obca': 
                # OBCA for redlight 
                redlight_obca_lmbd = self.obca_lmbd_redlight
                d_min_red = 0 
                obca_lmbd = self.obca_lmbd 
                d_min = 0.8*(self.ev_length/2) 
                # d_min = 0.8 
                for t in range(1,self.N): 
                    ego_psi = self.route(nom_s[t] + self.s0)[2] 
                    Rev = ca.vertcat( 
                                     ca.horzcat(ca.cos(ego_psi), -ca.sin(ego_psi)),
                                     ca.horzcat(ca.sin(ego_psi), ca.cos(ego_psi)) )
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
                    self.opti.subject_to(self.opti.bounded(0,nom_s[t],self.redlight[0]-self.s0)) 
                else: 
                    raise ValueError(f"Unknown collision avoidance method: {self.config['collision_avoidance_method']}") 
                
            ''' 
            Collision avoidance constraints 
            ''' 
            for k in range(self.N_TV): 
                for j in range(len(self.mode_map)): 
                    m = self.mode_map[j][k] 
                    
                    # (unchanged) gain regularizer 
                    cost += self.multiplier * 0.019 * 1.5 * ca.trace( K[k][m] @ E_tv[k][m][:2*self.N, :] @ E_tv[k][m][:2*self.N, :].T @ K[k][m].T )
                    
                    # --------- AFFINE collision-avoidance: batch all t constraints ---------- 
                    if self.config['collision_avoidance_method'] == 'affine': 
                        soc_rows = [] # holds [ y+slack - z_norm ; y+slack ] for all t 
                        soc_rows_online = [] # masked rows for online screening (same length) 
                        soc_rows_gurobi = [] # masked rows for gurobi (same length) 
                        soc_rows_gurobi_online = [] # masked rows for gurobi online (same length) 
                        # Also batch L1 constraints: stack K entries and gains once 
                        if len(self.l1_constr[k][m][0]) == 0: 
                            # # Build stacked vectors: shape (2*(N-1), 1) 
                            K_stack_vec = ca.vertcat(*[ 
                                                       ca.vec(K[k][m][t, 2*t:2*(t+1)].T) # (2x1) 
                                                       for t in range(1, self.N) 
                                                       ]) 
                            gain_l1_stack = ca.vertcat(*[ 
                                                         ca.vec(self.gain_l1[k][m][t-1].T) # (2x1)
                                                         for t in range(1, self.N) 
                                                         ]) 
                            # # Single pair of inequalities for all t 
                            if self.solver == 'gurobi' or not self.offline: #evaluation only. Don't use for data collection
                                self.opti.subject_to(K_stack_vec - gain_l1_stack <= 0) 
                                self.opti.subject_to(-K_stack_vec - gain_l1_stack <= 0) 
                                self.l1_constr[k][m][0]+=[1] 
                                
                            # Single cost term for all t 
                            cost += self.l1_lmbd * ca.sum1(gain_l1_stack) 
                            
                        # Loop t only builds expressions; we impose them once after the loop 
                        for t in range(1, self.N): 
                            # EV–TV geometry 
                            diff = self.x_pos[:, t] - self.pos_tvs[k][m][:, t] 
                            mahalanobis_norm = ca.sqrt(diff.T @ self.Qs[k][m][t-1] @ diff) 
                            oa_ref = self.pos_tvs[k][m][:, t] + diff / (mahalanobis_norm + 1e-12) 
                            
                            # z (random part) and y (deterministic part) 
                            z = ( (oa_ref - self.pos_tvs[k][m][:, t]).T @ self.Qs[k][m][t-1] @ ca.horzcat( self.dpos[t-1] @ (B[2*t, :] @ M + E[2*t, :]), *[ self.dpos[t-1] @ B[2*t, :] @ K[l][self.mode_map[j][l]] @ E_tv[l][self.mode_map[j][l]][:2*self.N, :] - (int(l == k)) * self.dpos_tvs[k][m][t-1] @ E_tv[k][m][2*t, :] for l in range(self.N_TV) ] ) ).T 
                            z_norm = self.tight * ca.sqrt(ca.sumsqr(z) + 1e-10) 
                            
                            y = ( (oa_ref - self.pos_tvs[k][m][:, t]).T @ self.Qs[k][m][t-1] @ (self.x_pos[:, t] - oa_ref + self.dpos[t-1] * (A[2*t, :] @ self.z_curr + B[2*t, :] @ h - (self.z_lin[0, t] - self.s0))) ) 
                            if self.offline: 
                                self.ca_ineq.append(ca.vertcat(z,y)) 
                                self.lin_ineq_l1+=[K[k][m][t,2*t:2*(t+1)]-self.gain_l1[k][m][t-1]] #only the first constraint: g1 
                                self.ca_constr[k][j][t-1]+=[z_norm<=y+self.slack[k][m][t-1],0<=y+self.slack[k][m][t-1]] 
                                if self.solver == 'ipopt': 
                                    self.opti.subject_to(self.ca_constr[k][j][t-1][0]) 
                                    self.opti.subject_to(self.ca_constr[k][j][t-1][1]) 
                                    if len(self.l1_constr[k][m][t-1])==0: 
                                        self.l1_constr[k][m][t-1]+=[K[k][m][t,2*t:2*(t+1)]<=self.gain_l1[k][m][t-1], -self.gain_l1[k][m][t-1]<=K[k][m][t,2*t:2*(t+1)]] 
                                        self.opti.subject_to(self.l1_constr[k][m][t-1][0]) 
                                        self.opti.subject_to(self.l1_constr[k][m][t-1][1]) 
                            # Collect the two scalar rows for this t 
                            soc_rows.append(y + self.slack[k][m][t-1] - z_norm) 
                            soc_rows.append(y + self.slack[k][m][t-1]) 
                            soc_rows_gurobi.append(ca.soc(z,y+self.slack[k][m][t-1])) 
                            
                            if not self.offline: 
                                # Masked version (avoid if_else): keep*row + (1-keep)*1 
                                soc_rows_online.append(self.constr_keep[k][j][t-1]*(y + self.slack[k][m][t-1] - z_norm) + (1-self.constr_keep[k][j][t-1])*1e1) 
                                soc_rows_online.append(self.constr_keep[k][j][t-1]*(y + self.slack[k][m][t-1]) + (1-self.constr_keep[k][j][t-1])*1e1) 
                                soc_rows_gurobi_online.append(ca.soc(self.constr_keep[k][j][t-1]*z, (1-self.constr_keep[k][j][t-1])*1e1 + self.constr_keep[k][j][t-1]*(y+self.slack[k][m][t-1]))) 
                                if self.solver == 'gurobi': 
                                    self.opti.subject_to(soc_rows_gurobi_online[-1] > 0) 
                            else: 
                                # soc_rows_online.append((y - z_norm)* self.constr_keep[k][j][t-1] + self.slack[k][m][t-1])
                                # soc_rows_online.append(y*self.constr_keep[k][j][t-1] + self.slack[k][m][t-1] ) 
                                # self.soc_constr_online[k][j][t-1]+=[y + self.slack[k][m][t-1] - z_norm, y + self.slack[k][m][t-1]] 
                                # if self.solver == 'gurobi': 
                                # self.opti.subject_to(soc_rows_gurobi[-1] > 0) 
                                
                        # Impose all CA constraints at once for ipopt 
                        if self.solver == "ipopt": 
                            if self.offline: 
                                pass # already imposed inside t-loop 
                            else: 
                                self.opti.subject_to(ca.vertcat(*soc_rows_online) >= 0)
                    else: 
                        NotImplementedError("Only 'affine' collision avoidance is implemented for now.")
            cost += 1e4*self.slack_vec.T@self.slack_vec 
            self.opti.minimize( cost ) 
            self.cost = cost 
            
            #Canonical Form Computations 
            if self.offline: 
                #g(theta) <= 0 
                self.lin_ineq_constr =[] 
                self.lin_ineq_constr+=[A[t*2+1,:]@self.z_curr+B[t*2+1,:]@h-self.V_MAX for t in range(1,self.N+1)] 
                self.lin_ineq_constr+=[-A[t*2+1,:]@self.z_curr-B[t*2+1,:]@h + self.V_MIN for t in range(1,self.N+1)] 
                self.lin_ineq_constr+=[h[t]-self.A_MAX for t in range(self.N)] 
                self.lin_ineq_constr+=[-h[t] + self.A_MIN for t in range(self.N)] 
                self.f_l_i_c = ca.Function("lin_ineq", [self.vars_pol,self.params, self.V_MAX], self.lin_ineq_constr) 
                
                # F\theta <= f 
                self.F, self.f = ca.jacobian(ca.vertcat(*self.f_l_i_c(self.vars_pol,self.params,self.V_MAX)),self.vars_pol), -ca.vertcat(*self.f_l_i_c(ca.DM.zeros(*self.vars_pol.shape),self.params, self.V_MAX)) 
                
                # L\theta <= psi 
                self.f_l_i_l1 =ca.Function("l1_ineq", [self.vars_pol,self.vars_epi],self.lin_ineq_l1)
                self.L = ca.jacobian(ca.vertcat(*self.f_l_i_l1(self.vars_pol,self.vars_epi)), self.vars_pol)
                
                self.f_ca_i = ca.Function("ca_ineq", [self.vars_pol, self.params], self.ca_ineq) 
                
                # C\theta + c \in K_1 x K_2 x ................. 
                self.C = (ca.jacobian(ca_constr, self.vars_pol) for ca_constr in self.f_ca_i(self.vars_pol, self.params)) 
                self.c = self.f_ca_i(ca.DM.zeros(*self.vars_pol.shape), self.params) 
                
                self.f_cost = ca.Function("cost", [self.vars_pol, self.vars_epi, self.params, self.slack_vec], [cost])
                
                #Here, ca.hessian outputs hessian, J_grad. 
                self.Q, self.p = ca.hessian(self.f_cost(self.vars_pol, self.vars_epi,self.params, ca.DM.zeros(*self.slack_vec.shape)), self.vars_pol) 
                self.f_hessian = ca.Function("hessian", [self.vars_pol, self.vars_epi, self.params, self.slack_vec], [self.Q, self.p]) 
                self.d = self.f_cost(ca.DM.zeros(*self.vars_pol.shape),ca.DM.zeros(*self.vars_epi.shape),self.params,ca.DM.zeros(*self.slack_vec.shape))
            else: 
                #Precompute functions for constraints and variable screening 
                vars_epi = ca.DM.zeros(2*(self.N-1)*self.N_modes[0]*self.N_TV,1) 
                self.F_fn = ca.Function('F_fn', [self.vars_pol4screening, self.params, self.V_MAX], [ca.jacobian(ca.vertcat(*self.canon_prob_fn['f_l_i_c'](self.vars_pol4screening,self.params,self.V_MAX)),self.vars_pol4screening)]) 
                self.L_fn = ca.Function('L_fn', [self.vars_pol4screening], [ca.jacobian(ca.vertcat(*self.canon_prob_fn['f_l_i_l1'](self.vars_pol4screening,vars_epi)), self.vars_pol4screening)]) 
                self.C_fn = ca.Function('C_fn',[self.params, self.slack_vec],[ca.substitute(ca.jacobian(ca.simplify(ca.vertcat(*self.canon_prob_fn['f_ca_i'](self.vars_pol4screening, self.params))), self.vars_pol4screening), self.vars_pol4screening, ca.DM.zeros(*self.vars_pol4screening.shape))]) 
                C = self.C_fn(np.ones(self.params.shape),ca.DM.zeros(*self.slack_vec.shape)) # Evaluate C_fn to get the shape and non-zero indices 
                self.C_shape = C.shape 
                self.nonzero_inds = np.nonzero(np.ravel(C,order='F'))[0] 
                # Get the row and column indices of the non-zero elements in the sparse matrix 
                self.row_indices, self.col_indices = np.unravel_index(self.nonzero_inds, self.C_shape,order='F') 
                nonzero_vec = ca.vec(ca.substitute(ca.jacobian(ca.simplify(ca.vertcat(*self.canon_prob_fn['f_ca_i'](self.vars_pol4screening, self.params))), self.vars_pol4screening), self.vars_pol4screening, ca.DM.zeros(*self.vars_pol4screening.shape)))[self.nonzero_inds] 
                self.C_fn_nonzero = ca.Function('C_fn_nonzero',[self.params, self.slack_vec],[nonzero_vec]) 
                self.c_fn = ca.Function('c_fn',[self.params],[ca.vertcat(*self.canon_prob_fn['f_ca_i'](ca.DM.zeros(*self.vars_pol4screening.shape), self.params))]) 
                Q, p = self.canon_prob_fn['f_hessian'](ca.DM.zeros(*self.vars_pol4screening.shape),vars_epi,self.params,ca.DM.zeros(*self.slack_vec.shape)) 
                Q_inv = ca.inv(Q) 
                L = self.l1_lmbd * self.L_fn(ca.DM.zeros(*self.vars_pol4screening.shape)) 
                F, f = self.F_fn(ca.DM.zeros(*self.vars_pol4screening.shape),self.params,self.V_MAX), -ca.vertcat(*self.canon_prob_fn['f_l_i_c'](ca.DM.zeros(*self.vars_pol4screening.shape),self.params,self.V_MAX)) 
                L_Q_inv = L @ Q_inv 
                F_Q_inv = F @ Q_inv # Constuct functions for the least squares problem 
                self.L_Q_inv_fn = ca.Function('L_Q_inv_fn', [self.params], [L_Q_inv]) 
                self.F_Q_inv_fn = ca.Function('F_Q_inv_fn', [self.params, self.V_MAX], [F_Q_inv]) 
                self.Q_inv_fn = ca.Function('Q_inv_fn', [self.params], [Q_inv]) 
                
    def _set_canon_form_mats(self): 
        #In online mode, vars_epi is not defined. So, set it to zero 
        self.F, self.f = self.F_fn(ca.DM.zeros(*self.vars_pol4screening.shape),self.opti.value(self.params),self.opti.value(self.V_MAX)), -ca.vertcat(*self.canon_prob_fn['f_l_i_c'](ca.DM.zeros(*self.vars_pol4screening.shape),self.opti.value(self.params),self.opti.value(self.V_MAX))) 
        self.L = self.l1_lmbd * self.L_fn(ca.DM.zeros(*self.vars_pol4screening.shape)) 
        C = self.C_fn_nonzero(self.opti.value(self.params),ca.DM.zeros(*self.slack_vec.shape)) # Evaluate C_fn to get the non-zero elements 
        
        #construct sparse C matrix 
        self.C = sp.csr_matrix((np.array(C).reshape(-1), (self.row_indices, self.col_indices)), shape=self.C_shape) 
        self.c = self.c_fn(self.opti.value(self.params)) 
            
        #Here, ca.hessian outputs hessian, J_grad. #J_grad = Q*theta + p. Thus, if we evaluate J_grad with theta = 0, we get p. Note that hessian is not dependent on theta 
        self.Q, self.p = self.canon_prob_fn['f_hessian'](ca.DM.zeros(*self.vars_pol4screening.shape),ca.DM.zeros(2*(self.N-1)*self.N_modes[0]*self.N_TV,1),self.opti.value(self.params),ca.DM.zeros(*self.slack_vec.shape)) 
        
    def solve(self,first_solve=False): 
        try: 
            if self.offline: 
                if self.vars_ws is not None: 
                    self.opti.set_initial(self.vars_pol, self.vars_ws) 
                if self.vars_epi_ws is not None: 
                    self.opti.set_initial(self.vars_epi, self.vars_epi_ws) 
                else: 
                    if self.vars_ws is not None: 
                        self.opti.set_initial(self.vars_pol4screening, self.vars_ws) 
                    if self.solver == "ipopt": 
                        if hasattr(self, 'sol'):
                            try: 
                                self.opti.set_initial(self.opti.lam_g, self.sol.value(self.opti.lam_g)) 
                            except: 
                                pass 
            # self.plot_obca_obstacles_and_ego_path(self.pos_tvs, self.psi_tvs, self.tv_params ) 
            st = time.time() 
            self.sol = self.opti.solve() 
            solve_time = time.time() - st 
            
            # Collect Optimal solution. 
            u_control = self.sol.value(self.policy[0][0]) 
            h_opt = self.sol.value(self.policy[0]).squeeze()
            u_opt = self.sol.value(self.policy[0]).reshape((1,-1)) 
            M_opt = self.sol.value(self.policy[1]) 
            K_opt = [[self.sol.value(self.policy[2][k][j]) for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            nom_z_tv = [[self.sol.value(self.nom_z_tv[k][j]) for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            s0 = self.sol.value(self.s0) 
            nom_z = self.sol.value(self.nom_z).reshape(-1,2).T 
            nom_z[0,:] += s0 
            
            if self.offline and not first_solve: 
                self.vars_ws , self.vars_epi_ws = self.sol.value(self.vars_pol), self.sol.value(self.vars_epi) 
            elif not self.offline and not first_solve: 
                self.vars_ws = self.sol.value(self.vars_pol4screening)             
            if self.offline and self.solver=='ipopt':
                #g1 dual 
                l1_duals=[[[[self.sol.value(self.opti.dual(self.l1_constr[k][j][t][0]))] for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
                gain_l1 = [[[self.sol.value(self.gain_l1[k][j][t]) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)]
                #soc collision avoidance dual 
                ca_duals=[[[[self.sol.value(self.opti.dual(self.ca_constr[k][m][t][0]))] for t in range(self.N-1)] for m in range(len(self.mode_map))] for k in range(self.N_TV)] 
            leading_vehicle_active = (self.sol.value(self.opti.dual(self.lead_vehicle_constr)) > 1e-3) 
            is_opt = True 
            self.cost_opt = self.sol.value(self.cost) 
            self.cost_wo_slack = self.sol.value(self.cost - 1e4*self.slack_vec.T@self.slack_vec)
            print(f'Optimal Cost: {self.cost_opt}')
            print(f'Cost without slack: {self.cost_wo_slack}')
            
        except:
            if self.offline: 
                self.vars_ws , self.vars_epi_ws = None, None
            else: 
                self.vars_ws = None 
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
                u_control = self.u_backup
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
            leading_vehicle_active = False 
        t_proc_sum = sum(value for key, value in self.opti.stats().items() if key.startswith('t_proc'))
        t_wall_sum = sum(value for key, value in self.opti.stats().items() if key.startswith('t_wall')) 
        # solve_time = sum(value for key, value in self.opti.stats().items() if key.startswith('t_wall_solver')) if self.solver == 'grb' else t_wall_sum 
        sol_dict = {} 
        sol_dict['nom_z'] = nom_z # nominal state predictions 
        sol_dict['u_control'] = u_control # control input to apply based on solution 
        sol_dict['u_opt'] = u_opt # optimal control sequence 
        sol_dict['optimal'] = is_opt # whether the solution is optimal or not 
        sol_dict['leading_vehicle_active'] = leading_vehicle_active 
        sol_dict['optimal_cost'] = self.cost_opt if is_opt else None 
        sol_dict['optimal_cost_wo_slack'] = self.cost_wo_slack if is_opt else None
        
        if not self.offline: sol_dict['gap_radius'] = self.gap
        if is_opt: 
            sol_dict['h_opt']=h_opt
            sol_dict['M_opt']=M_opt 
            sol_dict['K_opt']=K_opt 
            sol_dict['nom_z_tv']=nom_z_tv 
            
            if self.offline and self.solver == 'ipopt': 
                sol_dict['l1_duals']=l1_duals 
                sol_dict['ca_duals']=ca_duals 
        
        sol_dict['solve_time'] = solve_time # how long the solver took in seconds 
        print(f'Optimization Solve Time [{self.solver}]: {solve_time} s')
        sol_dict['t_wall_sum'] = t_wall_sum 
        sol_dict['t_proc_sum'] = t_proc_sum 
        sol_dict['vars'] = self.vars_kept 
        sol_dict['constr'] = self.constr_kept 
        #reconstruct self.constr_keep lists 
        if not self.offline: 
            gain_keep=[[self.opti.value(self.gain_keep[k][j]) for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            constr_keep=[[self.opti.value(self.constr_keep[k][m]) for m in range(len(self.mode_map))] for k in range(self.N_TV)] 
            # constr_keep=[[self.constr_keep[k][m] for m in range(len(self.mode_map))] for k in range(self.N_TV)] 
            print(f'Gain Keep: {gain_keep}') print(f'Constr Keep: {constr_keep}') 
            
            sol_dict['gain_keep'] = gain_keep 
            sol_dict['constr_keep'] = constr_keep 
        if self.offline: 
            sol_dict['computation_time'] = {'solve_time': solve_time} 
        else:
            sol_dict['computation_time'] = {'solve_time': solve_time, 'safety_screening': self.time_safety_screening, 'set_canon_form_mats': self.time_set_canon_form_mats, 'time_least_squares_formulation': self.time_least_squares_formulation, 'time_least_squares_solve': self.time_least_squares_solve} 
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
        self._update_leading_vehicle_params(update_dict['leading_vehicle'])
        if not self.offline:
            st_first = time.time()
            if 'l1_duals' in update_dict.keys():
                if 'canon_prob' in update_dict.keys():
                    print('[smpc.py]: Calling set_canon_form_mats,,,')
                    st = time.time()
                    self._set_canon_form_mats()
                    self.time_set_canon_form_mats = time.time()-st
                    print(f'[smpc.py]: set_canon_form_mats took {self.time_set_canon_form_mats} seconds')
                    print('[smpc.py]: Finished set_canon_form_mats,,,')
                print('[smpc.py]: Calling update_gain_and_constr_keeps')
                _,_,_,_= self.update_gain_and_constr_keeps(*[update_dict[key] for key in ['l1_duals', 'ca_duals']])
                print('[smpc.py]: Finished update_gain_and_constr_keeps,,,')
            else:
                self.update_gain_and_constr_keeps()
            self.time_safety_screening = time.time() - st_first
            print(f'[smpc.py]: Total safety screening took {self.time_safety_screening} seconds')
    def _update_leading_vehicle_params(self, leading_vehicle=None):
        if leading_vehicle is None:
            self.opti.set_value(self.lead_vehicle_s, 1e6)
        else:
            self.opti.set_value(self.lead_vehicle_s, leading_vehicle.progress)
    def _update_speed_limit(self, speed_limit=None):
        if speed_limit is None:
            self.opti.set_value(self.V_MAX, self.config['v_max'])
        else:
            sl = min(speed_limit,self.config['v_max'])
            print(f"[smpc.py]: Speed Limit: {speed_limit}, SL: {sl}")
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
        print(f'[smpc.py]: Initial Velocity: {x0[1]}')
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
        canon_prob_fn.update({'f_l_i_c':self.f_l_i_c, 'f_l_i_l1': self.f_l_i_l1, 'f_ca_i':self.f_ca_i, 'f_hessian':self.f_hessian,'f_cost':self.f_cost})
        return canon_prob_fn
    
    def _get_canon_form_fns_precomputed(self):
        '''
        Returns dictionary containing canonical form of the problem (offline mode only). Call in
        '''
        canon_prob_fn ={}
        if not self.offline:
            canon_prob_fn.update({'F_fn':self.F_fn, 'L_fn': self.L_fn, 'C_fn_nonzero':self.C_fn_nonzero, 'c_fn':self.c_fn,'row_indices':self.row_indices,'col_indices':self.col_indices,'C_shape':self.C_shape,'vars_pol4screening_shape':self.vars_pol4screening.shape})
        return canon_prob_fn                        

    def update_gain_and_constr_keeps(self, l1_duals=None, ca_duals=None):
        """
        Faster version (~5-10x): batch updates to CasADi, fewer Python ops.
        Preserves return values and side effects.
        """
        # Fast path: no screening → set all ones in batch and return quickly.
        if l1_duals is None:
            self.gap = None
            print(f"[smpc.py]: Gap Radius is {self.gap}")
            st = time.time()

            # Build dense ones once
            ones_vec = np.ones((self.N-1, 1), dtype=float)

            # Gain keeps: shape (N_TV x N_modes[k]) of parameter vectors
            for k in range(self.N_TV):
                for j in range(self.N_modes[k]):
                    self.opti.set_value(self.gain_keep[k][j], ones_vec)

            # Constraint keeps: shape (N_TV x len(mode_map)) of parameter vectors
            M = len(self.mode_map)
            for k in range(self.N_TV):
                for m in range(M):
                    self.opti.set_value(self.constr_keep[k][m], ones_vec)

            vars_kept = 2 * (self.N-1) * sum(self.N_modes)  # each gain_keep gives 2 scalars per t
            constr_kept = (self.N-1) * self.N_TV * M
            self.vars_kept = vars_kept
            self.constr_kept = constr_kept

            solve_time = time.time() - st
            print('[smpc.py]: Update Gain and Constraint Setting Keep Time: ', solve_time, ' s')
            return None, None, None, self.gap

        # --- With screening (original behavior) ---
        st_first = time.time()
        self.mu_dim = 2*self.N * (self.N_TV+1) + 1
        self.num_ca_duals = len(ca_duals)

        # Recover feasible duals (unchanged call)
        mu, eta, g1 = self.solve_dual_approximation(
            self.Q, self.L, self.F, self.p, self.f, self.c,
            np.repeat(ca_duals, self.mu_dim), l1_duals
        )
        self.gap = self._compute_gap_radius(mu, eta, g1)
        print(f"[smpc.py]: Gap Radius is {self.gap:.5f}")

        # Unflatten once (existing utility)
        l1_dual_dim = [self.N-1, self.N_modes, self.N_TV]
        ca_dual_dim = [self.N-1, len(self.mode_map), self.N_TV]
        l1_duals_list, ca_duals_list = unflatten_duals(
            np.expand_dims(np.concatenate([l1_duals, ca_duals]), axis=0),
            l1_dual_dim=l1_dual_dim, ca_dual_dim=ca_dual_dim
        )

        # Buffers to batch-set into CasADi (avoid per-scalar set_value)
        gain_keep_buf = [
            [np.zeros((self.N-1, 1), dtype=float) for _ in range(self.N_modes[k])]
            for k in range(self.N_TV)
        ]
        constr_keep_buf = [
            [np.zeros((self.N-1, 1), dtype=float) for _ in range(len(self.mode_map))]
            for _ in range(self.N_TV)
        ]

        # Screening thresholds (keep logic unchanged)
        TOL = 0.3
        M = len(self.mode_map)
        vars_kept = 0
        constr_kept = 0

        # We need a running index for each CA constraint's μ-slice inside 'mu'
        ca_constr_ind_counter = 0

        for k in range(self.N_TV):
            for m in range(M):
                j = self.mode_map[m][k]  # local mode index for TV k
                for t in range(self.N-1):
                    # ---- L1 gain screening ----
                    # Original "tertiary" check preserved:
                    if (0 in l1_duals_list[k][j][t]) or (2 in l1_duals_list[k][j][t]):
                        gain_keep_val = 1
                    else:
                        # slice of g1 for this (k,j,t): two entries
                        g1_slice = g1[
                            len(self.mode_map[m][:k])*2*(self.N-1) + j*2*(self.N-1) + 2*t :
                            len(self.mode_map[m][:k])*2*(self.N-1) + j*2*(self.N-1) + 2*(t+1)
                        ]
                        gain_keep_val = self._safe_screen(
                            g1_slice, gap_radius=self.gap, dual_type='l1_dual'
                        )
                    gain_keep_buf[k][j][t, 0] = gain_keep_val
                    if gain_keep_val:
                        vars_kept += 2  # two gain entries per time step

                    # ---- CA constraint screening ----
                    if ca_duals_list[k][j][t][0]:
                        constr_keep_val = 1
                    else:
                        mu_slice = mu[
                            self.mu_dim * ca_constr_ind_counter :
                            self.mu_dim * (ca_constr_ind_counter + 1)
                        ]
                        constr_keep_val = self._safe_screen(
                            mu_slice, gap_radius=self.gap, dual_type='ca_dual'
                        )
                    ca_constr_ind_counter += 1
                    constr_keep_buf[k][m][t, 0] = constr_keep_val
                    constr_kept += constr_keep_val

        # Single set_value per vector parameter (huge win)
        for k in range(self.N_TV):
            for j in range(self.N_modes[k]):
                self.opti.set_value(self.gain_keep[k][j], gain_keep_buf[k][j])

        for k in range(self.N_TV):
            for m in range(M):
                self.opti.set_value(self.constr_keep[k][m], constr_keep_buf[k][m])

        self.vars_kept = vars_kept
        self.constr_kept = constr_kept
        print(f"vars:  {vars_kept} out of {g1.shape[0]}, constr: {constr_kept} out of {ca_duals.shape[0]}")

        solve_time = time.time() - st_first
        print('[smpc.py]: Update Gain and Constraint Setting Keep Time: ', solve_time, ' s')
        return mu, eta, g1, self.gap
    
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

    def _eta_best_response_fast(self, mu, g, *, rho_scale=1e-4, iters=0): 
        """ 
        Solve min_{eta >= 0} 0.5 eta^T H eta + b^T eta 
        with H = F Q^{-1} F^T, b = F Q^{-1}(p + C^T mu + L^T(2g-1)) + f
        Uses either Cholesky of H+rho I (preferred) or PGD if iters>0. 
        """ 
        solve_Q = self._solve_Q # cached 
        F = self._F; C = self._C; L = self._L 
        p = np.asarray(self._p).ravel() 
        f = np.asarray(self._f).ravel() 
        a = p + (C.T @ mu).ravel() + (L.T @ (2.0*g.ravel() - 1.0))
        b = (F @ solve_Q(a)) + f 
        # b = F Q^{-1} a + f 
        # Linear operators for H 
        # def H_mv(x): 
        # return F @ solve_Q(F.T @ x) 
        n = F.shape[0] 
        
        if iters <= 0:
            # Factorize H + rho I (cheap & robust)
            # Estimate diag(H) for rho via a few probes (or use Hutchinson, §3) # Here: one cheap diagonal estimate 
            z = np.random.randn(n) 
            diagH_est = np.maximum(1e-10, (H_mv(z)*z).sum() / (z*z).sum()) * np.ones(n) 
            rho = rho_scale * np.median(diagH_est) 
            # Cholesky of (H + rho I) by CG-precompute columns via lsq_linear on R? Use sla.cg on normal eq? 
            # We just use 'lsq_linear' on R form like your original function did: # Build R,y s.t. 1/2||R eta + y||^2 equivalent (factorization via sla.cholesky on a small n) 
            # Dense path (small n): materialize H approx for robustness 
            # If n is large, consider switching to PGD below (iters>0). 
            import scipy.linalg as sla 
            # Build dense H̃ once (small n case) 
            # If n is large, call with iters>0 instead. 
            H_dense = np.zeros((n,n)) 
            E = np.eye(n) f
            for j in range(n): 
                H_dense[:,j] = H_mv(E[:,j]) 
                H_tilde = H_dense + rho*np.eye(n) 
                R = sla.cholesky(H_tilde, lower=False, check_finite=False) 
                y = sla.solve_triangular(R.T, b, lower=True, check_finite=False) 
                from scipy.optimize import lsq_linear 
                sol = lsq_linear(R, -y, bounds=(0.0, np.inf), lsmr_tol=1e-4, max_iter=200)
            return sol.x.reshape(-1,1) 
        
        # PGD alternative (no factorization), with diagonal preconditioning 
        # # Approx diag(H) by Hutchinson (see §3) 
        diagH = self._diag_Hutchinson(H_mv, n, probes=6) 
        invD = 1.0 / np.maximum(1e-9, diagH) 
        tau = 0.95 # precond 
        step eta = np.zeros(n) if not hasattr(self, "_eta_ws") else self._eta_ws.copy() 
        for _ in range(iters): 
            gk = H_mv(eta) + b eta = np.maximum(0.0, eta - tau * invD * gk)
            self._eta_ws = eta.copy() 
        return eta.reshape(-1,1) 
            
    def solve_dual_approximation(self, Q, L, F, C, p, f, c, ca_dual, l1_dual): 
        """ 
        Solves regularized dual problem via least squares with constraint projection using LinearOperators (no explicit Q^{-1} or dense A). 
        Keeps verbose prints and stores timing/metadata on self. Returns feasible (mu, eta, g1) approximation. 
        """
        # -------- small helper to force 1-D numpy -------- 
        def _np1d(v): 
            return np.asarray(v, dtype=float).ravel() 
        
        # --------------------------- # Dimensions & quick logging # --------------------------- 
        st_first = time.time() 
        # Ensure C,F,L are scipy sparse (avoid any CasADi types) 
        C = C if sp.issparse(C) else sp.csr_matrix(np.asarray(C, dtype=float))
        F = F if sp.issparse(F) else sp.csr_matrix(np.asarray(F, dtype=float)) 
        L = L if sp.issparse(L) else sp.csr_matrix(np.asarray(L, dtype=float)) 
        n_mu = C.shape[0] 
        n_eta = F.shape[0] 
        n_g1 = L.shape[0] 
        m = n_mu + n_eta + n_g1 
        print(f"[Dual Approx] Dimensions: n_mu={n_mu}, n_eta={n_eta}, n_g1={n_g1}, total={m}") 
        
        # --------------------------- # Factorize Q once, reuse solve # ---------------------------
        t0 = time.time() 
        # Q_csc = sp.csc_matrix(Q) # factorization expects CSC 
        Q_csc = sp.csc_matrix(Q + 1e-8*sp.eye(Q.shape[0])) 
        solve_Q = spla.factorized(Q_csc) # closure: solve_Q(b) solves Q x = b (expects np.ndarray) 
        t_fac = time.time() - t0 
        print(f"[Dual Approx] Factorized Q in {t_fac:.3f}s")
        
        # Store for gap computation later 
        # self._solve_Q = solve_Q 
        # self._C, self._F, self._L = C, F, L 
        # self._p, self._c, self._f = p, c, f 
        self.blocks = [self.mu_dim - 1] * self.num_ca_duals
        
        # Ensure p,c,f are 1-D numpy 
        p = _np1d(p)
        c = _np1d(c) 
        f = _np1d(f) 
        
        # --------------------------- # Helper ops: C Q^{-1} v, etc. (always 1‑D in/out) # --------------------------- 
        
        def CQinv(v): w = solve_Q(_np1d(v))
            return _np1d(C @ w) 
        
        def FQinv(v): w = solve_Q(_np1d(v)) 
            return _np1d(F @ w) 
        
        def LQinv(v): 
            w = solve_Q(_np1d(v)) 
            return _np1d(L @ w) 
        # --------------------------- # Build RHS b = [b1; b2; b3] without materializing Q^{-1} # ---------------------------
        ones_ng1 = np.ones((n_g1, 1))
        pmvec = p.reshape(-1, 1) - (L.T @ ones_ng1) # (n_vars, 1) 
        pmvec = _np1d(pmvec) # -> (n_vars,) 
        
        b1 = _np1d(-c - CQinv(pmvec)) 
        b2 = _np1d(-f - FQinv(pmvec)) 
        b3 = _np1d(-2.0*LQinv(pmvec)) 
        b = _np1d(np.concatenate([b1, b2, b3])) 
        
        self.time_least_squares_formulation = time.time() - st_first 
        print(f"[Dual Approximation] Least Squares Formulation Time: " f"{self.time_least_squares_formulation:.6f} s (Depends on the Hessian/cost)")
        
        # --------------------------- # LinearOperator for A (symmetric): # y = A * [x1; x2; x3] via block matvec and Q-solves # Preallocate and fill slices instead of concatenate (shape‑safe). # ---------------------------
        def A_mv(x): 
            x = _np1d(x) 
            x1 = x[:n_mu] 
            x2 = x[n_mu:n_mu+n_eta] 
            x3 = x[n_mu+n_eta:] 
            y = np.empty(m, dtype=float) # three Q^{-1} applications 
            y[:n_mu] = _np1d(CQinv(C.T @ x1) + CQinv(F.T @ x2) + 2.0 * CQinv(L.T @ x3)) 
            y[n_mu:n_mu+n_eta]= _np1d(FQinv(C.T @ x1) + FQinv(F.T @ x2) + 2.0 * FQinv(L.T @ x3)) 
            y[n_mu+n_eta:] = _np1d(2.0 * LQinv(C.T @ x1) + 2.0 * LQinv(F.T @ x2) + 4.0 * LQinv(L.T @ x3)) 
            return y 
        
        def A_rmv(y):
            y = _np1d(y) 
            y1 = y[:n_mu] 
            y2 = y[n_mu:n_mu+n_eta] 
            y3 = y[n_mu+n_eta:] 
            # one Q^{-1} application for the adjoint 
            s = _np1d(C.T @ y1 + F.T @ y2 + 2.0 * L.T @ y3) 
            z = _np1d(solve_Q(s)) 
            
            out = np.empty(m, dtype=float) 
            out[:n_mu] = _np1d(C @ z) 
            out[n_mu:n_mu+n_eta] = _np1d(F @ z) 
            out[n_mu+n_eta:] = _np1d(2.0 * L @ z) 
            return out A_op = spla.LinearOperator((m, m), matvec=A_mv, rmatvec=A_rmv, dtype=float)
        
        # --------------------------- # RAID‑Net column reduction # (masks are already expanded to exact lengths) # --------------------------- 
        reduced_ls = self.config['reduced_ls'] 
        print(f"[Dual Approx] Reduced LS: {reduced_ls}") 
        if reduced_ls: keep_mu = np.asarray(ca_dual, dtype=bool) # length n_mu 
            keep_eta = np.ones(n_eta, dtype=bool) # keep all eta 
            keep_g1 = np.asarray(l1_dual, dtype=bool) # length n_g1 
            keep = np.concatenate([keep_mu, keep_eta, keep_g1]) 
            idx = np.flatnonzero(keep) 
        else: 
            keep = np.ones(m, dtype=bool) 
            idx = np.arange(m, dtype=int) 
        self.dual_dims = {"n_mu": int(n_mu), "n_eta": int(n_eta), "n_g1": int(n_g1), "m_total": int(m)}
        self.reduced_cols = int(idx.size) 
        self.total_cols = int(m)
        self.keep_mask = keep.copy() 
        self.keep_idx = idx.copy() 
        print(f"[Dual Approx] Kept {self.reduced_cols} / {self.total_cols} dual columns after RAID‑Net reduction") 
        
        # Selection operator S: x_full = S @ x_red 
        S = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(m, idx.size)) # Reduced operator Ar(xr) = A(S xr), Ar^T(y) = S^T A(y)
        # Bind A_op and S as default args to avoid late‑binding issues. 
        # reduced operator 
        def Ar_mv(xr, A_op=A_op, S=S): 
            return _np1d(A_op.matvec(_np1d(S @ xr))) 
        
        def Ar_rmv(y, A_op=A_op, S=S): 
            # Ar^T y = S^T A^T y 
            return _np1d(S.T @ _np1d(A_op.rmatvec(_np1d(y))))
        
        Ar = spla.LinearOperator((m, idx.size), matvec=Ar_mv, rmatvec=Ar_rmv, dtype=float) 
        
        # --------------------------- # LSQR solve on reduced system # --------------------------- 
        ls_tol = 1e-6 
        ls_max_iter = 5000 
        # warm start (seed g1 part with 0.5)
        ls_init_guess = np.zeros(Ar.shape[1], dtype=float) 
        if reduced_ls: 
            start_idx = int(np.sum(keep_mu)) + n_eta # kept mu + all eta 
            ls_init_guess[start_idx:] = 0.5 # g1 region 
        else: 
            ls_init_guess[n_mu+n_eta:] = 0.5 #0.5 
        # quick asserts before lsqr: 
        assert b.ndim == 1 and b.shape[0] == Ar.shape[0]
        assert ls_init_guess.ndim == 1 and ls_init_guess.shape[0] == Ar.shape[1] st = time.time() 
        # lsqr returns (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm) 
        x_red, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr( Ar, b, atol=ls_tol, btol=ls_tol, iter_lim=ls_max_iter, x0=ls_init_guess ) 
        self.time_least_squares_solve = time.time() - st 
        print(f"[Dual Approximation] LSQR finished in {self.time_least_squares_solve:.6f} s " f"with system {Ar.shape}, iters={itn}, istop={istop}") 
        print(f"[Dual Approximation] Residuals: r1norm={r1norm:.3e}, r2norm={r2norm:.3e}, " f"||A||≈{anorm:.3e}, cond≈{acond:.3e}, arnorm={arnorm:.3e}, ||x||={xnorm:.3e}") 
        
        # --------------------------- # Scatter back, split, and project # --------------------------- 
        x_full = np.zeros(m, dtype=float) 
        x_full[idx] = x_red 
        x = x_full 
        
        mu = self._proj_soc_dual_stacked_np(x[:n_mu], self.blocks).flatten()
        eta, g1 = x[n_mu:n_mu+n_eta], x[n_mu+n_eta:] 
        return mu, eta, g1 
    
    def _compute_gap_radius(self, f_mu, f_nu, f_g): 
        """ gap_radius = ||dual - Proj(dual - grad_d)|| * (1+sigma)/eta 
        where 
        grad_d = [C; F; 2L] Q^{-1} (C^T μ + F^T η + L^T(2g-1) + p) + [-c; f; 0] 
        with eta = 1/largest_eig(Q), sigma = 1/smallest_eig(Q). 
        """ 
        # ---- tiny helper: force 1-D numpy ---- 
        _np1d = lambda v: np.asarray(v, dtype=float).ravel() 
        
        # ---- inputs as 1-D numpy ---- 
        f_mu = _np1d(f_mu) 
        st = time.time() 
        eta_br = self._eta_best_response_fast(f_mu.reshape(-1,1), f_g.reshape(-1,1), rho_scale=1e-4, iters=0) # or iters=10 for PGD 
        print(f'[smpc.py]: Best Response Time: {time.time()-st:.6f} s') 
        f_nu = eta_br.ravel() 
        f_nu = _np1d(f_nu) 
        f_g = _np1d(f_g)
        
        # ---- sparse handles (set by solve_dual_approximation) ---- 
        C = self._C # scipy.sparse 
        F = self._F 
        L = self._L 
        p = _np1d(self._p) 
        c = _np1d(self._c) 
        f = _np1d(self._f) # ---- eigenvalues of Q (cache & print timing like before) ---- 
        st = time.time() 
        Q = sp.csr_matrix(self.Q) 
        # largest 
        self.largest_eig = spla.eigsh( Q, k=1, which='LM', v0=(self.largest_eig*np.ones(Q.shape[0]) if hasattr(self, 'largest_eig') else None), return_eigenvectors=False )[0] 
        # smallest (shift-invert around 0) 
        self.smallest_eig = spla.eigsh( Q, k=1, which='LM', sigma=0, v0=(self.smallest_eig*np.ones(Q.shape[0]) if hasattr(self, 'smallest_eig') else None), return_eigenvectors=False )[0]
        solve_time = time.time() - st 
        # print(f'[smpc.py]: Hessian Eigenvalue Computation Time: {solve_time:.6f} s')
        eta, sigma = self.largest_eig**(-1), self.smallest_eig**(-1) 
        
        # ---- gradient of dual objective (no explicit stacks, use Q^{-1} apply) ---- 
        st = time.time() 
        ones_fg = np.ones_like(f_g)
        temp = (C.T @ f_mu) + p + (F.T @ f_nu) + (L.T @ (2.0*f_g - ones_fg))
        temp = _np1d(temp)
        # Solve Q u = temp (uses cached factorization)
        u = _np1d(self._solve_Q(temp)) 
        # grad_d blockwise: 
        g1 = _np1d((C @ u)) 
        g2 = _np1d(F @ u) 
        g3 = _np1d(2*(L @ u)) 
        grad_d = np.concatenate([g1 + c, g2 + f, g3])
        self.gradient_computation_time = time.time() - st print(f'[smpc.py]: Dual gradient build time: {self.gradient_computation_time:.6f} s') 
        
        # ---- one projected step (alpha=1; equivalent to your proj_dual = dual - grad_d) ---- 
        dual = np.concatenate([f_mu, f_nu, f_g])
        #CHEATING 
        # grad_d[C.shape[0]:C.shape[0]+F.shape[0]] = np.maximum(grad_d[C.shape[0]:C.shape[0]+F.shape[0]],0) 
        proj_dual = dual - grad_d 
        
        # ---- projection to dual feasible set ---- 
        n_mu = C.shape[0] 
        n_eta = F.shape[0] 
        # μ ∈ SOC* (blockwise) 
        # self.blocks should be [self.mu_dim - 1] * self.num_ca_duals (set earlier). 
        # proj_dual[:n_mu] = self._proj_soc_dual_stacked_np(proj_dual[:n_mu], self.blocks).flatten() 
        proj_dual[:n_mu] = self._proj_normal_soc_stacked_vecfirst(proj_dual[:n_mu], -(g1-c), self.blocks, tol=1e-9).flatten() 
        # η ≥ 0 
        proj_dual[n_mu:n_mu+n_eta] = np.maximum(proj_dual[n_mu:n_mu+n_eta], 0.0) 
        # 0 ≤ g ≤ 1
        proj_dual[n_mu+n_eta:] = np.clip(proj_dual[n_mu+n_eta:], 0.0, 1.0) 
        
        # ---- gap radius ---- 
        st = time.time() 
        gap = np.linalg.norm(dual - proj_dual) * (1.0 + sigma) / eta 
        solve_time_norm = time.time() - st
        print(f'[smpc.py]: Gap norm time: {solve_time_norm:.6f} s') 
        print(np.linalg.norm((dual - proj_dual)[:n_mu]))
        print(np.linalg.norm((dual - proj_dual)[n_mu:n_mu+n_eta]))
        print(np.linalg.norm((dual - proj_dual)[n_mu+n_eta:])) 
        return gap 
                              
    def _split_slack_vecfirst(self, s_stack, blocks): """ s_stack holds concatenated (z_i, y_i) blocks from Cθ+c (or grad_dual slice). Returns (y_list, z_list) aligned with 'blocks'. """ s = np.asarray(s_stack, float).ravel() y_list, z_list = [], [] idx = 0 for n in blocks: z_i = s[idx: idx+n] y_i = s[idx+n] z_list.append(z_i.copy()) y_list.append(float(y_i)) idx += n + 1 return y_list, z_list 
    
    def _proj_normal_soc_stacked_vecfirst(self, mu_stack, s_stack, blocks, tol=1e-10): 
        """ 
        Project stacked μ (z,y per block) onto ⨅_i N_K(s_i),
        where 
        s_stack supplies current slacks (z,y per block) in the same order. 
        """ 
        mu = np.asarray(mu_stack, float).ravel()
        y_list, z_list = self._split_slack_vecfirst(s_stack, blocks) 
        out = [] 
        idx = 0 
        for n, y_i, z_i in zip(blocks, y_list, z_list): 
            mu_i = mu[idx: idx+n+1] 
            mu_hat = self._proj_normal_soc_block_vecfirst(mu_i, y_i, z_i, tol=tol) 
            out.append(mu_hat) 
            idx += n + 1 
        return np.concatenate(out)[:, None] 
        
    def _proj_normal_soc_block_vecfirst(self, mu_zy, y, z, tol=1e-10): 
        """ 
        Project unconstrained μ block (z,y) onto the normal cone N_K(s) at s=(y,z). 
        Returns μ̂ in (z,y) order. 
        """ 
        z = np.asarray(z, float); y = float(y) 
        nz = float(np.linalg.norm(z, 2))
        phi = nz - y # <0: interior, =0: boundary, >0: (slightly) infeasible 
        # Interior: μ = 0 
        if phi < -tol: 
            return np.zeros_like(mu_zy) 
        
        # Apex: N_K(0) = -K 
        if nz < tol and abs(y) < tol: 
            return self._proj_minusK_block_vecfirst(mu_zy) 
        
        # Boundary (and small violation): project onto ray u=(z/nz, -1) 
        u_z = z / (nz + 1e-16)
        u_y = -1.0 
        dot = float(np.dot(mu_zy[:-1], u_z) + mu_zy[-1] * u_y) # ⟨μ, u⟩ 
        # Ray projection: μ̂ = max(0, ⟨μ,u⟩) / ||u||^2 * u, ||u||^2 = 2 tau = max(0.0, dot) * 0.5 
        mu_hat = np.empty_like(mu_zy) 
        mu_hat[:-1] = tau * u_z 
        mu_hat[-1] = tau * u_y 
        return mu_hat 
    
    def _proj_soc_np(self, t, x, eps=1e-12): 
        """ 
        Project (t, x)
        onto K = { (tau, y): ||y||_2 <= tau }. 
        Returns (t_proj, x_proj).
        """ 
        x = np.asarray(x, dtype=float) 
        t = float(t) 
        nx = np.linalg.norm(x, 2) 
        
        if nx <= t: 
            return t, x 
        if nx <= -t: 
            return 0.0, np.zeros_like(x)
        
        # middle case: here nx > 0, so division is safe; keep exact alpha
        alpha = 0.5 * (nx + t)
        scale = alpha / max(nx, eps) 
        return alpha, scale * x 
    
     def _proj_soc_dual_stacked_np(self, vec, blocks, eps=1e-12):
         """ 
         Project a stacked vector with blocks [(x_i, t_i)] onto (-K)^m.
         'blocks' holds each x_i dimension n_i. 
         Returns a column vector with the same layout: [(x_i^proj, t_i^proj)]. 
         """
         vec = np.asarray(vec, dtype=float).ravel() 
         out = [] 
         idx = 0 
         for n in blocks:
            x = vec[idx: idx+n] 
            t = vec[idx+n]
            # project (t,x) onto -K via sign symmetry: Π_{-K}(t,x) = - Π_K(-t,-x)
            tp_K, xp_K = self._proj_soc_np(-t, -x, eps=eps) # onto K 
            tp = -tp_K xp = -xp_K 
            out.append(xp) 
            out.append(np.array([tp]))
            idx += n + 1 
        return np.concatenate(out)[:, None]
    
    def _safe_screen(self,dual, gap_radius, dual_type = "ca_dual"):
        keep = 1 
        # if hasattr(self, 'cost_wo_slack'): 
        # TOL = abs(self.cost_wo_slack)*0.20 #TODO: should it be dynamic? optimal cost changes as states and scenario change 
        # # else: 
        TOL = 0.3 
        
        if dual_type == "ca_dual": 
            # Do ca_dual sensitivity-based screen (l2 norm of mu) 
            if np.linalg.norm(dual, ord=2) + gap_radius < TOL: 
                keep = 0 
        else: 
            ''' Practically, we are taking half of the dual vector g that corresponds to the upperbound z_k,i >= S\theta in eq(4) and calling it g1. 
            Thus, when applying the safety screening condition ||g||_inf + gap_radius < 1, we are also checking the condition 0 < min_i(g1) - gap_radius because when g1 = 0, the corresponding g2 component is 1 as g1 + g2 = 1 (see the paper). 
            ''' 
            if gap_radius < TOL and (np.linalg.norm(dual,ord=np.inf) + gap_radius < 1) and (min(abs(dual)) - gap_radius > 1e-3):
                keep = 0
        return keep #output 1 or 0 