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
from scipy.sparse import block_diag, vstack 
from scipy.sparse.linalg import lsmr, svds, lsqr
import scipy.sparse as sp 
import torch 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    
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

        self.tight=2.3 

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
        self.dpos_tvs=[[[self.opti.parameter(2,1) for _ in range(self.N)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)] 
        self.Qs=[[[self.opti.parameter(2,2) for _ in range(self.N)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)] 
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
        self._update_tv_params([[4.47, 2]]*self.N_TV) 
        self._update_red_light(None,None) 
        self._update_speed_limit(self.config['v_max']) 
        self._update_leading_vehicle_params(None) 
        if not self.offline: 
            _,_,_,_ = self.update_gain_and_constr_keeps() 
        self.solve(first_solve=True) 

        self._constr_keep_buf = None
        self._gain_keep_buf = None
        self._prev_constr_keep = None
        self._prev_gain_keep = None
        
    def _return_policy_class(self):

        """
        EV Affine disturbance feedback + TV state feedback policies from https://arxiv.org/abs/2109.09792
        """ 
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

        self.vars_pol = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]))

        if self.offline:
            self.vars_epi = ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(self.gain_l1[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)])
        else:
            self.vars_pol4screening = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K4screening[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]))
        
        #Variables for warmstart
        self.vars_ws, self.vars_epi_ws  = None, None 
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
        for t in range(1,self.N): 
            self.opti.subject_to(self.opti.bounded(0,nom_s[t],self.redlight[0]-self.s0)) 
                
        '''
        Collision avoidance constraints
        '''
        for k in range(self.N_TV):
            for j in range(len(self.mode_map)):
                m = self.mode_map[j][k]
                # (unchanged) gain regularizer
                cost += self.multiplier * 0.019 * 1.5 * ca.trace(
                    K[k][m] @ E_tv[k][m][:2*self.N, :] @ E_tv[k][m][:2*self.N, :].T @ K[k][m].T
                )

                # --------- AFFINE collision-avoidance: batch all t constraints ----------
                if self.config['collision_avoidance_method'] == 'affine':

                    soc_rows = []         # holds [ y+slack - z_norm ; y+slack ] for all t
                    soc_rows_online = []  # masked rows for online screening (same length)
                    soc_rows_gurobi = []  # masked rows for gurobi (same length)
                    soc_rows_gurobi_online = []  # masked rows for gurobi online (same length)
                    # Also batch L1 constraints: stack K entries and gains once
                    if len(self.l1_constr[k][m][0]) == 0:
                        # # Build stacked vectors: shape (2*(N-1), 1)
                        K_stack_vec = ca.vertcat(*[
                            ca.vec(K[k][m][t, 2*t:2*(t+1)].T)  # (2x1)
                            for t in range(1, self.N)
                        ])
                        gain_l1_stack = ca.vertcat(*[
                            ca.vec(self.gain_l1[k][m][t-1].T)  # (2x1)
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
                        base = (oa_ref - self.pos_tvs[k][m][:, t]).T @ self.Qs[k][m][t-1]

                        # z parts
                        ev_rand = self.dpos[t-1] @ (B[2*t, :] @ M + E[2*t, :])
                        tv_rand = []
                        for l in range(self.N_TV):
                            term = self.dpos[t-1] @ B[2*t, :] @ K[l][self.mode_map[j][l]] @ E_tv[l][self.mode_map[j][l]][:2*self.N, :]
                            if l == k:
                                term = term - self.dpos_tvs[k][m][t-1] @ E_tv[k][m][2*t, :]
                            tv_rand.append(term)
                        z = (base @ ca.hcat([ev_rand] + tv_rand)).T   # one hcat, one @
                        z_norm = self.tight * ca.sqrt(ca.sumsqr(z) + 1e-10)

                        y = (
                            base
                            @ (self.x_pos[:, t] - oa_ref
                            + self.dpos[t-1] * (A[2*t, :] @ self.z_curr + B[2*t, :] @ h - (self.z_lin[0, t] - self.s0)))
                        )
                        if self.offline:
                            self.ca_ineq.append(ca.vertcat(z,y))
                            self.ca_constr[k][j][t-1]+=[z_norm<=y+self.slack[k][m][t-1],0<=y+self.slack[k][m][t-1]]
                            if self.solver == 'ipopt':
                                self.opti.subject_to(self.ca_constr[k][j][t-1][0])
                                self.opti.subject_to(self.ca_constr[k][j][t-1][1])
                                if len(self.l1_constr[k][m][t-1])==0:
                                    self.lin_ineq_l1+=[K[k][m][t,2*t:2*(t+1)]-self.gain_l1[k][m][t-1]] #only the first constraint: g1
                                    self.l1_constr[k][m][t-1]+=[K[k][m][t,2*t:2*(t+1)]<=self.gain_l1[k][m][t-1], -self.gain_l1[k][m][t-1]<=K[k][m][t,2*t:2*(t+1)]]
                                    self.opti.subject_to(self.l1_constr[k][m][t-1][0])
                                    self.opti.subject_to(self.l1_constr[k][m][t-1][1])
                            else:
                                 self.lin_ineq_l1+=[K[k][m][t,2*t:2*(t+1)]-self.gain_l1[k][m][t-1]] #only the first constraint: g1
                                 
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
                            if self.solver == 'gurobi':
                                self.opti.subject_to(soc_rows_gurobi[-1] > 0)
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
            self.F_fn = ca.Function('F_fn', [self.vars_pol4screening, self.params, self.V_MAX], [ca.jacobian(ca.vertcat(*self.canon_prob_fn['f_l_i_c'](self.vars_pol4screening,self.params,self.V_MAX)),self.vars_pol4screening)], {'post_expand': True}) 
            self.L_fn = ca.Function('L_fn', [self.vars_pol4screening], [ca.jacobian(ca.vertcat(*self.canon_prob_fn['f_l_i_l1'](self.vars_pol4screening,vars_epi)), self.vars_pol4screening)]) 
            self.C_fn = ca.Function('C_fn',[self.params, self.slack_vec],[ca.substitute(ca.jacobian(ca.simplify(ca.vertcat(*self.canon_prob_fn['f_ca_i'](self.vars_pol4screening, self.params))), self.vars_pol4screening), self.vars_pol4screening, ca.DM.zeros(*self.vars_pol4screening.shape))]) 
            C = self.C_fn(np.ones(self.params.shape),ca.DM.zeros(*self.slack_vec.shape)) # Evaluate C_fn to get the shape and non-zero indices 
            self.C_shape = C.shape 
            self.nonzero_inds = np.nonzero(np.ravel(C,order='F'))[0] 
            
            # Get the row and column indices of the non-zero elements in the sparse matrix 
            self.row_indices, self.col_indices = np.unravel_index(self.nonzero_inds, self.C_shape,order='F') 
            self.nonzero_vec = ca.vec(ca.substitute(ca.jacobian(ca.simplify(ca.vertcat(*self.canon_prob_fn['f_ca_i'](self.vars_pol4screening, self.params))), self.vars_pol4screening), self.vars_pol4screening, ca.DM.zeros(*self.vars_pol4screening.shape)))[self.nonzero_inds] 
            # self.C_fn_nonzero = ca.Function('C_fn_nonzero',[self.params, self.slack_vec],[self.nonzero_vec]) 
            self.nonzero_vec = ca.substitute(self.nonzero_vec, self.slack_vec, ca.DM.zeros(*self.slack_vec.shape)) # remove slack dependence
            # --- build the nonzero-value function with graph expansion (no JIT) ---
            # (inside the not self.offline block, right after you set self.nonzero_vec)
            self.C_fn_nonzero = ca.Function(
                'C_fn_nonzero',
                [self.params],                      # no slack arg
                [self.nonzero_vec],
                {'post_expand': True}                  # <-- important for runtime speed
            )

            # --- one-time CSR skeleton (perm from (row,col)->CSR order) ---
            rows = np.asarray(self.row_indices, dtype=np.int32)
            cols = np.asarray(self.col_indices, dtype=np.int32)
            nnz  = rows.size
            coo  = sp.coo_matrix((np.arange(nnz, dtype=np.int32), (rows, cols)), shape=self.C_shape)
            csr  = coo.tocsr()
            self._perm_coo2csr = csr.data.copy().astype(np.int32)

            self.C = sp.csr_matrix(
                (np.zeros(nnz, dtype=float), csr.indices.copy(), csr.indptr.copy()),
                shape=self.C_shape
            )            
            self.c_fn = ca.Function('c_fn',[self.params],[ca.vertcat(*self.canon_prob_fn['f_ca_i'](ca.DM.zeros(*self.vars_pol4screening.shape), self.params))], {'post_expand': True}) 
            Q, p = self.canon_prob_fn['f_hessian'](ca.DM.zeros(*self.vars_pol4screening.shape),vars_epi,self.params,ca.DM.zeros(*self.slack_vec.shape)) 
            
            #Q is constant, p is affine in params
            Q_num, _ = self.canon_prob_fn['f_hessian'](ca.DM.zeros(*self.vars_pol4screening.shape),vars_epi,ca.DM.ones(*self.params.shape),ca.DM.zeros(*self.slack_vec.shape))
            Q_csc = sp.csc_matrix(Q_num + 1e-8*sp.eye(Q_num.shape[0])) 
            solve_Q = spla.factorized(Q_csc) # closure: solve_Q(b) solves Q x = b (expects np.ndarray) 
            self._solve_Q = solve_Q 
    
            self.Q = sp.csr_matrix(np.asarray(Q_num, dtype=float))

            # -----------------------------------------
            # Cache spectral bounds once (Q is constant across replans)
            if not hasattr(self, "_eta_inv"):
                eval_max = spla.eigsh(self.Q, k=1, which='LM', return_eigenvectors=False)[0]
                eval_min = spla.eigsh(self.Q, k=1, which='LM', sigma=0, return_eigenvectors=False)[0]
                self._eta_inv = 1.0 / float(eval_max)   # = 1 / λ_max(Q)
                self._sigma   = 1.0 / float(eval_min)   # = 1 / λ_min(Q)

            Q_inv = ca.inv(Q) 
            L = self.l1_lmbd * self.L_fn(ca.DM.zeros(*self.vars_pol4screening.shape)) 
            self.L = sp.csr_matrix(np.asarray(L, dtype=float)) 
            
            ### ------------------- Accelerate p and f computation ------------------- ###
            # Select only the nonzero entries you care about (you already have these):
            p_sym = p
            _, p_nz = self.canon_prob_fn['f_hessian'](ca.DM.zeros(*self.vars_pol4screening.shape),vars_epi,ca.DM.ones(*self.params.shape),ca.DM.zeros(*self.slack_vec.shape)) 
            self.p_nonzero_inds = np.nonzero(np.ravel(p_nz,order='F'))[0] 
            idx = self.p_nonzero_inds

            # Affine decomposition: p(params) = p_0 + Jp * params
            Jp_sym = ca.jacobian(p_sym, self.params)

            p_0_fn = ca.Function('p_0_fn', [self.params], [p_sym])
            Jp_fn  = ca.Function('Jp_fn',  [self.params], [Jp_sym])

            p_0_dm = p_0_fn(ca.DM.zeros(self.params.shape))
            Jp_dm  = Jp_fn(ca.DM.zeros(self.params.shape))

            # Evaluate and cache as NumPy/SciPy once
            self._p_0  = np.asarray(p_0_dm).ravel() 
            self._Jp   = sp.csr_matrix(np.asarray(Jp_dm))

            # Preallocate the full p buffer once
            self._p_full = np.zeros(int(self.vars_pol4screening.shape[0]), dtype=float)
            self._p_idx  = np.array(idx, dtype=int)    
            
            f = -ca.vertcat(*self.canon_prob_fn['f_l_i_c'](ca.DM.zeros(*self.vars_pol4screening.shape),self.params,self.V_MAX))                   
            f_nz = -ca.vertcat(*self.canon_prob_fn['f_l_i_c'](ca.DM.zeros(*self.vars_pol4screening.shape),ca.DM.ones(*self.params.shape),1))
            self.f_nonzero_inds = np.nonzero(np.ravel(f_nz,order='F'))[0]
            idx = self.f_nonzero_inds
            Jf_sym = ca.jacobian(f, ca.vertcat(*[self.params, self.V_MAX]))
            Jf_sym_param = ca.jacobian(f, self.params)
            Jf_sym_vmax = ca.jacobian(f, self.V_MAX)

            f_0_fn = ca.Function('f_0_fn', [self.params, self.V_MAX], [f])
            Jf_param_fn = ca.Function('Jf_parama_fn', [self.params,self.V_MAX], [Jf_sym_param])
            Jf_vmax_fn = ca.Function('Jf_Vmax_fn', [self.params,self.V_MAX], [Jf_sym_vmax])
            f_0_dm = f_0_fn(ca.DM.zeros(self.params.shape), 0)
            Jf_param_dm = Jf_param_fn(ca.DM.zeros(self.params.shape), 0)
            Jf_vmax_dm = Jf_vmax_fn(ca.DM.zeros(self.params.shape), 0)
            self._f_0 = np.asarray(f_0_dm).ravel()
            self._Jf_param = sp.csr_matrix(np.asarray(Jf_param_dm))
            self._Jf_vmax = sp.csr_matrix(np.asarray(Jf_vmax_dm))
            
            self._f_full = np.zeros(int(f_0_dm.shape[0]), dtype=float)
            self._f_idx = np.array(idx, dtype=int)
              
              
    def _set_canon_form_mats(self): 
        #In online mode, vars_epi is not defined. So, set it to zero 
        par_val = self.opti.value(self.params)
        # self.F, self.f = self.F_fn(ca.DM.zeros(*self.vars_pol4screening.shape),par_val,self.opti.value(self.V_MAX)), -ca.vertcat(*self.canon_prob_fn['f_l_i_c'](ca.DM.zeros(*self.vars_pol4screening.shape),self.opti.value(self.params),self.opti.value(self.V_MAX))) 
        # self.f = np.array(self.f).ravel()
        st = time.time()
        self.F = self.F_fn(ca.DM.zeros(*self.vars_pol4screening.shape),par_val,self.opti.value(self.V_MAX))
        self.F = sp.csr_matrix(np.array(self.F))
        print(f"computing F time: {time.time()-st:.4f} seconds")
        st = time.time()
        self.f = self._f_full
        self.f = (self._f_0
                + (self._Jf_param @ par_val)
                + (self._Jf_vmax.toarray().ravel() * self.opti.value(self.V_MAX)))
        self.f = np.array(self.f).ravel()
        print(f"computing f time: {time.time()-st:.4f} seconds")
        
        #construct sparse C matrix 
        # st = time.time()           
        # C = self.C_fn_nonzero(par_val, ca.DM.zeros(*self.slack_vec.shape))
        # print(f"computing nonzero C elements time: {time.time()-st:.4f} seconds")
        # print(f"nonzero C elements: {C.shape[0]} out of {self.C_shape[0]*self.C_shape[1]} total elements")
        # st = time.time()
        # self.C = sp.csr_matrix((np.array(C).reshape(-1), (self.row_indices, self.col_indices)), shape=self.C_shape) 
        # print(f"computing C matrix time: {time.time()-st:.4f} seconds")

        st = time.time()
        Cvals = np.asarray(self.C_fn_nonzero(par_val)).ravel()
        self.C.data[:] = Cvals[self._perm_coo2csr]  # reorder to CSR
        print(f"computing nonzero C elements time: {time.time()-st:.4f} seconds")
        print(f"nonzero C elements: {Cvals.size} out of {self.C_shape[0]*self.C_shape[1]} total elements")

        st = time.time()
        self.c = self.c_fn(par_val) 
        print(f"computing c time: {time.time()-st:.4f} seconds")
            
        #Here, ca.hessian outputs hessian, J_grad. #J_grad = Q*theta + p. Thus, if we evaluate J_grad with theta = 0, we get p. Note that hessian is not dependent on theta 
        st = time.time()
        self.p = self._p_full
        self.p[self._p_idx] = self._p_0[self._p_idx] + (self._Jp @ par_val)[self._p_idx]
        
        self.p = self._np1d(self.p)
        self.c = self._np1d(self.c)
        print(f"computing p time: {time.time()-st:.4f} seconds")
        
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
            print(f'Gain Keep: {gain_keep}') 
            print(f'Constr Keep: {constr_keep}') 
            
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
        Faster version: vectorized SOC (μ) screening and time-vectorized gain (g1) screening.
        Preserves return values and side effects.
        """        
        # ---------------------- NO-SCREENING FAST PATH ----------------------
        if l1_duals is None:
            self.gap = None
            print(f"[smpc.py]: Gap Radius is {self.gap}")
            st = time.time()

            ones_vec = np.ones((self.N - 1, 1), dtype=float)

            # Gain keeps
            for k in range(self.N_TV):
                for j in range(self.N_modes[k]):
                    self.opti.set_value(self.gain_keep[k][j], ones_vec)

            # Constraint keeps
            M = len(self.mode_map)
            for k in range(self.N_TV):
                for m in range(M):
                    self.opti.set_value(self.constr_keep[k][m], ones_vec)

            vars_kept = 2 * (self.N - 1) * sum(self.N_modes)  # two gain entries per time if kept
            constr_kept = (self.N - 1) * self.N_TV * M
            self.vars_kept = vars_kept
            self.constr_kept = constr_kept

            print('[smpc.py]: Update Gain and Constraint Setting Keep Time: ', time.time() - st, ' s')
            return None, None, None, self.gap

        # ---------------------- WITH SCREENING ----------------------
        # Compute dual dimensions
        self.mu_dim = 2 * self.N * (self.N_TV + 1) + 1
        self.num_ca_duals = int(len(ca_duals) / self.mu_dim)

        # --- recover duals (unchanged pipeline) ---
        st = time.time()
        mu, eta, g1 = self.solve_dual_approximation(self.Q, self.L, self.F, self.C,
                                                    self.p, self.f, self.c,
                                                    ca_duals, l1_duals)
        print('[smpc.py]: Dual Approximation Time: ', time.time() - st, ' s')

        st = time.time()
        self.gap = self._compute_gap_radius(mu, eta, g1)
        print(f"[smpc.py]: Gap Radius is {self.gap:.5f}")
        print('[smpc.py]: Gap Radius Computation Time: ', time.time() - st, ' s')

        # Shapes and helpers
        st = time.time()
        num_t = self.N - 1
        M = len(self.mode_map)
        K = self.N_TV
        TOL = 0.3  # same as _safe_screen default

        # ---------- VECTORIZE: SOC (μ) SCREENING ----------
        # μ comes back flat; reshape to (K, M, T, mu_dim) with t as innermost in your build
        try:
            mu_mat = mu.reshape(K, M, num_t, self.mu_dim)
        except ValueError:
            # Fallback if shape unexpected: treat third dim as T and broadcast K,M=1
            mu_mat = mu.reshape(1, 1, -1, self.mu_dim)
            K, M, num_t = mu_mat.shape[0], mu_mat.shape[1], mu_mat.shape[2]

        # Vectorized norm over the μ blocks
        mu_norm = np.linalg.norm(mu_mat, axis=3)  # (K, M, T)

        # Keep rule: ||μ||_2 + gap >= TOL
        keep_vec = (mu_norm + self.gap >= TOL)

        # Respect predicted-active ternary from ca_duals (same (k,m,t) order)
        ca_duals = ca_duals[0:-1:self.mu_dim]
        ca_act = (np.asarray(ca_duals, dtype=float).reshape(K, M, num_t)) > 0.0
        keep_vec |= ca_act

        # Push back SOC keeps per (k,m) as (T,1) vectors
        constr_kept = int(keep_vec.sum())
        for k in range(K):
            for m in range(M):
                self.opti.set_value(self.constr_keep[k][m],
                                    keep_vec[k, m, :].astype(float).reshape(num_t, 1))
        # ---------- VECTORIZE: GAIN (g1) SCREENING ----------
        # Unflatten for quick "always-keep" mask
        l1_dual_dim = [self.N - 1, self.N_modes, self.N_TV]
        ca_dual_dim = [self.N - 1, len(self.mode_map), self.N_TV]
        l1_duals_list, _ = unflatten_duals(
            np.expand_dims(np.concatenate([l1_duals, ca_duals]), axis=0),
            l1_dual_dim=l1_dual_dim, ca_dual_dim=ca_dual_dim
        )

        # Precompute prefix sums of modes for indexing into g1
        # Layout assumption for L rows:
        #   order by k (TV), then j (local mode), then t, with 2 entries per t
        modes = np.asarray(self.N_modes, dtype=int)
        modes_prefix = np.zeros(self.N_TV, dtype=int)
        if self.N_TV > 1:
            modes_prefix[1:] = np.cumsum(modes[:-1])

        vars_kept = 0
        gap = float(self.gap)

        for k in range(self.N_TV):
            for j in range(self.N_modes[k]):
                # "always keep" from primal-active duals: (0 in …) or (2 in …)
                active_mask = np.array(
                    [(0 in l1_duals_list[k][j][t]) or (2 in l1_duals_list[k][j][t])
                    for t in range(num_t)], dtype=bool
                )

                keep_t = np.zeros(num_t, dtype=float)
                keep_t[active_mask] = 1.0

                # Remaining time steps: apply safety screen on g1 2-vectors
                if not active_mask.all():
                    base = 2 * num_t * (int(modes_prefix[k]) + int(j))
                    idx0 = base + 2 * np.arange(num_t)
                    idx1 = idx0 + 1
                    g1_pairs = np.vstack([g1[idx0], g1[idx1]]).T  # (T, 2)

                    # Vectorized _safe_screen("l1_dual") logic:
                    # (gap < 0.3) AND (||·||∞ + gap < 1) AND (min(|·|) - gap > 1e-3) → DROP
                    cond_drop = (gap < 0.3) & \
                                ((np.max(np.abs(g1_pairs), axis=1) + gap) < 1.0) & \
                                ((np.min(np.abs(g1_pairs), axis=1) - gap) > 1e-3)

                    # keep = ~drop for those not already active-kept
                    upd_mask = ~active_mask
                    keep_t[upd_mask] = (~cond_drop[upd_mask]).astype(float)

                # Count & push (2 gains per time if kept)
                vars_kept += int(2 * np.sum(keep_t))
                self.opti.set_value(self.gain_keep[k][j], keep_t.reshape(num_t, 1))

        self.vars_kept = vars_kept
        self.constr_kept = constr_kept

        print(f"vars:  {vars_kept} out of {g1.shape[0]}, constr: {constr_kept} out of {len(ca_duals)}")
        print('[smpc.py]: Update Gain and Constraint Setting Keep Time: ', time.time() - st, ' s')
        # Keep your original return signature
        return mu, eta, g1, self.gap

       
    def _eta_best_response_fast(
        self, mu, g, *,
        iters: int = 3,            # small; usually enough with warm-start
        tol: float = 8e-4,
        rho: float = 3e-3,         # small proximal ridge on H
        precond_probes: int = 4,   # Hutchinson probes for diag(H)
        precond_decay: float = 0.8,
        use_nesterov: bool = False,
        armijo_beta: float = 0.6,  # backtracking shrink
        armijo_sigma: float = 5e-5,# sufficient decrease
        polish: bool = True,       # tiny active-set NNLS polish
        max_armijo_tries: int = 2, # cap backtracking work
    ):
        """
        Solve: min_{eta >= 0} 0.5*eta^T H eta + b^T eta
        with H = F Q^{-1} F^T (+ rho I),  b = F Q^{-1}(p + C^T mu + L^T(2g-1)) + f.

        Returns:
            eta : (n_eta, 1) nonnegative vector (column)
        """
        import numpy as np

        # --- helpers: force 1-D float64 ---
        _vec1 = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)

        # Shapes
        n_eta = int(self.F.shape[0])

        # --- inputs as 1-D ---
        mu = _vec1(mu)
        g  = _vec1(g)

        # --- build b (strict 1-D) ---
        a = _vec1(self.p) + _vec1(self.C.T @ mu) + _vec1(self.L.T @ (2.0 * g - 1.0))
        w = _vec1(self._solve_Q(a))
        b = _vec1(self.F @ w) + _vec1(self.f)

        # --- H*x = F Q^{-1} F^T x (+ rho x) with fused, bufferized matvec if available ---
        if hasattr(self, "_make_H_mv"):
            H_mv = self._make_H_mv(rho=rho)
        else:
            # fallback (slower, but correct)
            Ft = self.F.T
            def H_mv(x):
                return _vec1(self.F @ self._solve_Q(_vec1(Ft @ _vec1(x)))) + rho * _vec1(x)

        # --- Hutchinson diag preconditioner (EMA) ---
        need_reset = (not hasattr(self, "_diagH")) or (self._diagH is None) \
                    or (self._diagH.shape[0] != n_eta) or (getattr(self, "_diagH_age", 1e9) > 20)

        if need_reset:
            diag_est = np.zeros(n_eta, dtype=np.float64)
            for _ in range(precond_probes):
                z = np.random.randn(n_eta).astype(np.float64)
                Hz = _vec1(H_mv(z))
                diag_est += Hz * z
            diag_est = np.abs(diag_est) / max(1, precond_probes)
            self._diagH = np.maximum(diag_est, 1e-9)
            self._diagH_age = 0
        else:
            z = np.random.randn(n_eta).astype(np.float64)
            Hz = _vec1(H_mv(z))
            refresh = np.maximum(np.abs(Hz * z), 1e-9)
            self._diagH = precond_decay * self._diagH + (1.0 - precond_decay) * refresh
            self._diagH_age += 1

        D = self._diagH
        inv_sqrtD = 1.0 / np.sqrt(D)
        sqrtD     = 1.0 / inv_sqrtD

        # --- Diagonal NNLS warm-start: eta0 = max(0, -b / diag(H)) ---
        eta = np.maximum(0.0, -b / np.maximum(D, 1e-12))

        # map to y-space: eta = D^{-1/2} y  <=>  y = D^{1/2} eta
        y = sqrtD * eta
        y_prev = y.copy()
        t_mom  = 1.0

        # one-shot objective + gradient in y-space (each call uses exactly one H_mv)
        def phi_and_grad_y(yv):
            et  = inv_sqrtD * yv
            Het = _vec1(H_mv(et))
            phi = 0.5 * float(et @ Het) + float(b @ et)
            gy  = inv_sqrtD * (Het + b)  # ∇φ(y) = D^{-1/2}(H eta + b)
            return phi, gy

        # early exit
        phi0, gk0 = phi_and_grad_y(y)
        eta_proj0 = np.maximum(0.0, inv_sqrtD * (y - 1.25 * gk0))
        pg0 = np.minimum(_vec1(H_mv(eta_proj0)) + b, 0.0)
        if np.linalg.norm(pg0, ord=np.inf) <= tol:
            return eta_proj0.reshape(-1, 1)

        tau = 1.25

        for _ in range(iters):
            # optional Nesterov extrapolation
            if use_nesterov:
                t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom))
                beta  = (t_mom - 1.0) / t_new
                y_ex  = y + beta * (y - y_prev)
            else:
                y_ex  = y

            phi_ex, gk = phi_and_grad_y(y_ex)

            tau_k = tau
            tries = 0
            while True:
                # trial + projection to eta >= 0
                y_trial = y_ex - tau_k * gk
                eta_tr  = np.maximum(0.0, inv_sqrtD * y_trial)
                y_proj  = sqrtD * eta_tr

                # one eval for the trial
                phi_tr, _ = phi_and_grad_y(y_proj)

                dy = y_proj - y_ex
                if phi_tr <= (phi_ex - armijo_sigma * float(dy @ dy) / max(tau_k, 1e-12)):
                    y_next = y_proj
                    break

                tau_k *= armijo_beta
                tries += 1
                if tries >= max_armijo_tries or tau_k < 1e-6:
                    y_next = y_proj
                    break

            # momentum housekeeping
            if use_nesterov and np.dot(y_next - y, y - y_prev) > 0.0:
                t_mom = 1.0
                y_prev = y.copy()
            else:
                y_prev = y.copy()
                if use_nesterov:
                    t_mom = t_new

            y = y_next

            # projected-gradient∞ stop (one H_mv)
            eta = inv_sqrtD * y
            pg = np.minimum(_vec1(H_mv(eta)) + b, 0.0)
            if np.linalg.norm(pg, ord=np.inf) <= tol:
                break

        # final eta (ensure nonnegativity numerically)
        eta = np.maximum(0.0, inv_sqrtD * y)

        # tiny active-set NNLS polish
        if polish and hasattr(self, "_polish_eta_active_set"):
            eta = _vec1(self._polish_eta_active_set(eta, b, H_mv, k_top=24, cg_iters=3))
            eta = np.maximum(0.0, eta)

        return eta


    
    def _polish_eta_active_set(self, eta, b, H_mv, *,
                            k_top: int = 48,  # keep it small: 32–64
                            cg_iters: int = 6):
        """
        Small NNLS polish on the most violated coordinates.
        Solves (H_AA eta_A = -b_A) on active/violated set A with CG,
        clamping to eta_A >= 0 each step. Uses H_mv only (no dense H).
        """
        eta = np.asarray(eta, float).ravel()
        g = H_mv(eta) + b               # KKT grad
        viol = np.maximum(-g, 0.0)      # only negative-grad violate nonnegativity

        # Build tiny active set: positive eta OR large violation
        A = np.flatnonzero((eta > 1e-10) | (viol > 1e-3))
        if A.size == 0:
            return eta.reshape(-1, 1)

        # Focus on top-k most violated to cap cost
        A = A[np.argsort(-viol[A])[:k_top]]

        # Subspace operator: gather/scatter around H_mv
        def H_sub_mv(xA):
            x = np.zeros_like(eta)
            x[A] = xA
            y = H_mv(x)
            return y[A]

        bA = b[A]
        x  = eta[A].copy()

        # CG on normal equations: H_AA x = -b_A
        r = -bA - H_sub_mv(x)
        p = r.copy()
        rr = float(np.dot(r, r))
        for _ in range(cg_iters):
            if rr < 1e-12:
                break
            Hp = H_sub_mv(p)
            denom = float(np.dot(p, Hp)) + 1e-12
            alpha = rr / denom
            x_new = x + alpha * p
            x_new = np.maximum(x_new, 0.0)  # project
            r = r - alpha * Hp
            rr_new = float(np.dot(r, r))
            beta = rr_new / (rr + 1e-12)
            p = r + beta * p
            x = x_new
            rr = rr_new

        eta_out = eta.copy()
        eta_out[A] = x
        return eta_out.reshape(-1, 1)
    
    def _np1d(self, v): return np.asarray(v, dtype=float).ravel()
    
    def solve_dual_approximation(self, Q, L, F, C, p, f, c, ca_dual, l1_dual):
        """
        Dual LS solve using LinearOperators with ONE Q^{-1} per matvec.
        Column reduction via RAID-Net on μ only (keep η, g1).
        Neutral g1 warm start (0.5). No right-preconditioning by default.
        Returns (mu, eta, g1).
        """


        st_first = time.time()
        n_mu, n_eta, n_g1 = C.shape[0], F.shape[0], L.shape[0]
        m = n_mu + n_eta + n_g1
        print(f"[Dual Approx] Dimensions: n_mu={n_mu}, n_eta={n_eta}, n_g1={n_g1}, total={m}")

        # keep block sizes for SOC projection later
        self.blocks = [self.mu_dim - 1] * self.num_ca_duals

        # ---- build b with ONE Q^{-1} ----
        ones_ng1 = np.ones((n_g1, 1))
        pmvec = self._np1d(p.reshape(-1, 1) - (L.T @ ones_ng1))
        w_b = self._np1d(self._solve_Q(pmvec))       # one Q-solve
        b = np.concatenate([-c - self._np1d(C @ w_b),
                            -f - self._np1d(F @ w_b),
                            -2.0 * self._np1d(L @ w_b)])

        self.time_least_squares_formulation = time.time() - st_first
        print(f"[Dual Approximation] Least Squares Formulation Time: {self.time_least_squares_formulation:.6f} s")
        st = time.time()
        # ---- A with ONE Q^{-1} per matvec/rmatvec ----
        def A_mv(x):
            x = self._np1d(x)
            x1, x2, x3 = x[:n_mu], x[n_mu:n_mu+n_eta], x[n_mu+n_eta:]
            s = self._np1d(C.T @ x1 + F.T @ x2 + 2.0 * L.T @ x3)
            w = self._np1d(self._solve_Q(s))          # one Q-solve
            return np.concatenate([self._np1d(C @ w), self._np1d(F @ w), self._np1d(2.0 * L @ w)])

        def A_rmv(y):
            y = self._np1d(y)
            y1, y2, y3 = y[:n_mu], y[n_mu:n_mu+n_eta], y[n_mu+n_eta:]
            s = self._np1d(C.T @ y1 + F.T @ y2 + 2.0 * L.T @ y3)
            z = self._np1d(self._solve_Q(s))          # one Q-solve
            out = np.empty(m, dtype=float)
            out[:n_mu] = self._np1d(C @ z)
            out[n_mu:n_mu+n_eta] = self._np1d(F @ z)
            out[n_mu+n_eta:] = self._np1d(2.0 * L @ z)
            return out

        A_op = spla.LinearOperator((m, m), matvec=A_mv, rmatvec=A_rmv, dtype=float)

        # ---- RAID-Net reduction: keep μ as predicted, keep all η, keep all g1 ----
        reduced_ls = bool(self.config.get('reduced_ls', False))
        print(f"[Dual Approx] Reduced LS: {reduced_ls}")
        if reduced_ls:
            keep_mu  = np.asarray(ca_dual, dtype=bool)            # length n_mu
            keep_eta = np.ones(n_eta, dtype=bool)                 # keep all η
            keep_g1  = np.ones(n_g1, dtype=bool)                 # keep all g1 (stability)
            keep = np.concatenate([keep_mu, keep_eta, keep_g1])
            idx = np.flatnonzero(keep)
        else:
            keep = np.ones(m, dtype=bool)
            idx = np.arange(m, dtype=int)

        self.dual_dims = {"n_mu": int(n_mu), "n_eta": int(n_eta), "n_g1": int(n_g1), "m_total": int(m)}
        self.reduced_cols = int(idx.size); self.total_cols = int(m)
        self.keep_mask = keep.copy(); self.keep_idx = idx.copy()
        print(f"[Dual Approx] Kept {self.reduced_cols} / {self.total_cols} dual columns after RAID-Net reduction")

        # Selection
        S = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(m, idx.size))
        def Ar_mv(xr, A_op=A_op, S=S):  return self._np1d(A_op.matvec(self._np1d(S @ xr)))
        def Ar_rmv(y,  A_op=A_op, S=S): return self._np1d(S.T @ self._np1d(A_op.rmatvec(self._np1d(y))))
        Ar = spla.LinearOperator((m, idx.size), matvec=Ar_mv, rmatvec=Ar_rmv, dtype=float)

        # ---- NO right-preconditioning and NO damp by default (stability first) ----
        ls_tol = float(self.config.get('ls_tol', 1e-5))
        ls_max_iter = int(self.config.get('ls_max_iter', 5000))
        damp = float(self.config.get('lsqr_damp', 0.0))  # 0 by default

        # neutral warm-start for g1
        x0 = np.zeros(Ar.shape[1], dtype=float)
        if reduced_ls:
            start_idx = int(np.sum(keep[:n_mu]) + np.sum(keep[n_mu:n_mu+n_eta]))
            g1_len_kept = int(np.sum(keep[n_mu+n_eta:]))
            if g1_len_kept > 0:
                x0[start_idx:start_idx+g1_len_kept] = 0.5
        else:
            x0[n_mu+n_eta:] = 0.5

        print(f"[Dual Approx] Time to build A: {time.time() - st:.6f} s")
        
        st = time.time()
        x_red, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(
            Ar, b, atol=ls_tol, btol=ls_tol, iter_lim=ls_max_iter, x0=x0, damp=damp
        )
        self.time_least_squares_solve = time.time() - st
        print(f"[Dual Approximation] LSQR finished in {self.time_least_squares_solve:.6f} s "
            f"with system {Ar.shape}, iters={itn}, istop={istop}")
        print(f"[Dual Approximation] Residuals: r1norm={r1norm:.3e}, r2norm={r2norm:.3e}, "
            f"||A||≈{anorm:.3e}, cond≈{acond:.3e}, arnorm={arnorm:.3e}, ||x||={xnorm:.3e}")

        # Scatter back
        x_full = np.zeros(m, dtype=float)
        x_full[idx] = x_red

        # If you later reduce g1, remember to set dropped entries to 0.5 (neutral).
        st = time.time()
        # mu = self._proj_soc_dual_stacked_np(x_full[:n_mu], self.blocks).flatten()
        mu, eta, g1 = x_full[:n_mu], x_full[n_mu:n_mu+n_eta], x_full[n_mu+n_eta:]
        return mu, eta, g1

        
    def _compute_gap_radius(self, f_mu, f_nu, f_g): 
        """ gap_radius = ||dual - Proj(dual - grad_d)|| * (1+sigma)/eta 
        where 
        grad_d = [C; F; 2L] Q^{-1} (C^T μ + F^T η + L^T(2g-1) + p) + [-c; f; 0] 
        with eta = 1/largest_eig(Q), sigma = 1/smallest_eig(Q). 
        """ 
        st_first = time.time()
        # ---- inputs as 1-D numpy ---- 
        st = time.time() 
        # inside your call site
        eta_br = self._eta_best_response_fast(
            f_mu, f_g,
            iters=3,          # keep small
            tol=2e-3,
            rho=1e-3,
            use_nesterov=False,
            polish=True,      # <— enable
        )
        f_nu = eta_br
        print(f'[smpc.py]: Best Response Time: {time.time()-st:.6f} s') 
                
        # ---- gradient of dual objective (no explicit stacks, use Q^{-1} apply) ---- 
        st = time.time() 
        ones_fg = np.ones_like(f_g)
        temp = (self.C.T @ f_mu) + self.p + (self.F.T @ f_nu) + (self.L.T @ (2.0*f_g - ones_fg))
        # Solve Q u = temp (uses cached factorization)
        u = self._solve_Q(temp)
        # grad_d blockwise: 
        g1 = (self.C @ u)
        g2 = self.F @ u
        g3 = 2*(self.L @ u)
        # grad_d = np.concatenate([g1 + self.c, g2 + self.f, g3])
        self.gradient_computation_time = time.time() - st 
        print(f'[smpc.py]: Dual gradient build time: {self.gradient_computation_time:.6f} s') 
        st = time.time()
        # ---- one projected step (alpha=1; equivalent to your proj_dual = dual - grad_d) ---- 
        dual = np.concatenate([f_mu, f_nu, f_g]) 
        # proj_dual = dual - grad_d 
        proj_dual = np.concatenate([f_mu - (g1 + self.c), f_nu - (g2 + self.f), f_g - g3])
        
        # ---- projection to dual feasible set ---- 
        n_mu = self.C.shape[0] 
        n_eta = self.F.shape[0] 
        # μ ∈ SOC* (blockwise) 
        # self.blocks should be [self.mu_dim - 1] * self.num_ca_duals (set earlier). 
        # proj_dual[:n_mu] = self._proj_soc_dual_stacked_np(proj_dual[:n_mu], self.blocks).flatten() 
        proj_dual[:n_mu] = self._proj_normal_soc_stacked_vecfirst(proj_dual[:n_mu],-(g1 - self.c),self.blocks,tol=1e-9).ravel()
        # η ≥ 0 
        proj_dual[n_mu:n_mu+n_eta] = np.maximum(proj_dual[n_mu:n_mu+n_eta], 0.0) 
        # 0 ≤ g ≤ 1
        proj_dual[n_mu+n_eta:] = np.clip(proj_dual[n_mu+n_eta:], 0.0, 1.0) 
        print(f'[smpc.py]: Dual projection time: {time.time()-st:.6f} s')
        # ---- gap radius ---- 
        st = time.time() 
        # gap = np.linalg.norm(dual - proj_dual) * (1.0 + self._sigma) / self._eta_inv
        gap = self._norm2_diff_inbuf(dual, proj_dual) * (1.0 + self._sigma) / self._eta_inv
        print(f'[smpc.py]: Gap norm time: {time.time() - st:.6f} s') 
        
        # print(np.linalg.norm((dual - proj_dual)[:n_mu]))
        # print(np.linalg.norm((dual - proj_dual)[n_mu:n_mu+n_eta]))
        # print(np.linalg.norm((dual - proj_dual)[n_mu+n_eta:])) 
        return gap 

    def _split_slack_vecfirst(self, s_stack, blocks): 
        """ s_stack holds concatenated (z_i, y_i) blocks from Cθ+c (or grad_dual slice). 
        Returns (y_list, z_list) aligned with 'blocks'. 
        """ 
        s = np.asarray(s_stack, float).ravel() 
        y_list, z_list = [], [] 
        idx = 0 
        for n in blocks: 
            z_i = s[idx: idx+n]
            y_i = s[idx+n] 
            z_list.append(z_i.copy()) 
            y_list.append(float(y_i)) 
            idx += n + 1 
        return y_list, z_list    

    def _proj_normal_soc_stacked_vecfirst(self, mu_stack, s_stack, blocks, tol=1e-10):
        mu = np.asarray(mu_stack, dtype=np.float64).ravel(order="C")
        s  = np.asarray(s_stack,  dtype=np.float64).ravel(order="C")
        blk = np.asarray(blocks,  dtype=np.int64)
        if _HAS_NUMBA:
            out = _proj_normal_soc_stacked_vecfirst_numba(mu, s, blk, tol)
            return out[:, None]
        # ---- NumPy fallback (no generators) ----
        out = np.empty_like(mu)
        if blk.size == 0:
            return out[:, None]
        # prefix sums for starts
        starts = np.empty(blk.size, dtype=np.int64)
        starts[0] = 0
        for i in range(1, blk.size):
            starts[i] = starts[i-1] + (blk[i-1] + 1)
        for i in range(blk.size):
            n_i = int(blk[i]); s0 = int(starts[i])
            z_i = s[s0:s0+n_i]; y_i = float(s[s0+n_i])
            mu_i = mu[s0:s0+n_i+1]
            nz = float(np.linalg.norm(z_i, 2))
            phi = nz - y_i
            if phi < -tol:
                out[s0:s0+n_i+1] = 0.0
                continue
            if nz < tol and abs(y_i) < tol:
                t = float(mu_i[-1]); x = mu_i[:-1]
                tK, xK = self._proj_soc_np(-t, -x)  # onto K
                out[s0:s0+n_i] = -xK; out[s0+n_i] = -tK
                continue
            u_z = z_i / (nz + 1e-16); u_y = -1.0
            dot = float(np.dot(mu_i[:-1], u_z) + mu_i[-1] * u_y)
            tau = 0.5 * max(0.0, dot)
            out[s0:s0+n_i] = tau * u_z
            out[s0+n_i]    = -tau
        return out[:, None]


    def _ensure_buf(self, n: int):
        """Make sure we have a float64, C-contiguous buffer of length n."""
        if not hasattr(self, "_buf_diff") or self._buf_diff is None or self._buf_diff.size != n:
            self._buf_diff = np.empty(n, dtype=np.float64)

    def _norm2_diff_inbuf(self, a, b):
        """
        Fast ‖a−b‖₂ using a reusable buffer (no allocation of (a-b)).
        Requires 1D float64, C-contiguous; casts if needed.
        """
        a = np.asarray(a, dtype=np.float64).ravel(order="C")
        b = np.asarray(b, dtype=np.float64).ravel(order="C")
        assert a.shape == b.shape
        self._ensure_buf(a.size)
        buf = self._buf_diff
        # buf = a - b  (in-place)
        np.copyto(buf, a)
        np.subtract(buf, b, out=buf)
        # return sqrt(buf·buf) via BLAS-backed dot
        return float(np.sqrt(np.dot(buf, buf)))

    def _norm2_diff_chunked(self, a, b, chunk: int = 1 << 14):
        """
        Chunked version to keep working set small; also uses the same buffer.
        """
        a = np.asarray(a, dtype=np.float64).ravel(order="C")
        b = np.asarray(b, dtype=np.float64).ravel(order="C")
        assert a.shape == b.shape
        self._ensure_buf(min(chunk, a.size))
        buf = self._buf_diff
        ss = 0.0
        n = a.size
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            np.subtract(a[i:j], b[i:j], out=buf[: (j - i)])
            ss += float(np.dot(buf[: (j - i)], buf[: (j - i)]))
        return float(np.sqrt(ss))

    try:
        from numba import njit, objmode
        _HAS_NUMBA = True
    except Exception:
        _HAS_NUMBA = False
    if _HAS_NUMBA:
        # JIT just the chunked accumulator (keeps Python out of the loop)
        @njit(cache=True, fastmath=True)
        def _acc_ss_numba(a, b):
            ss = 0.0
            for i in range(a.size):
                d = a[i] - b[i]
                ss += d * d
            return ss
        def _norm2_diff_numba(self, a, b):
            a = np.asarray(a, np.float64).ravel()
            b = np.asarray(b, np.float64).ravel()
            return float(np.sqrt(_acc_ss_numba(a, b)))

        
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
        # Ray projection: μ̂ = max(0, ⟨μ,u⟩) / ||u||^2 * u, ||u||^2 = 2 
        tau = max(0.0, dot) * 0.5 
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
            tp = -tp_K
            xp = -xp_K 
            out.append(xp) 
            out.append(np.array([tp]))
            idx += n + 1 
        return np.concatenate(out)[:, None]
    
    def _safe_screen(self,dual, gap_radius, dual_type = "ca_dual"):
        keep = 1 
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


'''
Numba-accelerated Functions
'''
if _HAS_NUMBA:
    @njit(cache=True)
    def _proj_soc_K_numba(t, x):  # project (t, x) onto K = { (tau, y): ||y|| <= tau }
        nx = 0.0
        for i in range(x.size):
            nx += x[i] * x[i]
        nx = math.sqrt(nx)
        if nx <= t:
            # already in the cone
            return t, x.copy()
        if nx <= -t:
            # projects to zero
            return 0.0, np.zeros_like(x)
        alpha = 0.5 * (nx + t)
        scale = alpha / max(nx, 1e-12)
        y = np.empty_like(x)
        for i in range(x.size):
            y[i] = scale * x[i]
        return alpha, y

    @njit(cache=True)
    def _proj_normal_soc_stacked_vecfirst_numba(mu, s, blocks, tol):
        """
        Numba version. All inputs 1D np.float64 (mu,s) and 1D np.int64 (blocks).
        Layout per block: (z[0:n], y)
        Returns 1D np.float64 (same layout).
        """
        n_blocks = blocks.size
        # prefix sums for block starts (each block length = n_i + 1)
        starts = np.empty(n_blocks, np.int64)
        if n_blocks > 0:
            starts[0] = 0
        for i in range(1, n_blocks):
            starts[i] = starts[i-1] + (blocks[i-1] + 1)

        out = np.empty_like(mu)

        for i in range(n_blocks):
            n_i = int(blocks[i])
            s0  = int(starts[i])

            # z_i view
            # (Numba supports slicing, but we’ll loop for clarity/speed)
            # compute nz = ||z_i||
            nz = 0.0
            for k in range(n_i):
                v = s[s0 + k]
                nz += v * v
            nz = math.sqrt(nz)
            y_i = float(s[s0 + n_i])
            phi = nz - y_i

            # Interior: normal cone is {0}
            if phi < -tol:
                for k in range(n_i + 1):
                    out[s0 + k] = 0.0
                continue

            # Apex: s ~ 0 ⇒ N_K(0) = -K  (project mu onto -K)
            if nz < tol and abs(y_i) < tol:
                # project (-t, -x) onto K, then negate
                t = float(mu[s0 + n_i])
                # build -x
                xm = np.empty(n_i, np.float64)
                for k in range(n_i):
                    xm[k] = -mu[s0 + k]
                tK, xK = _proj_soc_K_numba(-t, xm)
                # negate back to -K
                for k in range(n_i):
                    out[s0 + k] = -xK[k]
                out[s0 + n_i] = -tK
                continue

            # Boundary (or slight infeasibility): project onto ray u = (z/nz, -1)
            # dot = <mu, u> = mu_z · (z/nz) + mu_y * (-1)
            dot = 0.0
            for k in range(n_i):
                dot += mu[s0 + k] * (s[s0 + k] / (nz + 1e-16))
            dot += mu[s0 + n_i] * (-1.0)
            tau = 0.5 * (dot if dot > 0.0 else 0.0)

            # out_z = tau * z/nz, out_y = tau * (-1)
            inv = 1.0 / (nz + 1e-16)
            for k in range(n_i):
                out[s0 + k] = tau * s[s0 + k] * inv
            out[s0 + n_i] = -tau

        return out