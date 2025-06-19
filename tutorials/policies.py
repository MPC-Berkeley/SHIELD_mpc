from typing import Any
import torch as th
from torch import nn
import pdb
import numpy as np
import casadi as ca
import scipy.sparse as sp

class RAID_NET(nn.Module):
  '''
    Recurrent Transformer architecture for predicting the dual variables of a Stochastic MPC problem
  '''
  def __init__(self,config,input_dim, embed_dim, output_dim, horizon, num_layers, hidden_size,lambda_dim = None, eps = 0.8, lambda_ubd = 1000, pred_mode=["both duals",'tertiary','binary'],device='cuda:0'):
        super(RAID_NET ,self).__init__()
        self.pred_mode = pred_mode
        self.eps = eps
        # self.Q_dim = [6,4]   # num_vs x [state, mode]
        # self.lift=nn.Linear(self.Q_dim[1], embed_dim)
        self.norm = nn.BatchNorm2d(3)  # input, key, value are the features
        self.num_heads = config['num_heads']
        self.mh_attn=nn.MultiheadAttention(embed_dim, num_heads=self.num_heads)
        self.add_norm=nn.LayerNorm(embed_dim)

        self.dim_layers = num_layers
        self.fc_in = nn.Linear(embed_dim, hidden_size)
        self.hidden_layers = []

        self.n_tv = config['num_tvs']
        self.input_embedding = nn.Linear(int(input_dim/self.n_tv), embed_dim)

        self.embed_dim = embed_dim
        self.output_dim=output_dim
        self.device = device
        for i in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size,hidden_size))
            self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.1))

        self.fc_hidden = nn.Sequential(*self.hidden_layers)
        self.fc_out = nn.Linear(hidden_size, embed_dim)
        self.pred = nn.Sequential(self.fc_in,self.fc_hidden,self.fc_out)

        self.dim_encoding = 3 #self.Q_dim[0]

        # self.drop= th.nn.Dropout( p = 0.2)

        # Decoder architecture (attn)
        self.N=horizon # should be SMPC.N-1       
        self.mh_attn_dc = nn.MultiheadAttention(embed_dim, num_heads=self.num_heads)
        
        self.fc_in_d = nn.Linear(embed_dim, hidden_size)
        self.hidden_layers_d = []

        for _ in range(num_layers):
            self.hidden_layers_d.append(nn.Linear(hidden_size,hidden_size))
            self.hidden_layers_d.append(nn.LeakyReLU(negative_slope=0.1))

        self.fc_hidden_d = nn.Sequential(*self.hidden_layers)
        self.fc_out_d = nn.Linear(hidden_size, embed_dim)
        self.pred_d = nn.Sequential(self.fc_in_d,self.fc_hidden_d,self.fc_out_d)

        self.drop_dec = th.nn.Dropout(p = config['dropout_prob'])

        self.project = nn.Linear(embed_dim * self.n_tv, int(output_dim/self.N*3) if self.pred_mode[0]== "l1" and self.pred_mode[1]=='tertiary' else int(output_dim/self.N)) 

        self.rnn_d=nn.GRU(embed_dim*self.n_tv, embed_dim*self.n_tv, batch_first=True)
        self.sigmoid_act = nn.Sigmoid()
        self.log_softmax = th.nn.LogSoftmax(dim=-1) 
        
        #clip the output
        self.lambda_dim=lambda_dim if self.pred_mode[0]=="both duals" else output_dim
       
        self.clip_lmbd_dim=int(self.lambda_dim/self.N)
        self.lmbd_ubd=lambda_ubd
        if self.pred_mode[0] == "ca":
           self.clip_fn = None
        else:
            #l1 dual mode
            if self.pred_mode[1] == 'tertiary':
              self.clip_fn = None
            elif self.pred_mode[1] == 'binary':
              self.clip_fn = None
            else: #contintuous regression
              self.clip_fn = lambda x: (th.tanh(4*x) + 1) /2 #This is needed for cont. regression 

  def _set_obs_stats(self,mean,cov):
      self.obs_mean = mean
      self.obs_cov = cov

  def _get_Q(self, obs, n_tv):
      '''
      constructs Q from input
      Q = [[ego x, ego r], [tv x, tv p],...]: np.ndarray ## -> th.Tensor
      '''
      # if include_traj:
      #    pass
      # # ittc=obs['ttc']
      # Q = obs[:4].reshape(1,-1) #1st row of Q

      # #
      # dist = [1e6] #1 because this will be used as a scale in graph encoder and we wish to not change the scaling for the first row which corresponds to the ego vehicle

      # for i in range(n_tv):
      #   Q = th.vstack((Q,obs[5+4*i:5+4*(i+1)].reshape(1,-1)))
      #   mean = self.obs_mean[5+4*i:5+4*(i+1)][:2].reshape(-1,1)
      #   cov = self.obs_cov[5+4*i:5+4*(i+1),5+4*i:5+4*(i+1)][:2,:2]
      #   dist.append(sp.linalg.norm(sp.linalg.sqrtm(cov)@(obs[5+4*i:5+4*(i+1)][:2]).cpu().numpy().reshape(-1,1) + mean))
      #   #unnormalize obs[5+4*i:5+4*(i+1)][:2].reshape(1,-1)
      
      # #ittc should be of size n_tv + 1
      # #for now, use the distance to the tv from the ego

      # #unnormalize the observation
      # dist = th.tensor(dist)
      # ittc = (1/dist).to(th.float32)
      # return self._graph_encoder(Q, ittc)
      Q = []
      # print(obs.shape)
      for i in range(n_tv):
        tv_input_embed = self.input_embedding(obs[i*(4*(self.N+1) + 2):(i+1)*(4*(self.N+1) + 2)])#i-th TV's features
        Q.append(tv_input_embed)
      Q = th.vstack(Q)
      # print(Q)
      return Q


  def _graph_encoder(self, Q, ittc):
      '''
      compute ttc encoding as 
      Q_new[i]= Q[i]+ ittc[i]
      '''
      Q_new=Q+th.diag(ittc).to(self.device)@th.ones_like(Q, device=self.device)
      return self.lift(Q_new)
  
  def _clip(self, state):
      if self.pred_mode[0] == "both duals":
        if self.pred_mode[1] == 'tertiary':
          lambda_dv, mu_dv = state[:,:self.clip_lmbd_dim], state[:,self.clip_lmbd_dim:]
        else:
          lambda_dv, mu_dv = state[:,:self.clip_lmbd_dim], state[:,self.clip_lmbd_dim:]
        return th.cat((lambda_dv, mu_dv),dim=1) if self.clip_fn is None else th.cat((self.clip_fn(lambda_dv), mu_dv),dim=1)  
      else:
        return state if self.clip_fn is None else self.clip_fn(state)

  def __call__(self, x):
      '''
      x: th.Tensor
      out: th.Tensor
      '''
      ### Encoder ####
      batch_size=x.shape[0]

      Q=th.stack([self._get_Q(x[i],self.n_tv) for i in range(batch_size)])
      # Q_n = self.norm(th.stack([Q,Q,Q], dim=1))
      # Q = Q_n[:,0,:,:]
      attn, _ =self.mh_attn(Q,Q,Q)

      # attn = self.drop(attn)
      x=self.add_norm(Q+attn)
      x=self.add_norm(x+self.pred(x))

      ##### Recurrent units (Decoders) #####
      h_0 = th.zeros_like(x)
      h=h_0
      if self.pred_mode[0]=="both duals":
        l1_duals=[]; ca_duals = []

        for _ in range(self.N):
          attn, _=self.mh_attn_dc(x,x,h)
          attn = self.drop_dec(attn)
          attn=self.add_norm(x+attn)
          h=self.add_norm(attn+self.pred_d(attn)) #shape: (n_batch, n_tv + 1, embed_dim
          duals = self._clip(self.project(th.flatten(h,start_dim=1))) #shape: (n_batch, lambda_dim + mu_dim)
          x_o, h_o = self.rnn_d(x, th.stack([th.flatten(h,start_dim=1)]))
          # h = h_o[0,:,:].view(batch_size, n_tv+1, -1)
          # x, h = self.rnn_d(x, h)
          h = h_o[0,:,:].view(batch_size, self.n_tv, -1)
          x = x_o[:,:,:self.embed_dim]

          l1_duals.append(duals[:,:self.clip_lmbd_dim])
          ca_duals.append(duals[:,self.clip_lmbd_dim:])

        return th.hstack((th.hstack(l1_duals).flatten(start_dim=1),th.hstack(ca_duals).flatten(start_dim=1)))
      else:
        duals = []
        for k in range(self.N):
          attn, _=self.mh_attn_dc(x,x,h)
          attn = self.drop_dec(attn)
          attn=self.add_norm(x+attn)
          h=self.add_norm(attn+self.pred_d(attn)) #shape: (n_batch, n_tv, embed_dim)
          
          dual = self._clip(self.project(th.flatten(h,start_dim=1))) #shape: (n_batch, dual_dim)

          if self.pred_mode[0] == 'l1' and self.pred_mode[1] == 'tertiary':
             temp = dual.view(batch_size,int(self.output_dim/self.N),3)   
             dual = self.log_softmax(temp)  #shape: (N_batch, l1_dim/N, 3)    

          x_o, h_o = self.rnn_d(x.view(batch_size,1,-1), th.stack([th.flatten(h,start_dim=1)]))
          # h = h_o[0,:,:].view(batch_size, n_tv+1, -1)
          # x, h = self.rnn_d(x, h)

          h = h_o[0,:,:].view(batch_size, self.n_tv, -1) #reshape h for next iteration of multi-head attention
          # x = x_o[:,:,:self.embed_dim]
          x = x_o.view(batch_size,self.n_tv,-1)
          duals.append(dual[:,:self.clip_lmbd_dim])
        return th.hstack(duals) if self.pred_mode[0] == 'l1' and self.pred_mode[1] == 'tertiary' else th.hstack(duals).flatten(start_dim=1)
      
class STE_Binarizer(th.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # straight-through
    
class LeastSquares(nn.Module):
  def __init__(self,canon_form_fn, canon_form_fn_precomputed, l1_lmbd=10,batch_size=16,N=15,num_modes=2,N_TV=5,param_shape=None,device='cuda:0'):
      super(LeastSquares ,self).__init__()
      self.l1_lmbd = l1_lmbd
      self.batch_size = batch_size
      self.N = N
      self.N_TV = N_TV
      self.num_modes = num_modes
      self.device = device
      self.canon_form_fn = canon_form_fn
      self.canon_form_fn_precomputed = canon_form_fn_precomputed

      #Import casadi functions
      self.F_fn = self.canon_form_fn_precomputed['F_fn']
      self.L_fn = self.canon_form_fn_precomputed['L_fn']
      self.C_fn_nonzero = self.canon_form_fn_precomputed['C_fn_nonzero']
      self.c_fn = self.canon_form_fn_precomputed['c_fn']
      self.cost_hessian_fn = self.canon_form_fn_precomputed['cost_hessian_fn']

      #map casadi functions for batch processing
      self.F_fn = self.F_fn.map(self.batch_size,'thread',2)
      self.L_fn = self.L_fn.map(self.batch_size,'thread',2)
      self.C_fn_nonzero = self.C_fn_nonzero.map(self.batch_size,'thread',2)
      self.c_fn = self.c_fn.map(self.batch_size,'thread',2)
      self.cost_hessian_fn = self.cost_hessian_fn.map(self.batch_size,'thread',2)
      self.f_l_i_c = self.canon_form_fn['f_l_i_c'].map(self.batch_size,'thread',2)

      self.vars_pol4screening_shape = self.canon_form_fn_precomputed['vars_pol4screening_shape']
      L = self.L_fn(ca.DM(*(self.vars_pol4screening_shape[0],self.batch_size)))
      self.L = th.tensor(L.full(),device='cpu',dtype=th.float32).reshape(L.shape[0],self.batch_size,-1).permute(1,0,2)

      F = self.F_fn(ca.DM(*(self.vars_pol4screening_shape[0],self.batch_size)),ca.DM(*(param_shape[0],self.batch_size)))
      self.F = th.tensor(F.full(),device='cpu',dtype=th.float32).reshape(F.shape[0],self.batch_size,-1).permute(1,0,2)
      
      self.C_shape = self.canon_form_fn_precomputed['C_shape']
      self.row_indices = self.canon_form_fn_precomputed['row_indices']
      self.col_indices = self.canon_form_fn_precomputed['col_indices']

      self.C_Qinv = []
      self.F_Qinv = []
      self.L_Qinv = []

      self.relu = nn.ReLU()

  def solve_dual_approximation(self, S1, S2, rho1, rho2, C, F, L, Q, Q_inv, c, f, p):

      C_Qinv = th.bmm(C, Q_inv)
      F_Qinv = th.bmm(F, Q_inv)
      L_Qinv = th.bmm(L, Q_inv)

      self.C_Qinv.append(C_Qinv)
      self.F_Qinv.append(F_Qinv)
      self.L_Qinv.append(L_Qinv)

      A11 = th.bmm(C_Qinv, C.permute(0,2,1)) + 2 * rho1 * S1
      A12 = -th.bmm(C_Qinv, F.permute(0,2,1))
      A13 = -2 * th.bmm(C_Qinv, L.permute(0,2,1))
      A21 = -th.bmm(F_Qinv, C.permute(0,2,1))
      A22 = th.bmm(F_Qinv, F.permute(0,2,1))
      A23 = 2 * th.bmm(F_Qinv, L.permute(0,2,1))
      A31 = -2 * th.bmm(L_Qinv, C.permute(0,2,1))
      A32 = 2 * th.bmm(L_Qinv, F.permute(0,2,1))
      A33 = 4 * th.bmm(L_Qinv, L.permute(0,2,1)) + 2 * rho2 * S2
      A = th.cat([
          th.cat([A11, A12, A13], dim=2),
          th.cat([A21, A22, A23], dim=2),
          th.cat([A31, A32, A33], dim=2)
      ], dim=1)
      n_g1 = L.shape[1]
      ones_ng1 = th.ones((C.shape[0],n_g1, 1), device='cpu') 
      pmvec = (p - self.l1_lmbd * th.bmm(L.permute(0,2,1),ones_ng1))

      b = th.cat([
          c.unsqueeze(-1) + th.bmm(C_Qinv, pmvec),
          -f.unsqueeze(-1) - th.bmm(F_Qinv, pmvec),
          -2 * th.bmm(L_Qinv, pmvec) + self.l1_lmbd * rho2 * th.bmm(S2, ones_ng1)
      ], dim=1)
      
      #Regularize for ill-conditioned A
      eps = 1e-2  # You can try 1e-3 or 1e-2 if needed
      I = th.eye(A.shape[1], device=A.device).unsqueeze(0).expand(A.shape[0], -1, -1)
      # A_reg = A + eps * I
      A_reg = A
      # b = b
      # Q, R = th.linalg.qr(A_reg,mode='reduced')
      
      # Qt_b = th.bmm(Q.permute(0,2,1),b)
      # pdb.set_trace()
      # x = th.linalg.solve(R[0].to(self.device),Qt_b[0].to(self.device))
      # del Q, R
      # A_reg = A
      AtA = th.bmm(A_reg.permute(0,2,1),A_reg)
      aTa = (AtA + eps*I).to(self.device)
      Atb = th.bmm(A_reg.permute(0,2,1),b).to(self.device)
      try:
        x = th.linalg.solve(aTa[0], Atb[0])
      except:
        pdb.set_trace()
      # aTa.to('cpu')
      # Atb.to('cpu')
      # pdb.set_trace()
      
      # x = th.linalg.lstsq(A_reg.to(self.device), b.to(self.device)).solution
      # pdb.set_trace()
      # x = th.linalg.lstsq(A,b).solution
      n_mu = C.shape[1]
      n_eta = F.shape[1]
      x = x.unsqueeze(0)
      # pdb.set_trace()
      mu = self.relu(x[:, :n_mu])
      eta = self.relu(x[:, n_mu:n_mu + n_eta])
      g1 = th.clip(x[:, n_mu + n_eta:], 0, self.l1_lmbd)
      # mu = x[:, :n_mu]
      # eta = x[:, n_mu:n_mu + n_eta]
      # g1 = x[:, n_mu + n_eta:]
      # A_reg.to('cpu')
      # b.to('cpu')
      # Q.to('cpu')
      # R.to('cpu')
      # Qt_b.to('cpu')
      del A_reg, aTa, AtA, Atb,b, C_Qinv, F_Qinv, L_Qinv
      th.cuda.empty_cache()
      return mu, eta, g1
  
  def batched_kron(self,A, B):
    """
    Batched Kronecker product: returns kron(A[i], B) for i in batch.
    A: (B, m, m)
    B: (n, n)
    Returns: (B, m*n, m*n)
    """
    B = B.to(A.device)
    B_batch = B.view(1, 1, *B.shape)  # (1, 1, n, n)
    A_exp = A.unsqueeze(-1).unsqueeze(-1)  # (B, m, m, 1, 1)
    kron_prod = A_exp * B_batch  # (B, m, m, n, n)
    kron_prod = kron_prod.permute(0, 1, 3, 2, 4).reshape(A.shape[0], A.shape[1]*B.shape[0], A.shape[2]*B.shape[1])
    return kron_prod
  
  def reduce_Ab(self, A, b, mu_tilde: np.ndarray, g1_tilde: np.ndarray,eta_dim: int):
      """
      Reduce the matrix A using the mu and g1 estimate from the RAID-Net.
      A: (B, m, n)
      mu_tilde: (B, m)
      g1_tilde: (B, n)
      Returns: reduced A
      """
      # Create a boolean mask for the entire dual_dim
      B = mu_tilde.shape[0]
      # active_idx = mu_tilde # (B, mu_dim*ca_dual_dim from RAID-Net)
      S_mu = 1

      #eta
      # active_idx = np.concatenate((active_idx, np.ones((B,eta_dim))),dim=1)
      S_nu = th.eye(eta_dim)

      #g1
      # active_idx = np.concatenate((active_idx, 1),dim=1)

      #TODO instead of indexing, we need to use some kind of mapping
      A_red = A[:,active_idx]
      b_red = b[:,active_idx]

      return A_red, b_red, active_idx
  
  def __call__(self,mu_logit,g1_logit,params:np.ndarray):
      '''
      mu_logit: th.Tensor is a batch of mu_logit predictions from the RAID-Net
      g1_logit: th.Tensor is a batch of g1_logit predictions from the RAID-Net  
      params: np.ndarray is the parameters of the smpc problem in ndarray for computing the canonical form matrices
      '''
      self.C_Qinv = []
      self.F_Qinv = []
      self.L_Qinv = []

      #soft relaxation for differentiable binarization
      temperature = 0.1
      mu_tilde = STE_Binarizer.apply(mu_logit)  
      g1_tilde = STE_Binarizer.apply(g1_logit)

      # mu_dim = self.N * (self.num_modes**self.N_TV) * self.N_TV
      mu_dim = 2*self.N * (self.N_TV+1) + 1
      

      I_mu = th.eye(mu_dim, device='cpu')  # (181, 181)
      diag_mu = th.diag_embed(mu_tilde.cpu())  # (batch, 70, 70)

      S1 = self.batched_kron(diag_mu,I_mu)
      # S1 = th.kron(th.diag(mu_tilde).to('cpu'),th.eye(mu_dim,device='cpu'))
      # Now create S2, the selection matrix for batched g1 duals

      diag_g1 = th.diag_embed(1-g1_tilde.cpu())  # (batch, 2, 2)
      # S2 = self.batched_kron(diag_g1,I_g1).to_sparse()
      S2 = diag_g1

      #In online mode, vars_epi is not defined. So, set it to zero
      f = ca.vertcat(*self.f_l_i_c(ca.DM(*(self.vars_pol4screening_shape[0],self.batch_size)),params.T))
      self.f = th.tensor(f.full(),device='cpu',dtype=th.float32).T #(batch_size x 60)
      
      C = self.C_fn_nonzero(params.T).T
      B = self.batch_size
      m, n = self.C_shape
      C_tensor_np = np.zeros((B, m, n), dtype=np.float32)

      for b in range(B):
          C_ = sp.csr_matrix((np.array(C[b,:]).reshape(-1), (self.row_indices, self.col_indices)), shape=self.C_shape).toarray()
          C_tensor_np[b, :] = C_

      # Convert to torch if needed
      self.C = th.from_numpy(C_tensor_np).to('cpu')

      #construct sparse C matrix
      # C = sp.csr_matrix((np.array(C).reshape(-1), (self.row_indices, self.col_indices)), shape=self.C_shape)
      self.c = th.tensor(self.c_fn(params.T).full(),device='cpu',dtype=th.float32).T
      #Here, ca.hessian outputs hessian, J_grad. 
      #J_grad = Q*theta + p. Thus, if we evaluate J_grad with theta = 0, we get p. Note that hessian is not dependent on theta
      Q, p = self.cost_hessian_fn(ca.DM.zeros(*(self.vars_pol4screening_shape[0],self.batch_size)),params.T)
      self.Q = th.tensor(Q.full(),device='cpu',dtype=th.float32).reshape(Q.shape[0],self.batch_size,-1).permute(1,0,2)
      self.p = th.tensor(p.full(),device='cpu',dtype=th.float32).reshape(p.shape[0],self.batch_size,-1).permute(1,0,2)

      self.Q_inv = th.linalg.inv(self.Q)
      
      self.n_mu = self.C.shape[1]
      self.n_eta = self.F.shape[1]
      self.n_g1 = self.L.shape[1]

      mu_all, g1_all, eta_all = [], [], []
      subbatch_size = 1
      #solve by subbatch for GPU memory``
      for i in range(0,self.batch_size,subbatch_size):
        idx_end = min(i+subbatch_size, self.batch_size)
        mu, eta, g1 =  self.solve_dual_approximation(S1[i:idx_end], S2[i:idx_end], rho1, rho2, self.C[i:idx_end], self.F[i:idx_end], self.L[i:idx_end], self.Q[i:idx_end], self.Q_inv[i:idx_end], self.c[i:idx_end], self.f[i:idx_end], self.p[i:idx_end])
        mu_all.append(mu)
        eta_all.append(eta)
        g1_all.append(g1)

      self.C_Qinv = th.cat(self.C_Qinv,dim=0)
      self.F_Qinv = th.cat(self.F_Qinv,dim=0)
      self.L_Qinv = th.cat(self.L_Qinv,dim=0)
      return th.cat(mu_all,dim=0), th.cat(eta_all,dim=0), th.cat(g1_all,dim=0)

class LeastSquaresReduced(nn.Module):
  def __init__(self,canon_form_fn, canon_form_fn_precomputed, l1_lmbd=10,batch_size=16,N=15,N_TV=5,param_shape=None,device='cuda:0'):
      super(LeastSquaresReduced ,self).__init__()
      self.l1_lmbd = l1_lmbd
      self.batch_size = batch_size
      self.N = N
      self.N_TV = N_TV
      self.device = device
      self.canon_form_fn = canon_form_fn
      self.canon_form_fn_precomputed = canon_form_fn_precomputed

      #Import casadi functions
      self.F_fn = self.canon_form_fn_precomputed['F_fn']
      self.L_fn = self.canon_form_fn_precomputed['L_fn']
      self.C_fn_nonzero = self.canon_form_fn_precomputed['C_fn_nonzero']
      self.c_fn = self.canon_form_fn_precomputed['c_fn']
      self.cost_hessian_fn = self.canon_form_fn_precomputed['cost_hessian_fn']

      #map casadi functions for batch processing
      self.F_fn = self.F_fn.map(self.batch_size,'thread',2)
      self.L_fn = self.L_fn.map(self.batch_size,'thread',2)
      self.C_fn_nonzero = self.C_fn_nonzero.map(self.batch_size,'thread',2)
      self.c_fn = self.c_fn.map(self.batch_size,'thread',2)
      self.cost_hessian_fn = self.cost_hessian_fn.map(self.batch_size,'thread',2)
      self.f_l_i_c = self.canon_form_fn['f_l_i_c'].map(self.batch_size,'thread',2)

      self.vars_pol4screening_shape = self.canon_form_fn_precomputed['vars_pol4screening_shape']
      L = self.L_fn(ca.DM(*(self.vars_pol4screening_shape[0],self.batch_size)))
      self.L = th.tensor(L.full(),device='cpu',dtype=th.float32).reshape(L.shape[0],self.batch_size,-1).permute(1,0,2)

      F = self.F_fn(ca.DM(*(self.vars_pol4screening_shape[0],self.batch_size)),ca.DM(*(param_shape[0],self.batch_size)))
      self.F = th.tensor(F.full(),device='cpu',dtype=th.float32).reshape(F.shape[0],self.batch_size,-1).permute(1,0,2)
      
      self.C_shape = self.canon_form_fn_precomputed['C_shape']
      self.row_indices = self.canon_form_fn_precomputed['row_indices']
      self.col_indices = self.canon_form_fn_precomputed['col_indices']

      self.C_Qinv = []
      self.F_Qinv = []
      self.L_Qinv = []

      self.relu = nn.ReLU()

  def solve_dual_approximation(self, S1, S2, rho1, rho2, C, F, L, Q, Q_inv, c, f, p):

      C_Qinv = th.bmm(C, Q_inv)
      F_Qinv = th.bmm(F, Q_inv)
      L_Qinv = th.bmm(L, Q_inv)

      self.C_Qinv.append(C_Qinv)
      self.F_Qinv.append(F_Qinv)
      self.L_Qinv.append(L_Qinv)

      A11 = th.bmm(C_Qinv, C.permute(0,2,1)) + 2 * rho1 * S1
      A12 = -th.bmm(C_Qinv, F.permute(0,2,1))
      A13 = -2 * th.bmm(C_Qinv, L.permute(0,2,1))
      A21 = -th.bmm(F_Qinv, C.permute(0,2,1))
      A22 = th.bmm(F_Qinv, F.permute(0,2,1))
      A23 = 2 * th.bmm(F_Qinv, L.permute(0,2,1))
      A31 = -2 * th.bmm(L_Qinv, C.permute(0,2,1))
      A32 = 2 * th.bmm(L_Qinv, F.permute(0,2,1))
      A33 = 4 * th.bmm(L_Qinv, L.permute(0,2,1)) + 2 * rho2 * S2
      A = th.cat([
          th.cat([A11, A12, A13], dim=2),
          th.cat([A21, A22, A23], dim=2),
          th.cat([A31, A32, A33], dim=2)
      ], dim=1)
      n_g1 = L.shape[1]
      ones_ng1 = th.ones((C.shape[0],n_g1, 1), device='cpu') 
      pmvec = (p - self.l1_lmbd * th.bmm(L.permute(0,2,1),ones_ng1))

      b = th.cat([
          c.unsqueeze(-1) + th.bmm(C_Qinv, pmvec),
          -f.unsqueeze(-1) - th.bmm(F_Qinv, pmvec),
          -2 * th.bmm(L_Qinv, pmvec) + self.l1_lmbd * rho2 * th.bmm(S2, ones_ng1)
      ], dim=1)
      
      #Regularize for ill-conditioned A
      eps = 1e-2  # You can try 1e-3 or 1e-2 if needed
      I = th.eye(A.shape[1], device=A.device).unsqueeze(0).expand(A.shape[0], -1, -1)
      A_reg = A

      AtA = th.bmm(A_reg.permute(0,2,1),A_reg)
      aTa = (AtA + eps*I).to(self.device)
      Atb = th.bmm(A_reg.permute(0,2,1),b).to(self.device)
      try:
        x = th.linalg.solve(aTa[0], Atb[0])
      except:
        pdb.set_trace()
      # aTa.to('cpu')
      # Atb.to('cpu')
      # pdb.set_trace()
      
      # x = th.linalg.lstsq(A_reg.to(self.device), b.to(self.device)).solution
      # pdb.set_trace()
      # x = th.linalg.lstsq(A,b).solution
      n_mu = C.shape[1]
      n_eta = F.shape[1]
      x = x.unsqueeze(0)
      # pdb.set_trace()
      mu = self.relu(x[:, :n_mu])
      eta = self.relu(x[:, n_mu:n_mu + n_eta])
      g1 = th.clip(x[:, n_mu + n_eta:], 0, self.l1_lmbd)
      # mu = x[:, :n_mu]
      # eta = x[:, n_mu:n_mu + n_eta]
      # g1 = x[:, n_mu + n_eta:]
      # A_reg.to('cpu')
      # b.to('cpu')
      # Q.to('cpu')
      # R.to('cpu')
      # Qt_b.to('cpu')
      del A_reg, aTa, AtA, Atb,b, C_Qinv, F_Qinv, L_Qinv
      th.cuda.empty_cache()
      return mu, eta, g1
  
  def batched_kron(self,A, B):
    """
    Batched Kronecker product: returns kron(A[i], B) for i in batch.
    A: (B, m, m)
    B: (n, n)
    Returns: (B, m*n, m*n)
    """
    B = B.to(A.device)
    B_batch = B.view(1, 1, *B.shape)  # (1, 1, n, n)
    A_exp = A.unsqueeze(-1).unsqueeze(-1)  # (B, m, m, 1, 1)
    kron_prod = A_exp * B_batch  # (B, m, m, n, n)
    kron_prod = kron_prod.permute(0, 1, 3, 2, 4).reshape(A.shape[0], A.shape[1]*B.shape[0], A.shape[2]*B.shape[1])
    return kron_prod
  
  def __call__(self,mu_logit,g1_logit,params:np.ndarray):
      '''
      mu_tilde: th.Tensor is a batch of mu_tilde predictions from the RAID-Net
      g1_tilde: th.Tensor is a batch of g1_tilde predictions from the RAID-Net  
      params: np.ndarray is the parameters of the smpc problem in ndarray for computing the canonical form matrices
      '''
      self.C_Qinv = []
      self.F_Qinv = []
      self.L_Qinv = []

      #soft relaxation for differentiable binarization
      mu_tilde = STE_Binarizer.apply(mu_logit)  
      g1_tilde = STE_Binarizer.apply(g1_logit)

      mu_dim = 2*self.N * (self.N_TV+1) + 1
      
      I_mu = th.eye(mu_dim, device='cpu')  # (181, 181)
      diag_mu = th.diag_embed(mu_tilde.cpu())  # (batch, 70, 70)
      S1 = self.batched_kron(diag_mu,I_mu)
      # S1 = th.kron(th.diag(mu_tilde).to('cpu'),th.eye(mu_dim,device='cpu'))
      # Now create S2, the selection matrix for batched g1 duals

      I_g1 = th.eye(2, device='cpu')  # (2, 2)
      diag_g1 = th.diag_embed(1-g1_tilde.cpu())  # (batch, 2, 2)
      # S2 = self.batched_kron(diag_g1,I_g1).to_sparse()
      S2 = diag_g1
      # S2 = th.kron(th.diag(1-g1_tilde).to('cpu'),th.eye(2,device='cpu'))
      rho1, rho2 = 0.1, 0.1

      #In online mode, vars_epi is not defined. So, set it to zero
      f = ca.vertcat(*self.f_l_i_c(ca.DM(*(self.vars_pol4screening_shape[0],self.batch_size)),params.T))
      self.f = th.tensor(f.full(),device='cpu',dtype=th.float32).T #(batch_size x 60)
      
      # self.C = self.C_fn(self.opti.value(self.params))
      C = self.C_fn_nonzero(params.T).T
      B = self.batch_size
      m, n = self.C_shape
      C_tensor_np = np.zeros((B, m, n), dtype=np.float32)

      for b in range(B):
          C_ = sp.csr_matrix((np.array(C[b,:]).reshape(-1), (self.row_indices, self.col_indices)), shape=self.C_shape).toarray()
          C_tensor_np[b, :] = C_

      # Convert to torch if needed
      self.C = th.from_numpy(C_tensor_np).to('cpu')

      #construct sparse C matrix
      # C = sp.csr_matrix((np.array(C).reshape(-1), (self.row_indices, self.col_indices)), shape=self.C_shape)
      self.c = th.tensor(self.c_fn(params.T).full(),device='cpu',dtype=th.float32).T
      #Here, ca.hessian outputs hessian, J_grad. 
      #J_grad = Q*theta + p. Thus, if we evaluate J_grad with theta = 0, we get p. Note that hessian is not dependent on theta
      Q, p = self.cost_hessian_fn(ca.DM.zeros(*(self.vars_pol4screening_shape[0],self.batch_size)),params.T)
      self.Q = th.tensor(Q.full(),device='cpu',dtype=th.float32).reshape(Q.shape[0],self.batch_size,-1).permute(1,0,2)
      self.p = th.tensor(p.full(),device='cpu',dtype=th.float32).reshape(p.shape[0],self.batch_size,-1).permute(1,0,2)

      self.Q_inv = th.linalg.inv(self.Q)
      
      self.n_mu = self.C.shape[1]
      self.n_eta = self.F.shape[1]
      self.n_g1 = self.L.shape[1]

      mu_all, g1_all, eta_all = [], [], []
      subbatch_size = 1
      #solve by subbatch for GPU memory``
      for i in range(0,self.batch_size,subbatch_size):
        idx_end = min(i+subbatch_size, self.batch_size)
        mu, eta, g1 =  self.solve_dual_approximation(S1[i:idx_end], S2[i:idx_end], rho1, rho2, self.C[i:idx_end], self.F[i:idx_end], self.L[i:idx_end], self.Q[i:idx_end], self.Q_inv[i:idx_end], self.c[i:idx_end], self.f[i:idx_end], self.p[i:idx_end])
        mu_all.append(mu)
        eta_all.append(eta)
        g1_all.append(g1)

      self.C_Qinv = th.cat(self.C_Qinv,dim=0)
      self.F_Qinv = th.cat(self.F_Qinv,dim=0)
      self.L_Qinv = th.cat(self.L_Qinv,dim=0)
      return th.cat(mu_all,dim=0), th.cat(eta_all,dim=0), th.cat(g1_all,dim=0)