from typing import Any
import torch as th
from torch import nn
import pdb
import numpy as np
import scipy as sp

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
      for i in range(n_tv):
        tv_input_embed = self.input_embedding(obs[i*(4*(self.N+1) + 2):(i+1)*(4*(self.N+1) + 2)])#i-th TV's features
        Q.append(tv_input_embed)
      Q = th.vstack(Q)

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

      x=self.add_norm(Q+attn)
      x=self.add_norm(x+self.pred(x))

      ##### Recurrent units (Decoders) #####
      h_0 = th.zeros_like(x)
      h=h_0

      duals = []
      for i in range(self.n_tv):
         


      for k in range(self.N):
        attn, _=self.mh_attn_dc(x,x,h)
        attn = self.drop_dec(attn)
        attn=self.add_norm(x+attn)
        h=self.add_norm(attn+self.pred_d(attn)) #shape: (n_batch, n_tv, embed_dim)
        
        dual = self._clip(self.project(th.flatten(h,start_dim=1))) #shape: (n_batch, dual_dim)

        x_o, h_o = self.rnn_d(x.view(batch_size,1,-1), th.stack([th.flatten(h,start_dim=1)]))
        # h = h_o[0,:,:].view(batch_size, n_tv+1, -1)
        # x, h = self.rnn_d(x, h)

        h = h_o[0,:,:].view(batch_size, self.n_tv, -1) #reshape h for next iteration of multi-head attention
        # x = x_o[:,:,:self.embed_dim]
        x = x_o.view(batch_size,self.n_tv,-1)
        duals.append(dual[:,:self.clip_lmbd_dim])
      return th.hstack(duals) if self.pred_mode[0] == 'l1' and self.pred_mode[1] == 'tertiary' else th.hstack(duals).flatten(start_dim=1)
      
