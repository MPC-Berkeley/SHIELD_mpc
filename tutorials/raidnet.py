from typing import Any
import torch as th
from torch import nn
import pdb
import numpy as np
import casadi as ca
import scipy.sparse as sp

class Encoder(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim, dropout, activation=nn.ReLU()):
        super(Encoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(input_dim, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, input_dim)
        self.activation = activation
        self.feedforward = nn.Sequential(
            self.fc1,
            self.activation,
            self.fc2
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output, _      = self.self_attn(x, x, x)
        x                   = self.norm1(x + self.dropout(attn_output))
        feedforward_output  = self.feedforward(x)
        x                   = self.norm2(x + self.dropout(feedforward_output))
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim, dropout, activation=nn.ReLU()):
        super(Decoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(input_dim, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, input_dim)
        self.activation = activation
        self.feedforward = nn.Sequential(
            self.fc1,
            self.activation,
            self.fc2
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, _      = self.self_attn(x, x, x, tgt_mask)
        x                   = self.norm1(x + self.dropout(attn_output))
        cross_attn_output, _= self.cross_attn(x, enc_output, enc_output, src_mask)
        x                   = self.norm2(x + self.dropout(cross_attn_output))
        feedforward_output  = self.feedforward(x)
        x                   = self.norm3(x + self.dropout(feedforward_output))
        return x

class RAID_NET_V2(nn.Module):
  '''
    Recurrent Transformer architecture for predicting the dual variables of a Stochastic MPC problem
  '''
  def __init__(self,
                config, 
                observation_dim, 
                embed_dim, 
                output_dim,
                mpc_horizon,
                num_tv,
                num_layers,
                hidden_size,
                lambda_dim = None,
                eps = 0.8,
                lambda_ubd = 1000,
                pred_mode=["both duals",'tertiary','binary'],
                device='cuda:0',
                inference_mode = False):
        super(RAID_NET_V2 ,self).__init__()

        ### Build the input embedding layer ###
        self.input_embedding = nn.Linear(observation_dim, embed_dim)
        ### Build the Encoder ###
        self.encoder = Encoder(embed_dim, num_heads=config['num_heads'], feedforward_dim=hidden_size, dropout=config['dropout_prob'])

        ### Build the Decoder ###
        self.decoder = Decoder(embed_dim, num_heads=config['num_heads'], feedforward_dim=hidden_size, dropout=config['dropout_prob'])
        
        ### Decoder Output to Dual Class Prediction Projection ###
        if pred_mode[1] == 'binary':
            self.activation_proj = nn.Sigmoid()
            self.projection = nn.Linear(embed_dim, int(output_dim/(mpc_horizon*num_tv))) #Output dim is always divisible by mpc_horizon. projection output dim is the number of scenarios (M^V)
        elif pred_mode[1] == 'tertiary':
            self.activation_proj = nn.LogSoftmax(dim=-1) 
            self.projection = nn.Linear(embed_dim, int(3*output_dim/(mpc_horizon*num_tv))) #Multiplied by 3 for tertiary classification
        else:
            raise NotImplementedError(f"Prediction mode {pred_mode[1]} not implemented.")

        # Define the parameters
        self.N = mpc_horizon 
        self.pred_mode = pred_mode
        self.lambda_dim = output_dim
        self.output_dim=output_dim
        self.lmbd_ubd = lambda_ubd
        self.inference_mode = inference_mode

  def _set_obs_stats(self,mean,cov):
      self.obs_mean = mean
      self.obs_cov = cov
    
  def load_state_dict(self, state_dict: Any, strict: bool = True) -> None:
        super().load_state_dict(state_dict, strict)
        print("RAID-Net weights loaded successfully.")
      
  def forward(self, x: th.Tensor) -> th.Tensor:
        ### Encoder ####
        #Divide x into sequence of temporal inputs
        # encoder_outputs = []
        # for t in range(1,self.N+1):
            # encoder_outputs.append(self.encoder(th.cat((x[:,:,:2],x[:,:,2+3*(t)*2:2+3*(t+1)*2]),dim=-1)))
        
        #Original
        x_embed = self.input_embedding(x)
        encoder_output = self.encoder(x_embed)
        assert encoder_output.shape == x_embed.shape

        ##### Recurrent calls of the decoders #####
        h0    = x_embed
        h     = h0
        outputs = []
        for t in range(self.N):
            # encoder_output = encoder_outputs[t]
            decoder_output = self.decoder(h, encoder_output, src_mask=None, tgt_mask=None)
            h = decoder_output #Recurrent update

            # Projection
            if self.pred_mode[1] == 'tertiary':
                # out = self.activation_proj(self.projection(h).view(h.shape[0], h.shape[1],-1, 2, 3)) #-1 is num_modes. 2 for l1 dual dimension. 3 for tertiary
                out = self.projection(h).view(h.shape[0], h.shape[1],-1, 2, 3) #-1 is num_modes. 3 for tertiary
            elif self.pred_mode[1] == 'binary' and self.pred_mode[0] == 'l1':
                if self.inference_mode:
                    out = self.activation_proj(self.projection(h).view(h.shape[0], h.shape[1], -1, 2))
                else:
                    out = self.projection(h).view(h.shape[0], h.shape[1], -1, 2)
            else:
                if self.inference_mode:
                    out = self.activation_proj(self.projection(h))
                else:
                    out = self.projection(h)
            outputs.append(out)
        if self.pred_mode[1] == 'tertiary':
            return th.stack(outputs, dim=1).permute(0,2,3,1,4,5).reshape(x.shape[0],-1,3) #assume x.shape[0] is batch size
        elif self.pred_mode[1] == 'binary' and self.pred_mode[0] == 'l1':
            return th.stack(outputs, dim=1).permute(0,2,3,1,4).reshape(x.shape[0],-1)
        elif self.pred_mode[1] == 'binary' and self.pred_mode[0] == 'ca':
            return th.stack(outputs, dim=1).permute(0,2,3,1).reshape(x.shape[0],-1)
            # return th.stack(outputs, dim=1)  # Shape: (batch_size, N, V, M^V or [M,2,3]) varies by duals (ca dual and l1 dual respectively)
        else:
            raise NotImplementedError(f"Prediction mode {self.pred_mode[1]} not implemented.")