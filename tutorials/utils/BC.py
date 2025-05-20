from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    List,
)
import pdb
import numpy as np
import torch as th
from nuplan.planning.simulation.planner.utils.smpc_utils  import to_tensor_var, weighted_MSEloss, unflatten_duals, flatten
from datetime import datetime
from tutorials.policies import LeastSquares
import casadi as ca
import pickle
import scipy.sparse as sp
import os

class BC():
    def __init__(self,
                 policy: None,
                 device: Union[str, th.device],
                 optimizer: 'Adam',
                 optim_lr: 0.001,
                rng: np.random.Generator,
                demonstrations: None,
                logger: None,
                normalize: False,
                normalize_obs: False,
                config: None,
                ca_dual_dim: None,
                l1_dual_dim: None,
                batch_size: int = 32,
                dagger_mode: bool = False,
                ismlp: bool = False,
                joint_dual_pred: bool = False,):
        self.l1_dual_dim = l1_dual_dim
        self.policy = policy
        self.device = device
        self.logger = logger
        self.ca_dual_dim = ca_dual_dim
        self.normalize = normalize
        self.normalize_obs = normalize_obs
        self.file_path = None
        self.demonstrations = demonstrations
        self.dagger_mode = dagger_mode
        self.ismlp = ismlp
        self.joint_dual_pred = joint_dual_pred

        #Weighted sampling
        weights = th.DoubleTensor(self.demonstrations.dataset_weights)
        w_sampler = th.utils.data.sampler.WeightedRandomSampler(weights, self.demonstrations.max_size)
        self.train_loader = th.utils.data.DataLoader(self.demonstrations, batch_size=batch_size,sampler = w_sampler, pin_memory=True) 

        #Setting up the optimizers for the policies
        if self.joint_dual_pred:
            if optimizer=='Adam':
                    self.optimizer = th.optim.AdamW(self.policy.parameters(),  lr=optim_lr)
            else:
                self.optimizer = th.optim.RMSprop(self.policy.parameters(),  lr=optim_lr)
            self.lr_sched = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 =10, T_mult = 2, eta_min = optim_lr*0.1, last_epoch=-1 )
        else:
            assert isinstance(self.policy,List)
            if optimizer == 'Adam':
                self.optimizer = [th.optim.AdamW(pol.parameters(), lr=optim_lr) for i, pol in enumerate(self.policy)]
            else:
                self.optimizer = [th.optim.RMSprop(pol.parameters(), lr=optim_lr) for i, pol in enumerate(self.policy)]
            self.lr_sched = [th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0 =10, T_mult = 2, eta_min = optim_lr*0.1 if i == 0 else optim_lr*0.1, last_epoch=-1 ) for i, optim in enumerate(self.optimizer)]


        if th.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False 

        self.set_demonstrations(demonstrations)

        self.rng = rng
        self.config = config

        if self.joint_dual_pred:
            self.bce_loss = th.nn.BCEWithLogitsLoss(pos_weight=4*th.ones(self.policy.lambda_dim, device=th.device('cuda')))
            self.w_mse_loss = weighted_MSEloss(self.policy.lmbd_ubd)
            self.ce_loss = th.nn.CrossEntropyLoss(weight=th.tensor([1.,200.,200.], device=th.device('cuda')))
            self.l1_dual_max = self.policy.lmbd_ubd
            self.l1_dual_ind = self.policy.lambda_dim
        else:
            self.bce_loss = th.nn.BCEWithLogitsLoss(pos_weight=4*th.ones(self.policy[1].lambda_dim, device=th.device('cuda'))) #lambda_dim of the policy == output_dim (ca_dual_dim)
            self.bce_loss_l1 = th.nn.BCEWithLogitsLoss(pos_weight=4*th.ones(self.policy[0].lambda_dim, device=th.device('cuda')))
            self.ce_loss = th.nn.CrossEntropyLoss(weight=th.tensor([1.,20.,20.], device=th.device('cuda')))

            self.mse_loss = th.nn.MSELoss()
            # self.w_mse_loss = weighted_MSEloss(self.policy[0].lmbd_ubd)
            self.l1_dual_max = self.policy[0].lmbd_ubd
            self.l1_dual_ind = self.policy[0].lambda_dim

        N = config['N']
        with open(f'/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_N{str(N)}_canon_form.pkl','rb') as f:
            self.canon_prob_fn = pickle.load(f)
        with open(f'/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_N{str(N)}_canon_form_precomputed.pkl','rb') as f:
            self.canon_prob_fn_precomputed = pickle.load(f)
        self.least_squares = LeastSquares(self.canon_prob_fn,self.canon_prob_fn_precomputed,batch_size=batch_size, param_shape=self.demonstrations.smpc_params_dim,device=device)

    def set_demonstrations(self, dataset):
        self.demonstrations = dataset
       
    def update_policy(self, new_policy):
        self.policy = new_policy

    def update_statistics(self, acs, binary = True):
        l1_duals = acs[:,:self.l1_dual_ind] 
        ca_duals = acs[:,self.l1_dual_ind:]
        if not binary: 
            if hasattr(self,'ca_dual_max'):
                self.ca_dual_max= np.max([self.ca_dual_max,np.max(np.max(ca_duals,axis=1))]) #maximum of the ca_duals so far (in batch)
            else:
                self.ca_dual_max= np.max(np.max(ca_duals,axis=1)) #maximum element in the 2D array (scalar)
           
            self.ac_std = th.hstack([th.ones(self.l1_dual_ind,device=th.device('cuda'))*self.l1_dual_max,th.ones(int(acs.shape[1]-self.l1_dual_ind), device=th.device('cuda'))*self.ca_dual_max])
        else:
            # self.ac_std = th.hstack([th.ones(self.l1_dual_ind,device=th.device('cuda'))*self.l1_dual_max,th.ones(int(acs.shape[1]-self.l1_dual_ind), device=th.device('cuda'))])
            acs[:, self.l1_dual_ind:] = th.Tensor(ca_duals > 1e-3).float()

            '''
            binary and tertiary classifications
            '''
            #class 0: |l1 dual|_inf \in int(K_s)
            #class 1: l1_dual = 0
            #class 2: l1_dual = l1_dual_max      
            self.ac_std = th.ones(int(acs.shape[1]), device=th.device('cuda'))
            l1_class = th.where(l1_duals<1e-3,2*th.ones_like(l1_duals),th.zeros_like(l1_duals)) + th.where(l1_duals>self.l1_dual_max - 1e-3,3*th.ones_like(l1_duals),th.zeros_like(l1_duals)) + th.where((1e-3 <= l1_duals) & (l1_duals <= self.l1_dual_max - 1e-3),th.ones_like(l1_duals),th.zeros_like(l1_duals))
            assert not 0 in l1_class
            self.l1_dual_class_exp = to_tensor_var(l1_class - th.ones_like(l1_class))#to follow the class convention: 0,1,2
    
    def train(
        self,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        binary_pred = False,
        model_name: Optional[str] = None,
        pred_mode: Optional[List] = None,
    ):
        training_log = {}
        self.training_loss = []
        self.epochs = [i for i in range(n_epochs)]
        if not os.path.exists(self.config['root_dir']+self.config['model_save_dir']+model_name):
            os.makedirs(self.config['root_dir']+self.config['model_save_dir']+model_name)
        self.file_path = self.config['root_dir']+self.config['model_save_dir']+model_name+'/'+model_name
        for itr in range(n_epochs):
            logs = {}
            #Get Training batch
            ob_batch, ac_batch, opt_duals = next(iter(self.train_loader)) #for weighted sampling

            #Policy Gradient Descent 
            #Compute the loss
            expert_ac_batch = to_tensor_var(ac_batch, use_cuda=self.use_cuda)
            # expert_ac_batch = to_tensor_var([np.fromiter(flatten(unflatten_duals(ac_batch[k,:][None],self.l1_dual_dim, self.ca_dual_dim, data2tar=True)),float) for k in range(n_batches)], use_cuda=self.use_cuda)

            # self.loss = []
            if pred_mode[0] == 'both duals' and self.joint_dual_pred:
                self.optimizer.zero_grad()
                l1_duals = expert_ac_batch[:,:self.policy.lambda_dim]
                l1_class = th.where(l1_duals<1e-3,2*th.ones_like(l1_duals),th.zeros_like(l1_duals)) + th.where(l1_duals>self.l1_dual_max - 1e-3,3*th.ones_like(l1_duals),th.zeros_like(l1_duals)) + th.where((1e-3 <= l1_duals) & (l1_duals <= self.l1_dual_max - 1e-3),th.ones_like(l1_duals),th.zeros_like(l1_duals))
                assert not 0 in l1_class
                self.l1_dual_class_exp = to_tensor_var(l1_class.cpu() - th.ones_like(l1_class).cpu())#to follow the class convention: 0,1,2
                ca_correct = th.sum(th.sigmoid(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda)))[:,-self.policy.lambda_dim:].round()==expert_ac_batch[:,-self.policy.lambda_dim:])
                sum_pred = th.sum(th.sigmoid(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda))[:,-self.policy.lambda_dim:]).round()).item()
                sum_tar  = th.sum(expert_ac_batch[:,-self.policy.lambda_dim:]).item()
                training_ca_acc = ca_correct.item()/(n_batches*(self.policy.output_dim-self.policy.lambda_dim))

                #l1 dual loss
                # self.loss.append(self.w_mse_loss(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda))[:,:self.policy.lambda_dim],expert_ac_batch[:,:self.policy.lambda_dim]))
                # #ca loss
                # self.loss.append(self.bce_loss(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda))[:,-self.policy.lambda_dim:],expert_ac_batch[:,-self.policy.lambda_dim:]))
                # for loss in self.loss:
                #     loss.backward()
                # loss_value = 0
                # for loss in self.loss:
                #     loss_value += loss.detach().cpu().numpy()    
                # self.loss = self.w_mse_loss(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda))[:,:self.policy.lambda_dim],expert_ac_batch[:,:self.policy.lambda_dim]) \
                            # + self.bce_loss(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda))[:,self.policy.lambda_dim:],expert_ac_batch[:,self.policy.lambda_dim:])
                self.loss = self.ce_loss(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda))[:,:self.policy.lambda_dim],expert_ac_batch[:,:self.policy.lambda_dim]) \
                            + self.bce_loss(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda))[:,self.policy.lambda_dim:],self.l1_dual_class_exp)
                self.loss.backward()
                loss_value = self.loss.detach().cpu().numpy()
                '''
                tetiary pred
                '''
                correct_l1 = th.sum((th.argmax(th.exp(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda)))[:,:self.policy.lambda_dim], dim=-1)) == (self.l1_dual_class_exp))
                training_l1_acc = correct_l1.item()/(n_batches*(self.policy.lambda_dim))
                sum_pred = th.sum((th.argmax(th.exp(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda)[:,:self.policy.lambda_dim])), dim=-1)) > 1e-3)
                sum_tar = th.sum(self.l1_dual_class_exp > 1e-3)
                print("Correct L1: ",correct_l1.item(), " out of ", n_batches*(self.policy.lambda_dim), training_l1_acc*100,'% acc')
                print(f'Non-zero class in pred L1: {sum_pred}, Non-zero class in target L1: {sum_tar}')
                print("Correct CA: ",ca_correct.item(), " out of ", n_batches*(self.policy.output_dim-self.policy.lambda_dim),training_ca_acc*100,"% acc")
                print("Ones in pred CA: ", sum_pred, " Ones in target CA:",sum_tar)
                self.optimizer.step()
                if hasattr(self,'lr_sched'):
                    self.lr_sched.step()

                loss_value = self.loss.detach().cpu().numpy()    
                # pdb.set_trace()
                self.training_loss.append(loss_value) #for batches
            else:
                if self.policy[0].pred_mode[1] == 'tertiary':
                    l1_duals = expert_ac_batch[:,:self.policy[0].lambda_dim]
                    l1_class = th.where(l1_duals<1e-3,2*th.ones_like(l1_duals),th.zeros_like(l1_duals)) + th.where(l1_duals>self.l1_dual_max - 1e-3,3*th.ones_like(l1_duals),th.zeros_like(l1_duals)) + th.where((1e-3 <= l1_duals) & (l1_duals <= self.l1_dual_max - 1e-3),th.ones_like(l1_duals),th.zeros_like(l1_duals))
                    assert not 0 in l1_class
                    self.l1_dual_class_exp = to_tensor_var(l1_class.cpu() - th.ones_like(l1_class).cpu())#to follow the class convention: 0,1,2
                elif self.policy[0].pred_mode[1] == 'binary':
                    # l1_duals = expert_ac_batch[:,:self.policy[0].lambda_dim]
                    # l1_class = th.where(l1_duals<1e-3,2*th.ones_like(l1_duals),th.zeros_like(l1_duals)) + th.where(l1_duals>self.l1_dual_max - 1e-3,3*th.ones_like(l1_duals),th.zeros_like(l1_duals)) + th.where((1e-3 <= l1_duals) & (l1_duals <= self.l1_dual_max - 1e-3),th.ones_like(l1_duals),th.zeros_like(l1_duals))
                    # assert not 0 in l1_class
                    # self.l1_dual_class = to_tensor_var((l1_class.cpu() - th.ones_like(l1_class).cpu()) > 1e-3)
                    self.l1_dual_class = expert_ac_batch[:,:self.policy[0].lambda_dim]
                else:
                    raise ValueError('Invalid pred_mode for policy[0]')
                self.loss = []
                for i, optim in enumerate(self.optimizer):
                    optim.zero_grad()
                    if i==1:
                        policy_output = self.policy[1](to_tensor_var(ob_batch, use_cuda=self.use_cuda))
                        pred = th.sigmoid(policy_output).round()
                        ca_correct = th.sum(pred==expert_ac_batch[:,self.policy[0].lambda_dim:])
                        sum_pred = th.sum(pred).item()
                        sum_tar  = th.sum(expert_ac_batch[:,self.policy[0].lambda_dim:]).item()
                        training_ca_acc = ca_correct.item()/(n_batches*(self.policy[1].output_dim))
                        print("Correct CA: ",ca_correct.item(), " out of ", n_batches*(self.policy[1].output_dim),training_ca_acc*100,"% acc")
                        print("Ones in pred CA: ", sum_pred, " Ones in target CA:",sum_tar)
                        self.loss.append(self.bce_loss(policy_output,expert_ac_batch[:,self.policy[0].lambda_dim:]))
                        self.loss[-1] += gap_radius_loss
                        # self.loss = self.bce_loss(self.policy(to_tensor_var(ob_batch, use_cuda=self.use_cuda)),expert_ac_batch[:,-self.policy.lambda_dim:])
                    else:
                        if self.policy[0].pred_mode[1] == 'tertiary':
                            '''
                            tetiary pred
                            '''
                            policy_output = self.policy[0](to_tensor_var(ob_batch, use_cuda=self.use_cuda))
                            correct_l1 = th.sum((th.argmax(th.exp(policy_output), dim=-1)) == (self.l1_dual_class_exp))
                            training_l1_acc = correct_l1.item()/(n_batches*(self.policy[0].output_dim))
                            sum_pred = th.sum((th.argmax(th.exp(policy_output), dim=-1)) > 1e-3)
                            sum_tar = th.sum(self.l1_dual_class_exp > 1e-3)
                            print("Correct L1: ",correct_l1.item(), " out of ", n_batches*(self.policy[0].output_dim), training_l1_acc*100,'% acc')
                            print(f'Non-zero class in pred L1: {sum_pred}, Non-zero class in target L1: {sum_tar}')
                            self.loss.append(self.ce_loss(th.exp(policy_output).movedim(2,1),self.l1_dual_class_exp.long()))    

                        elif self.policy[0].pred_mode[1] == 'binary':
                            '''
                            binary pred
                            '''
                            policy_output = self.policy[0](to_tensor_var(ob_batch, use_cuda=self.use_cuda))
                            correct_l1 = th.sum((th.sigmoid(policy_output).round()) == (self.l1_dual_class).float())
                            training_l1_acc = correct_l1.item()/(n_batches*(self.policy[0].output_dim))
                            sum_pred = th.sum(th.sigmoid(policy_output).round())
                            sum_tar = th.sum(expert_ac_batch[:,:self.policy[0].lambda_dim]>1e-3)
                            print("Correct L1: ",correct_l1.item(), " out of ", n_batches*(self.policy[0].output_dim), training_l1_acc*100,'% acc')
                            print(f'Non-zero class in pred L1: {sum_pred}, Non-zero class in target L1: {sum_tar}')
                            self.loss.append(self.bce_loss_l1(policy_output,expert_ac_batch[:,:self.policy[0].lambda_dim]))
                            th.autograd.set_detect_anomaly(True)
                            mu, nu, g1 = self.least_squares(self.policy[1](to_tensor_var(ob_batch, use_cuda=self.use_cuda)),self.policy[0](to_tensor_var(ob_batch, use_cuda=self.use_cuda)),ob_batch[:,-self.demonstrations.smpc_params_dim[0]:].detach().numpy())
  
                            #compute the gap radius loss
                            ones_fg = th.ones(g1.shape)
                            temp = (-th.bmm(self.least_squares.C.permute(0,2,1), mu.to('cpu'))
                                    + self.least_squares.p
                                    + th.bmm(self.least_squares.F.permute(0,2,1), nu.to('cpu'))
                                    + th.bmm(self.least_squares.L.permute(0,2,1), (2 * g1.to('cpu') - self.policy[0].lmbd_ubd * ones_fg)))
                            stacked_prod = th.cat([-self.least_squares.C_Qinv, self.least_squares.F_Qinv, 2*self.least_squares.L_Qinv],dim=1)      
                            grad_d = stacked_prod @ temp
                            grad_d += th.cat([-self.least_squares.c.unsqueeze(dim=-1), self.least_squares.f.unsqueeze(dim=-1), th.zeros(g1.shape)],dim=1)
                            duals = th.cat((mu,nu,g1),dim=1)
                            proj_duals = duals - grad_d.to('cuda')
                            start_idx = mu.shape[1] + nu.shape[1]

                            # proj_duals[:start_idx] = self.least_squares.relu(proj_duals[:start_idx])
                            # proj_duals[start_idx:] = th.clip(proj_duals[start_idx:],0,self.least_squares.l1_lmbd)
                            gap_radius_loss = self.mse_loss(duals,proj_duals)
                            print('gap radius mean:',th.sqrt(gap_radius_loss))
                            pdb.set_trace()
                            self.loss[-1] += gap_radius_loss
                        else:
                            raise ValueError('Invalid pred_mode for policy[0]')

                    self.loss[-1].backward()
                    self.optimizer[i].step()
                    if hasattr(self,'lr_sched'):
                        self.lr_sched[i].step()

                loss_value = sum([l.detach().cpu().numpy() for l in self.loss])  
                self.training_loss.append(loss_value) #for batches
                
            #Logging
            if self.logger:
                if self.joint_dual_pred:
                    logs.update({'Epochs':itr, 'Training_Loss':self.loss.detach().cpu().numpy(), 'Training_EnvStepsSofar': n_batches * itr})
                else:
                    logs.update({'Epochs':itr, 'L1_training_loss':self.loss[0].detach().cpu().numpy(), 'ca_training_loss':self.loss[1].detach().cpu().numpy(), 'Training_EnvStepsSofar': n_batches * itr, 'training_ca_acc': training_ca_acc, 'training_l1_acc': training_l1_acc, 'gap_radius_loss':gap_radius_loss.value.item()})
                    for key, value in logs.items():
                        print("{} : {}".format(key, value))
                        self.logger.log_scalar(value, key,itr)
                    print("Done logging...\n\n")
                    print('-'.center(80,'-'))
                    self.logger.flush()

            #OPTIONAL (save):
            if itr % self.config['model_save_period']== 0 and itr != 0:
                self.save(self.config['root_dir']+self.config['model_save_dir']+model_name+'/'+model_name,config=self.config,iter=itr)
        if self.joint_dual_pred:
            training_log.update({'training_batch_size': n_batches,'Training_Loss': np.concatenate([self.training_loss]), 'Epochs': self.epochs})
        else:
            #Update only the final losses
            training_log.update({'Epochs':itr, 'L1_training_loss':self.loss[0].detach().cpu().numpy(), 'ca_training_loss':self.loss[1].detach().cpu().numpy(), 'Training_EnvStepsSofar': n_batches * itr, 'training_ca_acc': training_ca_acc, 'training_l1_acc': training_l1_acc})
        return training_log 

    def save(self, model_save_dir, config,iter=None):
        if self.joint_dual_pred:
            th.save({'model_state_dict': self.policy.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),'config': config},
                    self.file_path + '_' + str(iter) + 'epoch'+ '.pt')
        else:
            th.save({'model_state_dict': self.policy[0].state_dict(),
                        'optimizer_state_dict': self.optimizer[0].state_dict(),'config': config},
                    self.file_path + '_L1_'+ str(iter) + 'epoch'+ '.pt')    
            th.save({'model_state_dict': self.policy[1].state_dict(),
                        'optimizer_state_dict': self.optimizer[1].state_dict(),'config': config},
                    self.file_path + '_CA_'+ str(iter) + 'epoch'+ '.pt')   
        
        
