from nuplan.planning.simulation.planner.utils.smpc_utils import *
from collections import Counter
import torch as th
import scipy.linalg as la
import os

NUPLAN_ROOT_DIR = os.environ['NUPLAN_ROOT_DIR']

class ReplayBuffer(th.utils.data.Dataset):

    def __init__(self, max_size=1000000,training_dataset=True):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.opt_duals = None
        self.smpc_params_dim = None
        self.rews = None
        self.next_obs = None
        self.terminals = None
        self.dual_classes = None
        self.num_classes=4
        self.class_weights = np.zeros(self.num_classes)
        self.training_dataset = training_dataset

    def __len__(self):
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0
        
    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx=idx.tolist()

        return self.obs[idx], self.acs[idx], self.opt_duals[idx]
    
    def normalize(self,l1_dim,l1_lmbd,feature_mean=None,feature_cov=None,target_mean=None,target_cov=None,l1_pred_mode='binary',policy_type='RAIDNET',policy=None):
        self.n = self.obs.shape[1]
        self.d = self.acs.shape[1]

        if feature_mean is None:
            self.feature_mean = np.mean(self.obs, axis=0)
        else:
            self.feature_mean = feature_mean
        if feature_cov is None:
            self.feature_cov = np.cov(self.obs, rowvar=False)
        else:
            self.feature_cov = feature_cov

        if target_mean is None:
            self.target_mean = np.mean(self.acs, axis=0)
        else:
            self.target_mean = target_mean
        if target_cov is None:
            self.target_cov = np.cov(self.acs, rowvar=False)
        else:
            self.target_cov = target_cov

        # Regularize the covariance matrices
        reg_value = 4e-5
        self.feature_cov += reg_value * np.eye(self.feature_cov.shape[0])
        self.target_cov += reg_value * np.eye(self.target_cov.shape[0])
        self.obs = np.real(la.solve(np.real(la.sqrtm(self.feature_cov)), (self.obs - self.feature_mean).T, assume_a='pos').T)
        #For RAIDNET self.acs is categorical variable of shape l1_dim + ca_dim
        #ca duals
        self.acs[:,l1_dim:] = 1.*(self.acs[:,l1_dim:] > 1e-3)
        #l1 duals
        # self.acs[:,:l1_dim] = 1.0*np.logical_or((1-(self.acs[:,:l1_dim]<(l1_lmbd-1e-3)*np.ones_like(self.acs[:,:l1_dim]))), 
        #               (1-(self.acs[:,:l1_dim]>1e-3*np.ones_like(self.acs[:,:l1_dim]))))
        if l1_pred_mode == 'binary':
            l1_duals = self.acs[:, :l1_dim]
            eps = 1e-3

            # --- classic tertiary classes in {0,1,2} ---
            # 0: ~0, 1: (eps, lmbd-eps], 2: ~lmbd
            l1_lmbd = 100
            lmbd = float(l1_lmbd)  # prefer the function arg
            l1_class_tern = (l1_duals > eps).astype(np.int32) \
                            + (l1_duals > (lmbd - eps)).astype(np.int32)

            # --- map to binary: 2->0, 1->1, 0->0 ---
            l1_bin = (l1_class_tern == 1).astype(np.float32)  # or np.int32 if you prefer ints

            # store back as the L1 targets
            self.acs[:, :l1_dim] = l1_bin

        #Save feature mean and cov for unnormalization in the policy
        if self.training_dataset:
            np.savez(NUPLAN_ROOT_DIR+'/nuplan/nn_models/'+policy_type+'_'+'nuplan_expert_data_N14_wayformer_affine_training_stats.npz', feature_mean=self.feature_mean, feature_cov=self.feature_cov, feature_cov_inv =la.inv(self.feature_cov), target_mean=self.target_mean, target_cov=self.target_cov)
            print('Normalization done! Feature mean and cov saved to ../nuplan-devkit/nuplan/nn_models/'+policy_type+'_'+'nuplan_expert_data_N14_wayformer_affine_training_stats.npz')
    
    def normalize4evaluation(self, l1_dim, feature_mean=None, feature_cov=None, target_mean=None, target_cov=None,l1_pred_mode='tertiary'):
        assert feature_mean is not None, 'Feature mean must be provided for evaluation normalization!'
        self.acs[:,l1_dim:] = 1.*(self.acs[:,l1_dim:] > 1e-3)
        self.obs = np.real(la.solve(np.real(la.sqrtm(feature_cov)), (self.obs - feature_mean).T, assume_a='pos').T)
        if l1_pred_mode == 'binary':
            l1_duals = self.acs[:, :l1_dim]
            eps = 1e-3

            # --- classic tertiary classes in {0,1,2} ---
            # 0: ~0, 1: (eps, lmbd-eps], 2: ~lmbd
            l1_lmbd = 100
            lmbd = float(l1_lmbd)  # prefer the function arg
            l1_class_tern = (l1_duals > eps).astype(np.int32) \
                            + (l1_duals > (lmbd - eps)).astype(np.int32)

            # --- map to binary: 2->0, 1->1, 0->0 ---
            l1_bin = (l1_class_tern == 1).astype(np.float32)  # or np.int32 if you prefer ints

            # store back as the L1 targets
            self.acs[:, :l1_dim] = l1_bin

    def set_weights(self):
        
        if self.obs is None:
            raise(ValueError('No data for weight calculation!'))
            
        self.class_weights = np.zeros(self.num_classes)

        class_count = Counter(self.dual_classes)
        for i in range(self.num_classes):
            if i in class_count:
                # self.class_weights[i] = class_count[i]*(1-0.9*int(i==1 or i==2 or i ==3))
                self.class_weights[i] = 1/class_count[i]
        # total =  np.sum(self.class_weights)

        # self.class_weights = (total - self.class_weights)/total
        self.dataset_weights = np.zeros(self.max_size)

        # for i in range(self.max_size):
            # self.dataset_weights[i] = self.class_weights[self.dual_classes[i]]
        self.dataset_weights = self.class_weights[self.dual_classes]
        
    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        self.class_weights = np.zeros(self.num_classes)
        
        for path in paths:
            self.paths.append(path)
           

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals, dual_classes = (
            convert_listofrollouts(paths, concat_rew))

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.dual_classes = dual_classes[-self.max_size:]


        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]
            self.dual_classes = np.concatenate(
                [self.dual_classes, dual_classes]
            )[-self.max_size:]
        
        #Update the class weights

        self.set_weights()



        
