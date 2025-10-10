from tutorials.policies import RAID_NET
from tutorials.raidnet import RAID_NET_V2
import torch as th
import argparse
import yaml
import numpy as np
import pdb
import os
import time
from itertools import product
from tutorials.utils.replay_buffer import ReplayBuffer
from tutorials.utils.BC import BC
from tutorials.utils.logger import Logger
import pickle
import gzip

def Train_BC(smpc_config,config,policy,device,policy_type,l1_dual_dim,ca_dual_dim,l1_num,pred_mode):
    #Initiate logger
    logdir = (
        policy_type
        + "_"
        + 'NuPlan'
        + "_"
        + "N" + str(smpc_config['N'])
        + "_N_TV" + str(smpc_config['num_tvs'])
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    model_name = logdir

    logdir = os.path.join(config['root_dir'] + config['training_log_path'], logdir)
    if not (os.path.exists(logdir)):
        print(logdir)
        os.makedirs(logdir)
    logger = Logger(logdir)

    normalize = config['normalize']
    N_EPOCHS = config['n_epochs']
    optimizer = config['optimizer']
    lr = config['lr']

    #Load expert dataset
    with gzip.open(config['expert_data_dir'],'rb') as file:
        expert_data = pickle.load(file)
    replay_buffer = ReplayBuffer(config['max_replay_buffer_size'])

    #remove empty list in expert_data
    expert_data['observation'] = [obs for obs in expert_data['observation'] if len(obs)>0]
    expert_data['optimal_duals'] = [acs for acs in expert_data['optimal_duals'] if len(acs)>0]
    expert_data['dual_class'] = [acs for acs in expert_data['dual_class'] if len(acs)>0]

    #For debugging issue. This issue happened due to some error in smpc_planner.py. I addressed this issue and the new dataset is being collected 
    for j in range(len(expert_data['optimal_duals'])):
        if (len(expert_data['optimal_duals'][j]) > len(expert_data['observation'][j])):
            #delete the last element of optimal duals
            expert_data['optimal_duals'][j] = expert_data['optimal_duals'][j][:-1]
        elif (len(expert_data['optimal_duals'][j]) == len(expert_data['observation'][j])):
            pass
        else:
            print('Length mismatch between observation and optimal duals')
            pdb.set_trace()

    observation = np.squeeze(np.concatenate([obs for obs in expert_data["observation"]]),axis=1)
    optimal_duals = np.concatenate([acs for acs in expert_data["optimal_duals"]])

    replay_buffer.obs = observation; replay_buffer.acs = optimal_duals; replay_buffer.opt_duals = optimal_duals; replay_buffer.terminals = np.zeros_like(observation); replay_buffer.next_obs = np.zeros_like(observation); replay_buffer.rews = np.zeros_like(observation) 
    replay_buffer.smpc_params_dim = expert_data['smpc_params_dim']

    flattened_dual_class = []
    for data in expert_data["dual_class"]:
        flattened_dual_class.extend(data)
    replay_buffer.dual_classes = np.array(flattened_dual_class)
    replay_buffer.normalize(l1_num,l1_lmbd=smpc_config['l1_lmbd'],l1_pred_mode=config['l1_pred_mode'],policy_type=policy_type)
    replay_buffer.set_weights()

    #Set the observation statistics for the policy for unnormalization
    if isinstance(policy, list):
        for pol in policy:
            pol._set_obs_stats(replay_buffer.feature_mean,replay_buffer.feature_cov)
    else:
        policy._set_obs_stats(replay_buffer.feature_mean,replay_buffer.feature_cov)

    bc_learner = BC(policy=policy,optimizer=optimizer,optim_lr=lr, demonstrations=replay_buffer, rng = np.random.default_rng(0),device = device, batch_size=config['batch_size'],logger=logger,normalize=normalize,config=config, normalize_obs=normalize, l1_dual_dim= l1_dual_dim,ca_dual_dim= ca_dual_dim,joint_dual_pred=config['joint_dual_pred'],policy_type=policy_type)
    if config['pretrain']:
        print('Loading pretrained model...')
        checkpoint = []
        checkpoint.append(th.load('/home/mpc/nuplan-devkit/nuplan/nn_models/RAIDNET_NuPlan_N14_N_TV3_12-08-2025_19-07-11/RAIDNET_NuPlan_N14_N_TV3_12-08-2025_19-07-11_L1_100.pt'))
        checkpoint.append(th.load('/home/mpc/nuplan-devkit/nuplan/nn_models/RAIDNET_NuPlan_N14_N_TV3_12-08-2025_19-07-11/RAIDNET_NuPlan_N14_N_TV3_12-08-2025_19-07-11_CA_100.pt'))
        #L1
        bc_learner.optimizer[0].load_state_dict(checkpoint[0]['optimizer_state_dict'])
        bc_learner.policy[0].load_state_dict(checkpoint[0]['model_state_dict'])
        #CA
        bc_learner.optimizer[1].load_state_dict(checkpoint[1]['optimizer_state_dict'])
        bc_learner.policy[1].load_state_dict(checkpoint[1]['model_state_dict'])


    print('TRAINING STARTED'.center(80,'*'))
    training_log = bc_learner.train(n_epochs=N_EPOCHS,model_name=model_name,pred_mode=pred_mode)

    bc_learner.save(config['model_save_dir'],config=config)
    with open(config['root_dir']+config['training_log_path'] + 'training_log.pkl', 'wb') as handle:
        pickle.dump(training_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return bc_learner

def main(smpc_config,config):
    # n_modes = [2]*2 + [1]*(smpc_config['num_tvs']-2) #2 vehicles with lane change modes
    n_modes = [smpc_config['num_modes'] for _ in range(smpc_config['num_tvs'])]
    mode_map = dict(enumerate(product(*[range(n_modes[k]) for k in range(smpc_config['num_tvs'])])))
    observation_dim = smpc_config['num_tvs'] * (smpc_config['num_modes'] * (3*config['N']) + 2)
    ca_num = len(mode_map)*(config['N']-1)*smpc_config['num_tvs']
    l1_num = sum(n_modes)*(config['N']-1)*2
    num_layers = config['num_layers']
    hidden_dim = config['hidden_dim']
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu") 

    l1_dual_dim = [config['N']-1, n_modes, smpc_config['num_tvs']]
    ca_dual_dim = [config['N']-1, len(mode_map), smpc_config['num_tvs']]

    raidnet_config = {'num_tvs': smpc_config['num_tvs'], 'num_heads': config['num_heads'],'dropout_prob':config['dropout_prob']}
    #Initialize RAIDNET
    policy_type = 'RAIDNET_V1'
    pred_mode = ['both duals','tertiary','binary']
    if policy_type == 'RAIDNET_V1':
        l1_policy = RAID_NET(raidnet_config,int(observation_dim/(smpc_config['num_tvs'])), observation_dim, l1_num, config['N']-1, smpc_config['num_tvs'], num_layers//2, hidden_dim//2,lambda_dim=l1_num, lambda_ubd=smpc_config['l1_lmbd'], pred_mode=['l1','tertiary','binary'])
        ca_policy = RAID_NET(raidnet_config,int(observation_dim/(smpc_config['num_tvs'])), observation_dim, ca_num, config['N']-1, smpc_config['num_tvs'], num_layers//2, hidden_dim//2,lambda_dim=ca_num, lambda_ubd=smpc_config['l1_lmbd'], pred_mode=['ca','binary','binary'])
    elif policy_type == 'RAIDNET_V2':
        l1_policy = RAID_NET_V2(raidnet_config,int(observation_dim/(smpc_config['num_tvs'])), observation_dim, l1_num, config['N']-1, smpc_config['num_tvs'], num_layers//2, hidden_dim//2,lambda_dim=l1_num, lambda_ubd=smpc_config['l1_lmbd'], pred_mode=['l1','tertiary','binary'])
        ca_policy = RAID_NET_V2(raidnet_config,int(observation_dim/(smpc_config['num_tvs'])), observation_dim, ca_num, config['N']-1, smpc_config['num_tvs'], num_layers//2, hidden_dim//2,lambda_dim=ca_num, lambda_ubd=smpc_config['l1_lmbd'], pred_mode=['ca','binary','binary'])
    else:
        raise NotImplementedError(f'Policy type {policy_type} not implemented.')
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    l1_policy.to(device)
    ca_policy.to(device)
    policy = [l1_policy, ca_policy]
    print(l1_num,ca_num)
    trainable_params = sum(p.numel() for p in l1_policy.parameters() if p.requires_grad)
    print(f"Number of trainable parameters (L1): {trainable_params}")
    trainable_params = sum(p.numel() for p in ca_policy.parameters() if p.requires_grad)
    print(f"Number of trainable parameters (CA): {trainable_params}")
    checkpoint = None

    bc_learner = Train_BC(smpc_config,config,policy,device,policy_type,l1_dual_dim=l1_dual_dim,ca_dual_dim=ca_dual_dim,l1_num=l1_num,pred_mode=pred_mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpc_config', required=False,type=str, default='/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_config.yaml')
    parser.add_argument('--config', required=False,type=str, default='/home/mpc/nuplan-devkit/tutorials/training_config.yaml')
    args = parser.parse_args()
    with open(args.smpc_config, 'r') as f:
        smpc_config = yaml.load(f,Loader=yaml.FullLoader)
    with open(args.config, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    main(smpc_config,config)