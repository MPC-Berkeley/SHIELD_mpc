# nuplan_RAIDNET_train_joint.py

from tutorials.policies import RAID_NET, MLP      # V1 + your MLP
from tutorials.raidnet import RAID_NET_V2         # V2
import torch as th
from torch import nn
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


def Train_BC(smpc_config, config, policy, device, policy_type,
             l1_dual_dim, ca_dual_dim, l1_num, pred_mode):
    # ----------------- logging -----------------
    logdir = (
        f"{policy_type}_NuPlan_"
        f"N{smpc_config['N']}"
        f"_N_TV{smpc_config['num_tvs']}_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    model_name = logdir
    logdir = os.path.join(config['root_dir'] + config['training_log_path'], logdir)
    if not os.path.exists(logdir):
        print(logdir)
        os.makedirs(logdir)
    logger = Logger(logdir)

    normalize = config['normalize']
    N_EPOCHS  = config['n_epochs']
    optimizer = config['optimizer']
    lr        = config['lr']

    # ----------------- load expert data -----------------
    with gzip.open(config['expert_data_dir'], 'rb') as file:
        expert_data = pickle.load(file)
    replay_buffer = ReplayBuffer(config['max_replay_buffer_size'])

    # remove empties
    expert_data['observation']   = [obs for obs in expert_data['observation']   if len(obs) > 0]
    expert_data['optimal_duals'] = [acs for acs in expert_data['optimal_duals'] if len(acs) > 0]
    expert_data['dual_class']    = [acs for acs in expert_data['dual_class']    if len(acs) > 0]

    # length guard (dataset bug mitigation)
    for j in range(len(expert_data['optimal_duals'])):
        if len(expert_data['optimal_duals'][j]) > len(expert_data['observation'][j]):
            expert_data['optimal_duals'][j] = expert_data['optimal_duals'][j][:-1]
        elif len(expert_data['optimal_duals'][j]) == len(expert_data['observation'][j]):
            pass
        else:
            print('Length mismatch between observation and optimal duals')
            pdb.set_trace()

    observation   = np.squeeze(np.concatenate([obs for obs in expert_data["observation"]]), axis=1)
    optimal_duals = np.concatenate([acs for acs in expert_data["optimal_duals"]])

    # fill buffer
    replay_buffer.obs        = observation
    replay_buffer.acs        = optimal_duals
    replay_buffer.opt_duals  = optimal_duals
    replay_buffer.terminals  = np.zeros_like(observation)
    replay_buffer.next_obs   = np.zeros_like(observation)
    replay_buffer.rews       = np.zeros_like(observation)
    replay_buffer.smpc_params_dim = expert_data['smpc_params_dim']

    flattened_dual_class = []
    for data in expert_data["dual_class"]:
        flattened_dual_class.extend(data)
    replay_buffer.dual_classes = np.array(flattened_dual_class)

    # Train L1 as **binary** derived from tertiary:
    # mapping: class 2->0, class 1->1, class 0->0 (active iff class==1).
    replay_buffer.normalize(
        l1_num,
        l1_lmbd=smpc_config['l1_lmbd'],
        l1_pred_mode='binary',           # <--- key line (tertiary -> binary mapping inside)
        policy_type=policy_type
    )
    replay_buffer.set_weights()

    # set obs stats for unnormalization
    policy._set_obs_stats(replay_buffer.feature_mean, replay_buffer.feature_cov)

    # Single (joint) model path
    bc_learner = BC(
        policy=policy,
        optimizer=optimizer,
        optim_lr=lr,
        demonstrations=replay_buffer,
        rng=np.random.default_rng(0),
        device=device,
        batch_size=config['batch_size'],
        logger=Logger(logdir),
        normalize=normalize,
        config=config,
        normalize_obs=normalize,
        l1_dual_dim=l1_dual_dim,
        ca_dual_dim=ca_dual_dim,
        joint_dual_pred=True,             # <--- train jointly
        policy_type=policy_type
    )

    if config.get('pretrain', False):
        print('Loading pretrained model...')
        checkpoint = th.load(config['pretrain_joint_path'])  # if you have one
        bc_learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        bc_learner.policy.load_state_dict(checkpoint['model_state_dict'])

    print('TRAINING STARTED'.center(80, '*'))
    training_log = bc_learner.train(
        n_epochs=N_EPOCHS,
        model_name=model_name,
        pred_mode=['both duals', 'binary', 'binary']  # joint head, L1 binary, CA binary
    )

    bc_learner.save(config['model_save_dir'], config=config)
    with open(config['root_dir'] + config['training_log_path'] + 'training_log.pkl', 'wb') as handle:
        pickle.dump(training_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return bc_learner


def build_joint_policy(policy_type: str,
                       raidnet_config: dict,
                       smpc_config: dict,
                       config: dict,
                       observation_dim: int,
                       total_out_dim: int,
                       l1_num: int):
    """
    Create a single (joint) model (V1, V2, or MLP) that outputs a flat tensor of size (l1_num + ca_num).
    pred_mode=['both duals','binary','binary'] so both heads are treated as binary logits.
    """
    per_tv_in = int(observation_dim / smpc_config['num_tvs'])
    Nm1       = config['N'] - 1
    num_tvs   = smpc_config['num_tvs']
    nlayers   = max(1, int(config['num_layers'] // 2))
    hdim      = max(8, int(config['hidden_dim'] // 2))   # keep it sane
    lmbd_ubd  = smpc_config['l1_lmbd']

    pt = (policy_type or '').upper()

    if pt in ['RAIDNET_V1_JOINT', 'RAIDNET_V1']:
        # RAID-NET V1 (joint)
        model = RAID_NET(
            raidnet_config,
            per_tv_in,                 # per-TV slice
            observation_dim,
            total_out_dim,             # joint output = l1_num + ca_num
            Nm1,
            num_tvs,
            nlayers,
            hdim,
            lambda_dim=total_out_dim,  # expose all logits
            lambda_ubd=lmbd_ubd,
            pred_mode=['both duals', 'binary', 'binary']  # flat joint logits
        )
        model._name = 'RAIDNET_V1_JOINT'
        return model

    if pt in ['RAIDNET_V2_JOINT', 'RAIDNET_V2']:
        # RAID-NET V2 (joint)
        model = RAID_NET_V2(
            raidnet_config,
            per_tv_in,
            observation_dim,
            total_out_dim,
            Nm1,
            num_tvs,
            nlayers,
            hdim,
            lambda_dim=total_out_dim,
            lambda_ubd=lmbd_ubd,
            pred_mode=['both duals', 'binary', 'binary']
        )
        model._name = 'RAIDNET_V2_JOINT'
        return model

    if pt in ['MLP_JOINT', 'MLP']:
        # Your requested joint MLP (flat logits of size total_out_dim)
        model = MLP(
            input_dim=observation_dim,
            output_dim=total_out_dim,
            hidden_layers=int(config['num_layers']),
            hidden_size=int(config['hidden_dim']),
            device=str(th.device("cuda:0" if th.cuda.is_available() else "cpu")),
            tertiary=False  # joint binary logits, not 3-way
        )
        # ensure BC has what it expects for joint models
        model.lmbd_ubd = float(lmbd_ubd)
        model.output_dim = total_out_dim
        model._name = 'MLP_JOINT'
        return model

    raise NotImplementedError(f'Unknown policy_type="{policy_type}". '
                              f'Use RAIDNET_V1_JOINT, RAIDNET_V2_JOINT, or MLP_JOINT.')


def main(smpc_config, config):
    # problem sizes
    n_modes = [smpc_config['num_modes'] for _ in range(smpc_config['num_tvs'])]
    mode_map = dict(enumerate(product(*[range(n_modes[k]) for k in range(smpc_config['num_tvs'])])))
    observation_dim = smpc_config['num_tvs'] * (smpc_config['num_modes'] * (3 * config['N']) + 2)
    ca_num = len(mode_map) * (config['N'] - 1) * smpc_config['num_tvs']
    l1_num = sum(n_modes)   * (config['N'] - 1) * 2
    total_out_dim = l1_num + ca_num

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    l1_dual_dim = [config['N'] - 1, n_modes, smpc_config['num_tvs']]
    ca_dual_dim = [config['N'] - 1, len(mode_map), smpc_config['num_tvs']]

    raidnet_config = {
        'num_tvs': smpc_config['num_tvs'],
        'num_heads': config['num_heads'],
        'dropout_prob': config['dropout_prob']
    }

    # Choose joint policy type here (or pass via CLI):
    policy_type = (config.get('policy_type') or 'RAIDNET_V2_JOINT')  # RAIDNET_V1_JOINT / RAIDNET_V2_JOINT / MLP_JOINT

    # ----------------- SINGLE MODEL (JOINT) -----------------
    policy = build_joint_policy(
        policy_type=policy_type,
        raidnet_config=raidnet_config,
        smpc_config=smpc_config,
        config=config,
        observation_dim=observation_dim,
        total_out_dim=total_out_dim,
        l1_num=l1_num
    )
    policy.to(device)

    print(l1_num, ca_num)
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Number of trainable parameters (JOINT): {trainable_params}")

    # Train
    _ = Train_BC(
        smpc_config, config, policy, device, policy_type,
        l1_dual_dim=l1_dual_dim, ca_dual_dim=ca_dual_dim,
        l1_num=l1_num, pred_mode=['both duals', 'binary', 'binary']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpc_config', required=False, type=str,
                        default='/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_config.yaml')
    parser.add_argument('--config', required=False, type=str,
                        default='/home/mpc/nuplan-devkit/tutorials/training_config.yaml')
    parser.add_argument('--policy_type', required=False, type=str,
                        default=None, help='RAIDNET_V1_JOINT | RAIDNET_V2_JOINT | MLP_JOINT')
    args = parser.parse_args()

    with open(args.smpc_config, 'r') as f:
        smpc_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # allow override via CLI
    if args.policy_type is not None:
        config['policy_type'] = args.policy_type

    main(smpc_config, config)
