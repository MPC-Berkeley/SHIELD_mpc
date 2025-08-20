import gzip
import pickle
import pdb
import numpy as np

def open_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_heading_wayformer(self, pred_filtered, init_psi):
    """
    Get the heading of the vehicle from the planner input
    :param planner_input: planner input
    :return: heading of the vehicle
    """
    # Get the ego vehicle's current state
    n_tv, M, T, _  = pred_filtered.shape
    heading = torch.zeros((n_tv, M, T-1, 1))
    heading[:,:,0,0] = init_psi
    # Get the heading of the vehicle
    anomaly_heading_idx = []
    threshold = 0.4 # Threshold for detecting anomalies in heading change (radians)
    for i in range(n_tv):
        for t in range(1,T-1):            
            #instantaneous heading
            xy_diff = pred_filtered[i, :, t+1, 0:2] - pred_filtered[i, :, t, 0:2]
            torch.atan2(xy_diff[:,1], xy_diff[:,0], out=heading[i, :, t, 0])
            for m in range(xy_diff.shape[0]):
                if xy_diff[m,1] == 0 or abs(heading[i, m, t, 0]-heading[i, m, t-1, 0])>threshold:
                    anomaly_heading_idx.append((i,m,t))
    return heading, anomaly_heading_idx

def main(file_path):
    data = open_pickle(file_path)
    num_sims = len(data['scenario_id'])
    observations = []
    for i in range(num_sims):
        ego_opt_traj = data['ego_opt_sol'][i] #List[np.array.shape = (N,4)]
        agent_preds = data['preds'][i] #List[np.array.shape = (n_tv,num_modes,4,N)]; Multi-modal predictions from Wayformer
        agent_params = data['agent_params'][i] #List[np.array.shape = (n_tv,2)]
        ego_opt_traj = list(map(lambda x: np.expand_dims(x,axis=(2,3)), ego_opt_traj))
        ego_opt_traj_transposed = list(map(np.transpose, ego_opt_traj))
        delta_traj = [a[:,:,[0,1,3],:] - b[:,:,[0,1,3],:] for a, b in zip(agent_preds, ego_opt_traj_transposed)]
        delta_traj = list(map(lambda x: np.transpose(x,(0,1,3,2)), delta_traj))
        delta_traj = list(map(lambda x: np.reshape(x,(agent_params[0].shape[0],-1)), delta_traj))

        smpc_params = data['smpc_params'][i]

        obs = [np.concatenate((a,b),axis=1) for a, b in zip(agent_params, delta_traj)]
        #flatten obs to row first (C-order)
        obs = list(map(lambda x: np.reshape(x,(1,-1)), obs))
        # #concatenate smpc_params to obs
        # obs = list(map(lambda x,y: np.concatenate((x,np.expand_dims(y,axis=0)),axis=1), obs, smpc_params))
        observations.append(obs)
    smpc_params_dim = smpc_params[0].shape
    out_data = {'observation': observations, 'dual_class': data['dual_class'], 'optimal_duals': data['optimal_duals'], 'smpc_params_dim': smpc_params_dim}

    #save out_data
    out_filename = file_path.split('.pkl.gz')[0] + '_processed.pkl.gz'
    with gzip.open(out_filename, 'wb') as f:
        pickle.dump(out_data, f)
    print(f'Processed data saved to {out_filename}')

if __name__ == "__main__":
    # filepath = '/home/mpc/nuplan-devkit/nuplan/expert_data/nuplan_expert_data_N14_wayformer_affine_training.pkl.gz'
    filepath = '/home/mpc/nuplan-devkit/nuplan/expert_data/nuplan_expert_data_N14_wayformer_affine (another copy).pkl.gz'
    main(filepath)