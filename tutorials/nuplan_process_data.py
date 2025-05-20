import gzip
import pickle
import pdb
import numpy as np

def open_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def main(file_path):
    data = open_pickle(file_path)
    num_sims = len(data['scenario_id'])
    observations = []
    for i in range(num_sims):
        ego_opt_traj = data['ego_opt_sol'][i] #List[np.array.shape = (N,4)]
        agent_preds = data['preds'][i] #List[np.array.shape = (n_tv,4,N)]
        agent_params = data['agent_params'][i] #List[np.array.shape = (n_tv,2)]
        ego_opt_traj = list(map(lambda x: np.expand_dims(x,axis=2), ego_opt_traj))
        ego_opt_traj_transposed = list(map(np.transpose, ego_opt_traj))
        delta_traj = [a - b for a, b in zip(agent_preds, ego_opt_traj_transposed)]
        delta_traj = list(map(lambda x: np.transpose(x,(0,2,1)), delta_traj))
        delta_traj = list(map(lambda x: np.reshape(x,(agent_params[0].shape[0],-1)), delta_traj))

        smpc_params = data['smpc_params'][i]

        obs = [np.concatenate((a,b),axis=1) for a, b in zip(agent_params, delta_traj)]
        #flatten obs to row first 
        obs = list(map(lambda x: np.reshape(x,(1,-1)), obs))

        #concatenate smpc_params to obs
        obs = list(map(lambda x,y: np.concatenate((x,np.expand_dims(y,axis=0)),axis=1), obs, smpc_params))
        observations.append(obs)
    smpc_params_dim = smpc_params[0].shape
    out_data = {'observation': observations, 'dual_class': data['dual_class'], 'optimal_duals': data['optimal_duals'], 'smpc_params_dim': smpc_params_dim}

    #save out_data
    out_filename = file_path.split('.pkl.gz')[0] + '_processed.pkl.gz'
    with gzip.open(out_filename, 'wb') as f:
        pickle.dump(out_data, f)
    print(f'Processed data saved to {out_filename}')

if __name__ == "__main__":
    filepath = '/home/mpc/nuplan-devkit/nuplan/expert_data/nuplan_expert_data_N15.pkl.gz'
    main(filepath)