import gzip
import argparse
import pickle
import pdb
import numpy as np
from nuplan.planning.simulation.planner.utils.smpc_utils import flatten
import matplotlib.pyplot as plt

def main(filename):
    #Load the evaluation dataset
    with gzip.open('../nuplan/expert_data/' + filename,'rb') as file:
        data = pickle.load(file)
    keys = list(data.keys())
    print("The data contains the following keys:")
    for key in keys:
        print(f"{key}: {type(data[key])}, length: {len(data[key]) if hasattr(data[key], '__len__') else 'N/A'}")
    
    n = len(data['scenario_type'])

    '''
    Collision Rate
    '''
    expert_collision_count = 0
    reduced_collision_count = 0
    for i in range(n):
        reduced_collision_count += np.any(data['reduced_collisions'][i])
        expert_collision_count += np.any(data['expert_collisions'][i])
    reduced_collision_rate = reduced_collision_count/n
    expert_collision_rate = expert_collision_count/n
    print(f"Total scenarios: {n}")
    print(f"Expert collision count: {expert_collision_count:.2%}, collision rate: {expert_collision_rate:.2%}")


    perc_cost_red = []
    perc_comp_red = []
    gain_keep = []
    constr_keep = []
    recalls = []
    reduced_opt_solve_time = []
    reduced_compt_time_arr = []
    reduced_set_canon_form_mats_time = []
    reduced_raidnet_query_time = []
    reduced_safety_screening_time = []
    reduced_time_least_squares_solve_time = []
    expert_compt_time_arr = []
    comp_time_keys = list(data['reduced_computation_time'][0][0].keys())
    steps = 0
    for i in range(n):
        for t in range(1,len(data['expert_optimal'][i])):
            if data['expert_optimal'][i][t] and data['reduced_smpc_optimal'][i][t]:
                '''
                Optimal Cost
                '''
                expert_cost = data['expert_optimal_cost'][i][t]
                reduced_cost = data['reduced_smpc_optimal_cost'][i][t]
                perc_cost_red.append((expert_cost - reduced_cost)/expert_cost) # percentage cost reduction
                '''
                gain and constr keep
                '''
                gain_keep.append(2*np.sum(data['reduced_gain_keep'][i][t]))
                constr_keep.append(np.sum(data['reduced_constr_keep'][i][t]))

                '''
                Recall
                '''
                if 'ipopt' in filename:
                    target = np.fromiter(flatten(data["expert_ca"][i][t]),float) > 1e-3
                    recall = (target & data['reduced_constr_keep'][i][t].astype(bool)).sum() / target.sum()
                    recalls.append(recall)
                

                '''
                Computation time
                '''
                #dict
                reduced_comp_time = 0
                expert_comp_time = data['expert_computation_time'][i][t]['solve_time']
                for k in comp_time_keys:
                    if k in ['solve_time','raidnet_query_time','safety_screening']:
                        reduced_comp_time += data['reduced_computation_time'][i][t][k]
                reduced_opt_solve_time.append(data['reduced_computation_time'][i][t]['solve_time'])
                reduced_set_canon_form_mats_time.append(data['reduced_computation_time'][i][t]['set_canon_form_mats'])
                reduced_raidnet_query_time.append(data['reduced_computation_time'][i][t]['raidnet_query_time'])
                reduced_safety_screening_time.append(data['reduced_computation_time'][i][t]['safety_screening'])
                reduced_time_least_squares_solve_time.append(data['reduced_computation_time'][i][t]['time_least_squares_solve'])
                expert_compt_time_arr.append(expert_comp_time)
                reduced_compt_time_arr.append(reduced_comp_time)

                perc_comp_red.append((reduced_comp_time - expert_comp_time) / expert_comp_time)
                steps += 1
    # l1_num = data['expert_l1'][0][0].shape[0]
    # ca_num = data['expert_ca'][0][0].shape[0]
    l1_num = 156
    ca_num = 312

    print('Solver: ', 'IPOPT' if 'ipopt' in filename else 'Gurobi')
    print(f"Average percentage cost reduction over {steps} steps: {np.mean(perc_cost_red):.2%} \pm {np.std(perc_cost_red):.2%}")
    print(f"Average gain keep over {steps} steps: {np.mean([x /l1_num for x in gain_keep]):.2%} \pm {np.std([x /l1_num for x in gain_keep]):.2%}")

    print(f"Average constraint keep over {steps} steps: {np.mean([x /ca_num for x in constr_keep]):.2%} \pm {np.std([x /ca_num for x in constr_keep]):.2%}")
    if 'ipopt' in filename:
        print(f"Average recall over {steps} steps: {np.mean(recalls):.2%} \pm {np.std(recalls):.2%} \pm {np.std(recalls):.2%}")
    print('Computation Time'.center(50,'-'))
    print(f"Average percentage computation time reduction over {steps} steps: {np.mean(perc_comp_red):.2%} \pm {np.std(perc_comp_red):.2%}")
    print(f"Average expert computation time over {steps} steps: {np.mean(expert_compt_time_arr):.4f} \pm {np.std(expert_compt_time_arr):.4f} seconds")
    print(f"Average reduced SMPC computation time over {steps} steps: {np.mean(reduced_compt_time_arr):.4f} \pm {np.std(reduced_compt_time_arr):.4f} seconds")

    print('Reduced SMPC computation time breakdown'.center(50,'-'))
    print(f"Average reduced SMPC opt. computation time over {steps} steps: {np.mean(reduced_opt_solve_time):.4f} \pm {np.std(reduced_opt_solve_time):.4f} seconds")
    print(f"Average reduced SMPC set canon form mats time over {steps} steps: {np.mean(reduced_set_canon_form_mats_time):.4f} \pm {np.std(reduced_set_canon_form_mats_time):.4f} seconds")
    print(f"Average reduced SMPC raidnet query time over {steps} steps: {np.mean(reduced_raidnet_query_time):.4f} \pm {np.std(reduced_raidnet_query_time):.4f} seconds")
    print(f"Average reduced SMPC safety screening time over {steps} steps: {np.mean(reduced_safety_screening_time):.4f} \pm {np.std(reduced_safety_screening_time):.4f} seconds")
    print(f"Average reduced SMPC least squares solve time over {steps} steps: {np.mean(reduced_time_least_squares_solve_time):.4f} \pm {np.std(reduced_time_least_squares_solve_time):.4f} seconds")
    def hist_with_stats(ax, x, bins=30, label=None,option=False, **hist_kwargs):
        # draw histogram
        counts, edges, patches= ax.hist(x, bins=bins, alpha=1, label=label,  **hist_kwargs)
        # stats
        mu  = np.mean(x)
        sig = np.std(x, ddof=1)
        # vertical mean line
        ax.axvline(mu, linestyle='--', linewidth=2, color='k')
        # 1σ band
        ax.axvspan(mu - sig, mu + sig, alpha=0.15, color='grey')
        # annotation (place slightly above the tallest bar)
        y = (counts.max() * 1.03) if len(counts) else 0
        txt = f'μ={mu:.3g}, σ={sig:.3g}'
        if option:
            txt_option = 'w/o outliers: \n'+f'μ={0.164:.3g}, σ={0.0151:.3g}'
        # find tallest bar to anchor the text
        k = int(np.argmax(counts))
        bar = patches[k]
        bx  = bar.get_x()
        bw  = bar.get_width()
        bh  = bar.get_height()
        x_text = bx + bw + 0.01 * (edges[-1] - edges[0])   # small horizontal offset
        y_text = 5000                                        # align with top of the bar
        ax.text(x_text, y_text, txt, ha='left', va='bottom', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
                color='k')
        if option:
           ax.text(x_text, 1500, txt_option, ha='left', va='bottom', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
                color='k')         
        return mu, sig
    
    #Plot a histogram of computation time
    # outlier removal
    # reduced_opt_solve_time  = np.array(reduced_opt_solve_time)
    # idx = np.where(reduced_opt_solve_time<0.2)
    # print(np.mean(reduced_opt_solve_time[idx]),np.std(reduced_opt_solve_time[idx]))
    # print(reduced_compt_time_arr[idx].shape,reduced_compt_time_arr.shape)
    fig, ax = plt.subplots()
    hist_with_stats(ax, np.array(expert_compt_time_arr), bins=100, label='Full MPC',color='red')
    hist_with_stats(ax, reduced_compt_time_arr, bins=100, label='Reduced MPC',color='green',option=True)
    ax.legend()
    plt.xlim(0, 30)
    ax.set_xlabel('Computation Time (s)'); 
    ax.set_ylabel('Count')
    plt.show() 

    plt.figure(figsize=(10,6))
    plt.hist(reduced_opt_solve_time, bins=30, alpha=0.7, color='green')
    plt.xlabel('Reduced SMPC Optimal Solve Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Reduced SMPC Optimal Solve Time Distribution: {"IPOPT" if "ipopt" in filename else "Gurobi"} Solver')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.hist(reduced_time_least_squares_solve_time, bins=30, alpha=0.7, color='red')
    plt.xlabel('Reduced SMPC Least Squares Solve Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Reduced SMPC Least Squares Solve Time Distribution: {"IPOPT" if "ipopt" in filename else "Gurobi"} Solver')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NuPlan Closed-Loop Evaluation Stats")
    parser.add_argument(
        "--filename",
        type=str,
        default="reduced_nuplan_evaluation_N14_wayformer_affine_wayformer_gurobi.pkl.gz",
        help="Path to the evaluation data file",
    )
    main(parser.parse_args().filename)