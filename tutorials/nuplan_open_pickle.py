import pickle
import gzip
import argparse 
import pdb
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patches
from nuplan.common.actor_state.agent import Agent
import matplotlib.pyplot as plt
from typing import List

def load_simulation_log(filepath: str):
    """Load simulation log from a pickle file."""
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
    data_temp = {}
    if isinstance(data['iteration_data'][0],List):
        for key in data.keys():
            temp = []
            if key != 'log_iter':
                for i in range(len(data[key])):
                    temp.extend(data[key][i])
                data_temp.update({key: temp})
            else:
                data_temp.update({key: data[key]})
        return data_temp
    else:
        return data

def visualize_observation(data,t):
    """Replay the observation from the simulation log."""
    # Implement your replay logic here
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ego_state = data['ego_states'][t]
    # Plot ego vehicle
    ego_x, ego_y = ego_state.center.point.x, ego_state.center.point.y
    ego_heading = ego_state.center.heading  # In radians
    ego_length, ego_width = ego_state.car_footprint.vehicle_parameters.length, ego_state.car_footprint.vehicle_parameters.width
    ego_box = patches.Rectangle((ego_x - ego_length/2, ego_y - ego_width/2), ego_length, ego_width, 
                                angle=np.degrees(ego_heading),
                                edgecolor='green', facecolor='green', alpha=1)
    ax.add_patch(ego_box)
    # for t in range(len(data['observation'])):
    observations = data['iteration_data'][t].tracked_objects.tracked_objects
    
    for obs in observations:
        if isinstance(obs, Agent):
            x, y = obs.box.center.x, obs.box.center.y
            width, length = obs.box.width, obs.box.length
            heading = obs.box.center.heading

            # Add rectangle for detected object
            det_box = patches.Rectangle((x - length / 2, y - width / 2), length, width, 
                                        angle=np.degrees(heading),
                                        edgecolor='red', facecolor='red', alpha=1)
            ax.add_patch(det_box)
    
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    ax.set_xlim(ego_x - 30, ego_x + 30)
    ax.set_ylim(ego_y - 30, ego_y + 30)
    plt.legend()
    plt.title("Ego and Observations Visualization")
    plt.show()

def replay_simulation(data):
    """Replay the simulation log. Save it as a video """
    # Implement your replay logic here
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    def init():
        artists = []
        return artists
    
    def animate(frame):
        ax.clear()
        artists = []
    
        observations = data['iteration_data'][frame].tracked_objects.tracked_objects
        ego_state = data['ego_states'][frame]
        # Plot ego vehicle
        ego_x, ego_y = ego_state.center.point.x, ego_state.center.point.y
        ego_heading = ego_state.center.heading  # In radians
        ego_length, ego_width = ego_state.car_footprint.vehicle_parameters.length, ego_state.car_footprint.vehicle_parameters.width
        ego_box = patches.Rectangle((ego_x - ego_length/2, ego_y - ego_width/2), ego_length, ego_width,
                                    angle=np.degrees(ego_heading),
                                    edgecolor='green', facecolor='green', alpha=1)
        for obs in observations:
            if isinstance(obs, Agent):
                x, y = obs.box.center.x, obs.box.center.y
                width, length = obs.box.width, obs.box.length
                heading = obs.box.center.heading

                # Add rectangle for detected object
                det_box = patches.Rectangle((x - length / 2, y - width / 2), length, width,
                                            angle=np.degrees(heading),
                                            edgecolor='red', facecolor='red', alpha=1)
                artists.append(ax.add_patch(det_box))

        # Add the patches to the plot
        artists.append(ax.add_patch(ego_box))
        ax.set_xlim(ego_x - 30, ego_x + 30)
        ax.set_ylim(ego_y - 30, ego_y + 30)
        return artists
        
    # Implement video saving logic here
    ani = animation.FuncAnimation(fig, animate, init_func=init, repeat=False, blit=True, interval=100, frames=range(len(data['observation'])))
    filename = '/home/mpc/nuplan-devkit/nuplan/expert_data/simulation_replay.mp4'
    ani.save(filename, writer='ffmpeg')
    plt.close(fig)
    pass

def main(args):
    print(args.filepath)
    data = load_simulation_log(args.filepath)
    print(data.keys())
    y = np.vstack(data['optimal_duals'])
    #PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(y[:10000,:])
    y_pca = pca.transform(y[10000:,:])
    y 
    pdb.set_trace()
    # print(data['optimal_duals'])
    # print(data['observation'])
    # print(data['dual_class'])
    # print(data['preds'])
    # visualize_observation(data,0)
    # replay_simulation(data)
    #Plot the histogram of dual class
    plt.hist(data['dual_class'], bins=3)
    plt.xlabel('Dual Class')
    plt.ylabel('Frequency')
    plt.title('Histogram of Dual Class')
    plt.show()
    # pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a simulation log from a pickle file.')
    parser.add_argument('--filepath', type=str, help='Path to the pickle file.', required=True)
    args = parser.parse_args()
    main(args)