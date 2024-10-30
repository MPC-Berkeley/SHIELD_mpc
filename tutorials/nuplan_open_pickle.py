import pickle
import argparse 
import pdb
import matplotlib.pyplot as plt

def load_simulation_log(file_path: str):
    """Load simulation log from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def main(args):
    print(args.file_path)
    data = load_simulation_log(args.file_path)
    print(data.keys())
    print(data['optimal_duals'])
    # print(data['observation'])
    print(data['dual_class'])
    # print(data['preds'])

    #Plot the histogram of dual class
    plt.hist(data['dual class'], bins=3)
    plt.xlabel('Dual Class')
    plt.ylabel('Frequency')
    plt.title('Histogram of Dual Class')
    plt.show()
    pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a simulation log from a pickle file.')
    parser.add_argument('--file_path', type=str, help='Path to the pickle file.', required=True)
    args = parser.parse_args()
    main(args)