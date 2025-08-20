import pickle
import gzip
import argparse
import pdb

def main(args):
    with gzip.open(args.dataset1, 'rb') as f1, gzip.open(args.dataset2, 'rb') as f2:
        data1 = pickle.load(f1)
        data2 = pickle.load(f2)
    # Combine the two datasets 
    keys = data1.keys()
    combined_data = {}
    for key in keys:
        if key in data2:
            combined_data[key] = data1[key] + data2[key]
        else:
            combined_data[key] = data1[key]  

    #Save the combined dataset to a new pickle file
    output_file = args.output
    data1dir = args.dataset1
    directory = ('/').join(data1dir.split('/')[:-1])+'/'
    with gzip.open(directory + output_file, 'wb') as f_out:
        pickle.dump(combined_data, f_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine two datasets from pickle files.')
    parser.add_argument('--dataset1', type=str, help='Path to the first dataset pickle file')
    parser.add_argument('--dataset2', type=str, help='Path to the second dataset pickle file')
    parser.add_argument('--output', type=str, help='Path to save the combined dataset pickle file', default='combined_dataset.pkl.gz')
    args = parser.parse_args()
    main(args)