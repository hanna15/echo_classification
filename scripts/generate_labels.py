import pandas as pd
import numpy as np
import pickle
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.data.ph_labels import label_map_3class, label_map_2class, label_map_2class_drop0bis1_drop3, \
    get_legal_float_labels

"""
This script creates labels to use for training from raw labels, according to specified number of classes.
It saves a pickle file containing a dictionary of video_id as key and processed label as value.
Some number of classes have different possible processed labels - in this case one file for each possible strategy
is saved.
"""


parser = ArgumentParser(
    description='Generates labels according to given number of classes',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_classes', type=int, default=3,
                    help='How many classes to create from raw labels.')
parser.add_argument('--valid_ratio', type=float, default=0.2,
                    help='Ratio of total data used for validation')
parser.add_argument('--raw_label_file_path', type=str, default='Echo-Liste_pseudonym.xlsx',
                    help='Path to the excel file storing the raw labels')
parser.add_argument('--out_dir', type=str, default='label_files',
                    help='Path to directory where to store the results')


def main():
    args = parser.parse_args()
    dfs = pd.read_excel(args.raw_label_file_path, sheet_name=0, header=2)
    ph_labels = dfs['Score PH'].to_numpy()
    ids = dfs['Nummer'].to_numpy()
    kapap_videos = dfs['KAPap'].to_numpy()

    if args.num_classes == 3:
        label_maps = [label_map_3class]
    elif args.num_classes == 2:
        label_maps = [label_map_2class, label_map_2class_drop0bis1_drop3]  # have 2 possibilities for 2-class
    else:
        print("Have not yet implemented any other class numbers except 2 and 3. Try again")
        label_maps = None
        exit(-1)

    for label_map in label_maps:  # In case more than 1 label map per class number => save as seperate index files
        final_samples = []
        final_labels = []
        for id, video, label in zip(ids, kapap_videos, ph_labels):
            if isinstance(video, float) and np.isnan(video):
                continue
            float_label = get_legal_float_labels(label)
            if float_label != -1:  # Then it's legal
                label_category = label_map[0][float_label]  # first value of label_map is the actual map
                if label_category is not None:
                    final_samples.append(id)  # create a dummy array
                    final_labels.append(label_category)

        file_name = label_map[1]  # second value of the label_map is the descriptive name
        print(f"Have finished creating labels for label map {file_name}, with {len(final_samples)} samples")
        # Save label dictionary
        label_dict = dict(zip(final_samples, final_labels))
        print(label_dict)
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "labels_" + file_name + ".pkl"), "wb") as file:
            pickle.dump(label_dict, file)


if __name__ == '__main__':
    main()
