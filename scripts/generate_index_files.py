import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

"""
This script splits videos into training and validation data in a stratified way, keeping class ratios - according
to a given class formulation (i.e. the method used to convert raw labels to classes).
"""

parser = ArgumentParser(
    description='Generates index files for train and validation.',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--label_file_path', type=str, default=None,
                    help='Full path to the pickle file containing the desired labels - to split evenly.'
                         'E.g.: label_files/labels_3class.pkl. If None - apply for all files in --label_dir instead')
parser.add_argument('--label_dir', type=str, default='label_files',
                    help='Path to directory containing all label files - in order to create a pair of index files for '
                         'each label file. Set to None, in order to only create index files for a given label file')
parser.add_argument('--valid_ratio', type=float, default=0.2,
                    help='Ratio of total data used for validation')
parser.add_argument('--video_cache_dir', type=str, default='~/.heart_echo',
                    help='Path of the video cache dir (usually: ~/.heart_echo).')
parser.add_argument('--scale_factor', default=0.5,
                    help='Scaling factor of the cached videos')
parser.add_argument('--out_dir', default='index_files',
                    help='Path to directory where results should be stored')


def print_res(train_labels, valid_labels):
    """
    Prints the results, i.e. the ratio of each label.
    :param train_labels: List of labels for train samples
    :param valid_labels: List of labels for valid samples
    :return:
    """
    cnt_test = dict()
    cnt_train = dict()
    train_total_cnt = 0
    valid_total_cnt = 0
    for test_label in valid_labels:
        if test_label in cnt_test:
            cnt_test[test_label] += 1
        else:
            cnt_test[test_label] = 1
        valid_total_cnt += 1
    for train_label in train_labels:
        if train_label in cnt_train:
            cnt_train[train_label] += 1
        else:
            cnt_train[train_label] = 1
        train_total_cnt += 1

    print()
    print('Number of training samples:', train_total_cnt)
    print('Number of valid samples:', valid_total_cnt)
    for label in cnt_train.keys():
        print(f'Train ratio label {label}: {cnt_train[label] / train_total_cnt}')
    print('Train distribution:')
    print(cnt_train)
    print('Valid distribution:')
    print(cnt_test)


def main():
    args = parser.parse_args()
    video_cache_dir = os.path.join(os.path.expanduser(args.video_cache_dir), str(args.scale_factor))
    video_ending = 'KAPAP.npy'
    kapap_cache_videos = [video for video in os.listdir(video_cache_dir) if video.endswith(video_ending)]
    label_dicts = []
    label_files = []
    if args.label_file_path is None:  # create index files for all label files in label directory
        for label_file in os.listdir(args.label_dir):
            label_files.append(label_file)
            label_file_path = os.path.join(args.label_dir, label_file)
            with open(label_file_path, "rb") as file:
                label_dict = pickle.load(file)
            label_dicts.append(label_dict)

    for label_dict, label_file in zip(label_dicts, label_files):
        print("Results for", label_file)
        labels_in_use = []
        video_ids_in_use = []
        video_ending_len = len(video_ending)
        for video in kapap_cache_videos:
            video_id = int(video[:-video_ending_len])
            if video_id not in label_dict:
                print(f'video {video_id} does not have a legal label - skipping')
            else:
                label = label_dict[video_id]
                labels_in_use.append(label)
                video_ids_in_use.append(video_id)
        # Split samples into train and test, stratified according to the labels
        samples_train, samples_test, y_train, y_test = train_test_split(np.asarray(video_ids_in_use), labels_in_use,
                                                                        test_size=args.valid_ratio,
                                                                        shuffle=True, stratify=labels_in_use)
        # Save index files for train and test
        os.makedirs(args.out_dir, exist_ok=True)
        file_name = label_file.split('labels_')[1][:-4]
        np.save(os.path.join(args.out_dir, 'train_samples_' + file_name + '.npy'), samples_train)
        np.save(os.path.join(args.out_dir, 'test_samples_' + file_name + '.npy'), samples_test)

        # Print results
        print_res(y_train, y_test)


if __name__ == '__main__':
    main()
