import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


"""
This script splits videos into training and validation data in a stratified way, keeping class ratios - according
to a given class formulation (i.e. the method used to convert raw labels to classes).
It also stored the processed labels according to the class formulation in a file. 
"""

parser = ArgumentParser(
    description='Generates labels according to given strategy and index files for train and validation.',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_classes', type=int, default=3,
                    help='How many classes to create from raw labels.')
parser.add_argument('--valid_ratio', type=float, default=0.2,
                    help='Ratio of total data used for validation')
parser.add_argument('--raw_label_file_path', type=str, default='Echo-Liste_pseudonym.xlsx',
                    help='Path to the excel file storing the raw labels')


def get_legal_float_labels(raw_ph_label):
    """
    Given a raw label, return empty string if not legal label (e.g. nan, 'undecided', or wrong range).
    In case of a legal label, return the floating point equivalent of the label (0, 0.5, 1, 1.5, 2, 2.5, 3).
    (The .5 comes from labels 'between' two categories.)
    :param raw_ph_label:
    :return: Floating point mapping of the label, or 0.0 if non-legal.
    """
    if isinstance(raw_ph_label, int) and 0 <= raw_ph_label <= 3:  # legal
        return float(raw_ph_label)
    if isinstance(raw_ph_label, str):  # string is legal if contains 'bis'
        if 'bis' in raw_ph_label:
            return (int(raw_ph_label[-1]) + int(raw_ph_label[0]))/2.0
        else:  # Not a legal label (e.g. 'nichts bestimmt', etc.)
            return -1
    return -1   # if the label is not int, nor string, e.g. a nan - then not legal


label_map_3class = {
    0: 0,
    0.5: 0,
    1: 1,
    1.5: 1,
    2: 2,
    2.5: 2,
    3: 2
}

label_map_2class = {  # 0 and 0-1 is 'norma', rest (1, 1-2, 2, 2-3, 3) is 'abnormal)
    0: 0,
    0.5: 0,
    1: 1,
    1.5: 1,
    2: 1,
    2.5: 1,
    3: 1
}


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
        valid_total_cnt +=1
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
    print(cnt_test)
    print('Valid distribution:')
    print(cnt_train)


def main():
    args = parser.parse_args()
    dfs = pd.read_excel(args.raw_label_file_path, sheet_name=0, header=2)
    ph_labels = dfs['Score PH'].to_numpy()
    ids = dfs['Nummer'].to_numpy()
    kapap_videos = dfs['KAPap'].to_numpy()

    final_samples = []
    final_labels = []
    for id, video, label in zip(ids, kapap_videos, ph_labels):
        if isinstance(video, float) and np.isnan(video):
            continue
        float_label = get_legal_float_labels(label)
        if float_label != -1:  # Then it's legal
            if args.num_classes == 3:
                label_category = label_map_3class[float_label]
            elif args.num_classes == 2:
                label_category = label_map_2class[float_label]
            else:
                print("Have not yet implemented any other class numbers except 2 and 3. Try again")
                label_category = None
                exit(-1)
            final_samples.append(id)  # create a dummy array
            final_labels.append(label_category)

    print(f"Have finished creating labels for total {len(final_samples)} number of samples")
    # Save label dictionary
    label_dict = dict(zip(final_samples, final_labels))
    with open("labels" + str(args.num_classes) + ".pkl", "wb") as file:
        pickle.dump(label_dict, file)

    # Split samples into train and test, stratified according to the labels
    samples_train, samples_test, y_train, y_test = train_test_split(np.asarray(final_samples), final_labels,
                                                                    test_size=args.valid_ratio,
                                                                    shuffle=True, stratify=final_labels)
    # Save index files for train and test
    np.save('train_samples.npy', samples_train)
    np.save('test_samples.npy', samples_test)

    # Print results
    print_res(y_train, y_test)


if __name__ == '__main__':
    main()
