import os


def get_index_file_path(no_folds, curr_fold, label_type, train=True):
    idx_dir = 'index_files' if no_folds is None else os.path.join('index_files', 'k' + str(no_folds))
    idx_file_end = '' if curr_fold is None else '_' + str(curr_fold)
    idx_file_base_name = 'train_samples_' if train else 'valid_samples_'
    index_file_path = os.path.join(idx_dir, idx_file_base_name + label_type + idx_file_end + '.npy')
    return index_file_path

