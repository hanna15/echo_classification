import numpy as np
import os
import pandas as pd
import torch
import csv
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score, roc_curve, balanced_accuracy_score
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt

parser = ArgumentParser(
    description='Get metrics',
    formatter_class=ArgumentDefaultsHelpFormatter)
# Paths, file name, model names, etc
parser.add_argument('--res_dir', type=str, default=None,
                    help='Set path of directory containing results for all desired runs - if this is desired.')
parser.add_argument('--run_paths', type=str, default=None, nargs='+',
                    help='Set paths to all individual runs that you wish to get results for - if this is desired.')
parser.add_argument('--out_names', type=str, default=None, nargs='+',
                    help='In the case of multiple res-paths, it is optional to provide a shorter name for each run, '
                         'to be used for results')
parser.add_argument('--out_dir', type=str, default='metric_results')
parser.add_argument('--cr',  action='store_true', help='Set this flag to save also classification report per run')
parser.add_argument('--cm',  action='store_true', help='Set this flag to also save confusion matrix per run')
parser.add_argument('--train',  action='store_true', help='Set this flag to save also classification report per run')
parser.add_argument('--only_plot',  action='store_true', help='Set this flag to only plot ROC_AUC')
parser.add_argument('--plot_title',  type=str, default=None, nargs='+', help='title of ROC_AUC plot, if not default')

metrics = ['Frame ROC_AUC', 'Frame bACC', 'Video ROC_AUC', 'Video bACC', 'Video F1 (macro)', 'Video F1, pos',
           'Video F1, neg', 'Video CI']


def get_save_classification_report(targets, preds, file_name, metric_res_dir='results', epochs=None):
    """
    Get classification report for the given targets and predictions, and save it.
    Furthermore, save the epoch that the model stopped training, if epochs is specified.
    :param targets: Ground truth labels
    :param preds: Model predicted labels
    :param file_name: Name of the resulting file
    :param metric_res_dir: Name of the base result directory
    :param epochs: List of epochs per fold, if desired to save this info. Else None
    """
    report = classification_report(targets, preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['support'] = df['support'].astype(int)
    file_name = os.path.join(metric_res_dir, 'classification_reports', file_name)
    df.to_csv(file_name, float_format='%.2f')
    if epochs is not None:  # Add also epochs info
        with open(file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(['epochs'] + epochs)


def get_save_confusion_matrix(targets, preds, file_name, metric_res_dir='results'):
    """
    Get classification report for the given targets and predictions, and save it.
    Furthermore, save the epoch that the model stopped training, if epochs is specified.
    :param targets: Ground truth labels
    :param preds: Model predicted labels
    :param file_name: Name of the resulting file
    :param metric_res_dir: Name of the base result directory
    :param epochs: List of epochs per fold, if desired to save this info. Else None
    """
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    cm_dir = os.path.join(metric_res_dir, 'confusion_matrix')
    os.makedirs(cm_dir, exist_ok=True)
    file_name = os.path.join(cm_dir, file_name)
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['', '', 'True', 'True'])
        writer.writerow(['', '', '1', '0'])
        writer.writerow(['Predicted', '1', tp, fp])
        writer.writerow(['Predicted', '0', fn, tn])


def read_results(res_dir, subset='val'):
    """
    Read (get) results for model (preds, targets, samples) from numpy files
    :param res_dir: directory of model results
    :param subset: train or val
    :return: list of model predictions, list of targets, list of sample names
    """
    outs = np.load(os.path.join(res_dir, f'{subset}_preds.npy'))
    targets = np.load(os.path.join(res_dir, f'{subset}_targets.npy'))
    samples = np.load(os.path.join(res_dir, f'{subset}_samples.npy'))
    sm = torch.nn.Softmax(dim=-1)
    if len(outs) == 0:
        return None, None, None, None
    if isinstance(outs[0], (list, np.ndarray)):
        preds = np.argmax(outs, axis=1)
        # out_1s = outs[:, 1]  # extract output for class 1
        soft_m = np.asarray(sm(torch.tensor(outs)))  # get soft-maxed prob corresponding to class 1
        probs = soft_m[:, 1]
    else:
        preds = outs
        probs = None
    return preds, probs, targets, samples


def get_metrics_for_run(res_base_dir, run_name, out_dir, subset='val', get_clf_report=False, get_confusion=False):
    """
    Get list of metrics (in a string format) for current model / run, averaged over all fold, and per-video
    :param res_base_dir: Directory where results for this model are stored
    :param run_name: Name of this model / run
    :param out_dir: Name of directory to store resulting metrics
    :param col: Colour for this run, for ROC_AUC plot
    :param subset: train or val
    :param get_clf_report: Whether or not to also get classification report (frame-wise)
    :param first: Set to true, if this is the first run
    :param out_name: Shorter name to use for saving results for this run, if desired.
    :return: list of metric strings, to be written to csv
    """
    res_path = os.path.join(res_base_dir, run_name) if res_base_dir is not None else run_name
    # start with first fold, ignore .DS_store and other non-dir files
    fold_paths = [os.path.join(res_path, fold_path) for fold_path in sorted(os.listdir(res_path)) if
                  os.path.isdir(os.path.join(res_path, fold_path))]
    preds = []
    probs = []
    targets = []
    samples = []
    for fold_dir in fold_paths:
        # fold_dir = os.path.join(res_path, fold_dir)
        fold_preds, fold_probs, fold_targets, fold_samples = read_results(fold_dir, subset)
        preds.extend(fold_preds)
        probs.extend(fold_probs)
        targets.extend(fold_targets)
        samples.extend(fold_samples)

    unique_video_res = {}  # is format is {'vid_id': target, [predicted frames]}
    i = 0
    for targ, pred, prob, sample in zip(targets, preds, probs, samples):
        vid_id = sample.split('_')[0]
        if vid_id in unique_video_res:
            unique_video_res[vid_id][1].append(pred)  # append all predictions for the given video
            unique_video_res[vid_id][2].append(prob)
        else:
            unique_video_res[vid_id] = (targ, [pred], [prob])  # first initialise it

    video_targets = []
    video_preds = []
    video_probs = []
    out_probs = []
    video_ci = []  # confidence interval
    # Get a single prediction per video & get combined soft-maxed probs
    for res in unique_video_res.values():
        curr_targ = res[0]
        curr_preds = res[1]
        curr_out_probs = res[2]
        ratio_pred_1 = np.sum(curr_preds) / len(curr_preds)
        ratio_pred_0 = 1 - ratio_pred_1
        pred = 1 if ratio_pred_1 >= 0.5 else 0  # Change to a single pred value per video
        video_ci.append(ratio_pred_1 if pred == 1 else ratio_pred_0)
        video_probs.append(ratio_pred_1)
        video_preds.append(pred)
        video_targets.append(curr_targ)
        out_probs.append(np.nanmean(curr_out_probs))
    print(run_name)
    print(video_ci)
    if get_clf_report:  # Classification report on a frame-level
        get_save_classification_report(video_targets, video_preds, f'{subset}_report_video_{run_name}.csv',
                                       metric_res_dir=out_dir)
    if get_confusion:
        get_save_confusion_matrix(video_targets, video_preds, f'{subset}_cm_video_{run_name}.csv', metric_res_dir=out_dir)


def main():
    res_dir = args.res_dir
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, 'classification_reports'), exist_ok=True)
    if res_dir is not None:
        all_runs = os.listdir(res_dir)
        # so out-names can be ordered by sorted run names
        all_runs = sorted([run for run in all_runs if os.path.isdir(os.path.join(res_dir, run))])
    else:
        all_runs = args.run_paths
    for i, run_name in enumerate(all_runs):
        get_metrics_for_run(res_dir, run_name, out_dir, get_clf_report=args.cr, get_confusion=args.cm)
        if args.train:
            get_metrics_for_run(res_dir, run_name, out_dir, subset='train', get_clf_report=args.cr)


if __name__ == "__main__":
    args = parser.parse_args()
    main()


