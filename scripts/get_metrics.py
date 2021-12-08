import numpy as np
import os
import pandas as pd
import torch
import csv
from sklearn.metrics import roc_auc_score, classification_report, f1_score, roc_curve, balanced_accuracy_score
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
parser.add_argument('--train',  action='store_true', help='Set this flag to save also classification report per run')
parser.add_argument('--only_plot',  action='store_true', help='Set this flag to only plot ROC_AUC')
parser.add_argument('--plot_title',  type=str, default=None, nargs='+', help='title of ROC_AUC plot, if not default')

metrics = ['Frame ROC_AUC', 'Frame bAcc', 'Video ROC_AUC', 'Video bAcc', 'Video F1 (macro)', 'Video F1, pos',
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


def get_metrics_for_fold(fold_targets, fold_preds, fold_probs, fold_samples):
    """
    Get / collect metrics corresponding to a single fold
    :param fold_targets: Ground truth labels
    :param fold_preds: Model predicted labels
    :param fold_probs: Output probabilities of fold
    :param fold_samples: Sample names
    :return: results dictionary, video_wise_targets, video_wise_probs
    """

    # frame_roc_auc = roc_auc_score(fold_targets, fold_preds)
    frame_roc_auc = roc_auc_score(fold_targets, fold_probs)
    frame_b_acc = balanced_accuracy_score(fold_targets, fold_preds),
    # Get scores per video
    res_per_video = {}  # is format is {'vid_id': target, [predicted frames]}
    i = 0
    for targ, pred, sample in zip(fold_targets, fold_preds, fold_samples):
        vid_id = sample.split('_')[0]
        prob = np.nan if fold_probs is None else fold_probs[i]
        if vid_id in res_per_video:
            res_per_video[vid_id][1].append(pred)  # append all predictions for the given video
            res_per_video[vid_id][2].append(prob)
        else:
            res_per_video[vid_id] = (targ, [pred], [prob])  # first initialise it

        i += 1
    video_targets = []
    video_preds = []
    video_probs = []
    model_probs = []
    video_confidence_interval = []
    # Get a single prediction per video & get combined soft-maxed probs
    for res in res_per_video.values():
        ratio_pred_1 = np.sum(res[1]) / len(res[1])
        ratio_pred_0 = 1 - ratio_pred_1
        pred = 1 if ratio_pred_1 >= 0.5 else 0  # Change to a single pred value per video
        video_confidence_interval.append(ratio_pred_1 if pred == 1 else ratio_pred_0)
        video_probs.append(ratio_pred_1)
        video_preds.append(pred)
        video_targets.append(res[0])
        # model_probs.extend(res[2])  # save frame model (softmax) prob
        model_probs.append(np.nanmean(res[2]))

    res = {'Frame ROC_AUC': frame_roc_auc,
           'Frame bACC': frame_b_acc,
           #'Video ROC_AUC': roc_auc_score(video_targets, video_preds),
           'Video ROC_AUC': roc_auc_score(video_targets, model_probs),
           'Video bACC': balanced_accuracy_score(video_targets, video_preds),
           'Video F1 (macro)': f1_score(video_targets, video_preds, average='macro'),
           'Video F1, pos': f1_score(video_targets, video_preds, average='binary'),
           'Video F1, neg': f1_score(video_targets, video_preds, pos_label=0, average='binary'),
           'Video CI':  np.mean(video_confidence_interval)}
    return res, video_targets, video_probs, model_probs


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


def get_metrics_for_run(res_base_dir, run_name, out_dir, col, subset='val', get_clf_report=False, first=False,
                        out_name=None):
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
    metric_dict = {key: [] for key in metrics}
    targets = []
    preds = []
    vid_targets = []
    vid_probs = []
    avg_softm_probs = []
    epochs = []
    res_path = os.path.join(res_base_dir, run_name) if res_base_dir is not None else run_name
    # start with first fold, ignore .DS_store and other non-dir files
    fold_paths = [os.path.join(res_path, fold_path) for fold_path in sorted(os.listdir(res_path)) if
                  os.path.isdir(os.path.join(res_path, fold_path))]
    for fold_dir in fold_paths:
        epoch = int(fold_dir.rsplit('_e', 1)[-1])
        epochs.append(epoch)
        # fold_dir = os.path.join(res_path, fold_dir)
        fold_preds, fold_probs, fold_targets, fold_samples = read_results(fold_dir, subset)
        if fold_preds is None:
            print(f'failed for model {os.path.basename(fold_dir)}')
            continue
        results, vid_targ, vid_prob, avg_prob = get_metrics_for_fold(fold_targets, fold_preds, fold_probs, fold_samples)
        for metric, val in results.items():
            metric_dict[metric].append(val)
        vid_probs.extend(vid_prob)
        vid_targets.extend(vid_targ)
        preds.extend(fold_preds)
        targets.extend(fold_targets)
        avg_softm_probs.extend(avg_prob)
    # Save Results
    if get_clf_report:  # Classification report on a frame-level
        get_save_classification_report(targets, preds, f'{subset}_report_{run_name}.csv',
                                       metric_res_dir=out_dir, epochs=epochs)

    if first:  # Plot random baseline, only 1x
        p_fpr, p_tpr, _ = roc_curve(targets, [0 for _ in range(len(targets))], pos_label=1)
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue', label='random')

    # Get ROC_AUC plot on a video-level, with thresholds referring to probability of frames
    run_label = out_name if out_name is not None else run_name[-25:]
    preds_to_plot = vid_probs if np.isnan(avg_softm_probs[0]) else avg_softm_probs
    targets_to_plot = vid_targets  # if np.isnan(avg_softm_probs[0]) else targets
    # fpr1, tpr1, thresh1 = roc_curve(vid_targets, preds_to_plot, pos_label=1)
    fpr1, tpr1, thresh1 = roc_curve(targets_to_plot, preds_to_plot, pos_label=1, drop_intermediate=False)
    # plt.plot(fpr1, tpr1, color=col, label=run_label, marker='.')
    plt.plot(fpr1, tpr1, color=col, label=run_label)

    ret = []
    for metric_values in metric_dict.values():
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        metric_str = f'{mean:.2f} (std: {std:.2f})'
        ret.append(metric_str)
    return ret


def main():
    if args.cr:
        print("Also save classification reports")
    res_dir = args.res_dir
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, 'classification_reports'), exist_ok=True)
    if res_dir is not None:
        all_runs = os.listdir(res_dir)
        # so out-names can be ordered by sorted run names
        all_runs = sorted([run for run in all_runs if os.path.isdir(os.path.join(res_dir, run))])
    else:
        all_runs = args.run_paths
    no_runs = len(all_runs)
    val_data = [[] for _ in range(no_runs)]  # list of lists, for each run
    train_data = [[] for _ in range(no_runs)]  # list of lists, for each run
    colorMap = plt.get_cmap('jet', no_runs)
    for i, run_name in enumerate(all_runs):
        col = colorMap(i/no_runs)
        out_name = None if args.out_names is None else args.out_names[i]
        res = get_metrics_for_run(res_dir, run_name, out_dir, col, get_clf_report=args.cr, first=(i == 0),
                                  out_name=out_name)
        val_data[i] = res
        if args.train:
            res_train = get_metrics_for_run(res_dir, run_name, out_dir, col, subset='train', get_clf_report=args.cr,
                                            out_name=out_name)
            train_data[i] = res_train
    if args.out_names:
        df_names = args.out_names
    else:
        df_names = [os.path.basename(run) for run in all_runs]
    if not args.only_plot:
        df = pd.DataFrame(val_data, index=df_names, columns=metrics)
        if args.train:
            df_train = pd.DataFrame(train_data, index=df_names, columns=metrics)
            df = pd.concat([df, df_train], keys=['val', 'train'], axis=1)
        df.to_csv(os.path.join(out_dir, 'summary.csv'), float_format='%.2f')

    # finalise roc_auc curve
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    title = ' '.join(args.plot_title) if args.plot_title is not None else ''
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(out_dir, 'val_roc_auc_curve'))


if __name__ == "__main__":
    args = parser.parse_args()
    main()


