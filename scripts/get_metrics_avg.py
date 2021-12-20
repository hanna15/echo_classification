import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from echo_ph.evaluation.metrics import Metrics, get_save_classification_report, get_save_confusion_matrix
from statistics import multimode

parser = ArgumentParser(
    description='Get metrics',
    formatter_class=ArgumentDefaultsHelpFormatter)
# Paths, file name, model names, etc
parser.add_argument('--run_paths', type=str, default=None, nargs='+',
                    help='Set path of directory containing results for all desired runs to be averaged.')
parser.add_argument('--weights', type=float, default=None, nargs='+',
                    help='Set weights for weighted average - same order as res_dirs.')
parser.add_argument('--out_dir', type=str, default='metric_results')
parser.add_argument('--out_name', type=str, default='metric_results')
parser.add_argument('--cr',  action='store_true', help='Set this flag to save also classification report per run')
parser.add_argument('--cm',  action='store_true', help='Set this flag to also save confusion matrix per run')
parser.add_argument('--train',  action='store_true', help='Set this flag to save also classification report per run')
parser.add_argument('--only_plot',  action='store_true', help='Set this flag to only plot ROC_AUC')
parser.add_argument('--plot_title',  type=str, default=None, nargs='+', help='title of ROC_AUC plot, if not default')
parser.add_argument('--multi_class', action='store_true', help='Set this flag if not binary classification')


def get_metrics_for_fold(fold_targets, fold_preds, fold_probs, fold_samples):
    """
    Get / collect metrics corresponding to a single fold
    :param fold_targets: Ground truth labels
    :param fold_preds: Model predicted labels
    :param fold_probs: Output probabilities of fold
    :param fold_samples: Sample names
    :return: results dictionary, video_wise_targets, video_wise_probs
    """
    binary = True if not args.multi_class else False
    metrics = Metrics(fold_targets, fold_samples, preds=fold_preds, sm_probs=fold_probs, binary=binary, tb=False)
    all_metrics = metrics.get_per_sample_scores()  # first get sample metrics only
    subject_metrics = metrics.get_per_subject_scores()  # then get subject metrics
    all_metrics.update(subject_metrics)  # finally update sample metrics dict with subject metrics, to get all metrics
    vid_targ, vid_pred, vid_avg_prob, vid_conf, vid_ids = metrics.get_subject_lists()
    all_metrics.update({'Video CI':  np.mean(vid_conf)})
    return all_metrics, vid_targ, vid_avg_prob, vid_pred, vid_ids


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
        soft_m = np.asarray(sm(torch.tensor(outs)))  # get soft-maxed prob corresponding to class 1
        # probs = soft_m[:, 1]
        probs = soft_m
    else:
        preds = outs
        probs = None
    return preds, probs, targets, samples


def get_metrics(all_runs, out_dir, subset='val', get_clf_report=False, get_confusion=False):
    metric_dict = {key: [] for key in metric_list}
    targets = []
    preds = []
    vid_targets = []
    vid_preds = []
    vid_ids = []
    avg_softm_probs = []
    # start with first fold, ignore .DS_store and other non-dir files
    no_folds = 1
    all_runs_all_folds = np.asarray([sorted(os.listdir(run)) for run in all_runs])
    for i in range(no_folds):
        fold_i_runs = all_runs_all_folds[:, i]
        fold_probs = []
        for run_no, fold_i_run_name in enumerate(fold_i_runs):
            fold_i_run = os.path.join(all_runs[run_no], fold_i_run_name)
            model_fold_preds, model_fold_probs, model_fold_targets, model_fold_samples = read_results(fold_i_run, subset)
            fold_probs.append(model_fold_probs)
        print('la')
        fold_probs = np.average(np.asarray(fold_probs), axis=0, weights=args.weights)
        fold_targets = model_fold_targets   # just the last one
        fold_samples = model_fold_samples  # just the last one
        fold_preds = np.argmax(fold_probs, axis=-1)  # Works for multi-class classification => Get class with highest avg. prob
        fold_probs = fold_probs[:, 1]  # To get prob for class-1, in binary classification
        results, vid_targ, avg_prob, vid_pred, video_ids = get_metrics_for_fold(fold_targets, fold_preds, fold_probs,
                                                                                fold_samples)
        for metric, val in results.items():
            metric_dict[metric].append(val)

        vid_targets.extend(vid_targ)
        vid_preds.extend(vid_pred)
        vid_ids.extend(video_ids)

        preds.extend(fold_preds)
        targets.extend(fold_targets)
        if avg_prob is not None:
            avg_softm_probs.extend(avg_prob)

    if get_clf_report or get_confusion:
        all_unique_video_res = {}
        for v_id, v_target, v_pred in zip(vid_ids, vid_targets, vid_preds):
            if v_id in all_unique_video_res:
                all_unique_video_res[v_id][1].append(v_pred)  # add current prediction
            else:
                all_unique_video_res[v_id] = [v_target, [v_pred]]  # (video target, list of video predictions)
        vid_preds = [max(multimode(all_unique_video_res[vid_id][1])) for vid_id in all_unique_video_res.keys()]
        vid_targets_unique = [v[0] for v in all_unique_video_res.values()]  # Single target for each unique video
        if get_clf_report:
            # Classification report on a frame-level
            get_save_classification_report(targets, preds, f'{subset}_report_{args.out_name}.csv',
                                           metric_res_dir=out_dir)
            # Classification report on a video-level
            get_save_classification_report(vid_targets_unique, vid_preds, f'{subset}_report_video_{args.out_name}.csv',
                                           metric_res_dir=out_dir)
        if get_confusion:
            get_save_confusion_matrix(targets, preds, f'{subset}_cm_{args.out_name}.csv', metric_res_dir=out_dir)
            get_save_confusion_matrix(vid_targets_unique, vid_preds, f'{subset}_cm_video_{args.out_name}.csv',
                                      metric_res_dir=out_dir)

    # ROC_AUC Plotting
    if not args.multi_class:
        # Plot random baseline
        p_fpr, p_tpr, _ = roc_curve(targets, [0 for _ in range(len(targets))], pos_label=1)
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue', label='random')
        # Get ROC_AUC plot on a video-level, with thresholds referring to probability of frame
        run_label = args.out_name
        fpr1, tpr1, thresh1 = roc_curve(vid_targets, avg_softm_probs, pos_label=1, drop_intermediate=False)
        plt.plot(fpr1, tpr1, label=run_label)

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
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, 'classification_reports'), exist_ok=True)
    all_runs = args.run_paths

    val_data = get_metrics(all_runs, out_dir, get_clf_report=args.cr, get_confusion=args.cm)
    if args.train:
        train_data = get_metrics(all_runs, out_dir, subset='train', get_clf_report=args.cr,
                                 get_confusion=args.cm)

    if not args.only_plot:
        df = pd.DataFrame(val_data, index=args.out_name, columns=metric_list)
        if args.train:
            df_train = pd.DataFrame(train_data, index=args.out_name, columns=metric_list)
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
    if args.multi_class:
        metric_list = ['Frame bACC', 'Video bACC', 'Video F1 (micro)', 'Video CI']
    else:
        metric_list = ['Frame ROC_AUC', 'Frame bACC', 'Video ROC_AUC', 'Video bACC', 'Video F1 (micro)',
                       'Video F1, pos', 'Video F1, neg', 'Video CI']
    main()


