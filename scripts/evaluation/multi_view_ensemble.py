from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.evaluation.metrics import read_results, Metrics, get_metric_dict
import os
import numpy as np
import random
from statistics import multimode
"""
This script performs majority vote (ensemble) on models from different views.
"""

parser = ArgumentParser(
    description='Analyse results from results files',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('base_dir', help='Path to the base directory where model files are stored')
parser.add_argument('--res_files', help='Path to the results file of kapap view', nargs='+')
parser.add_argument('--views', help='Path to the results file of kapap view', nargs='+')
parser.add_argument('--disagree_method', default='conf', choices=['conf', 'random', 'max'],
                    help='What method to use to decide on a prediction when all models disagree')
parser.add_argument('--no_folds', type=int, default=10, help='Number of folds')
parser.add_argument('--verbose', action='store_true',
                    help='set this flag to have more print statements')

SEED = 0
np.random.seed(SEED)
random.seed(SEED)


def majority_vote(pred_views, conf_views, method='random'):
    """
    :param pred_views: List of prediction for current subject, for all 3 views in order [kapap, kaap, cv]
    :param conf_views: List of confidence for current subject, for all 3 views in order [kapap, kaap, cv]
    :param method: Method to use to select a prediction when all models disagree. Choices: random, conf, max
                   random: Choose random prediction,
                   conf: Choose prediction associated to most confident model,
                   max: Choose the higher prediction (i.e. more severe)
    :return: The majority vote prediction and associated confidence for curr video, as well as number of models agreeing
    """
    pred_views = [np.nan if pred is None else pred for pred in pred_views]  # None to np.nan
    conf_views = [np.nan if pred is None else pred for pred in conf_views]  # None to np.nan
    mv_pred = multimode(pred_views)  # majority vote pred is set to most common class(es) predictions
    mv_conf = np.nanmean(conf_views)  # for now, just set conf to mean of ALL (later change)
    num_views = len(pred_views)
    no_unique_preds = len(set(pred_views))
    num_models_agree = num_views if no_unique_preds == 1 else -1  # -1 is general
    # MV gives a tie: In case of 3 models, only happens when all disagree
    if len(mv_pred) > 1:
        if no_unique_preds == no_unique_preds:  # All models disagree
            num_models_agree = 0
            if method == 'random':
                best_idx = np.random.randint(num_views)
                while np.isnan(pred_views[best_idx]):
                    best_idx = np.random.randint(num_views)
            elif method == 'conf':
                best_idx = np.nanargmax(conf_views)  # max confidence
            elif method == 'max':
                best_idx = np.nanargmax(pred_views)  # max pred. value
            else:
                print('Select method as one of: [random, conf]')
                exit()
            mv_pred = pred_views[best_idx]
            mv_conf = conf_views[best_idx]
        else:  # Else 2 or more models agree on one label, 2 or more models on another label (only if > 3 views)
            # In this case, just select higher prediction leave the confidence to the mean TODO: More sophisticated
            num_models_agree = num_views - len(mv_pred)
            mv_pred = np.nanmax(mv_pred)
    else:
        mv_pred = mv_pred[0]  # Only single prediction that is most common
    return mv_pred, mv_conf, num_models_agree


def main():
    args = parser.parse_args()
    no_folds = args.no_folds
    res_files = args.res_files  # [args.res_file_kapap, args.res_file_cv, args.res_file_la]
    res_paths = [os.path.join(args.base_dir, res_file) for res_file in res_files]
    views = args.views  # ['kapap', 'cv', 'la']

    all_res = {key: [] for key in ['Video bACC', 'Video F1 (micro)', 'Video CI']}
    cnt_all_disagree = 0
    for fold in range(0, no_folds):
        subj_pred_all_views = {}  # {'818': {'kapap':(pred, ci), 'kaap': (pred, ci)'}}
        subj_targ_all_views = {}  # {'818': 2, '128': 0, ... }
        joint_subj_preds = []
        joint_subj_targets = []
        joint_subj_conf = []
        for res_path, view in zip(res_paths, views):
            fold_path = sorted(os.listdir(res_path))[fold]
            fold_dir = os.path.join(res_path, fold_path)
            fold_preds, fold_probs, fold_targets, fold_samples, fold_outs = read_results(fold_dir)
            m = Metrics(fold_targets, fold_samples, model_outputs=fold_outs, preds=fold_preds,
                        binary=False, tb=False)
            subj_targets, subj_preds, _, subj_confs, subj_ids, subj_outs = m.get_subject_lists(raw_outputs=True)
            for subj_id, subj_t, subj_p, subj_ci, subj_out in zip(subj_ids, subj_targets, subj_preds, subj_confs,
                                                                  subj_outs):
                # Match video-ids from different views
                if subj_id in subj_pred_all_views:
                    subj_pred_all_views[subj_id][view] = (subj_p, subj_ci, subj_out)
                else:
                    subj_pred_all_views[subj_id] = {}
                    subj_pred_all_views[subj_id][view] = (subj_p, subj_ci, subj_out)
                if subj_id not in subj_targ_all_views:
                    subj_targ_all_views[subj_id] = subj_t

        for key in subj_pred_all_views.keys():
            # Initialise as None, will be filled with correct info if exists
            'Not all views available for all videos'
            preds_all_views = []
            conf_all_views = []
            for view in views:
                if view in subj_pred_all_views[key]:
                    pred, conf, _ = subj_pred_all_views[key][view]
                else:
                    pred = conf = None
                preds_all_views.append(pred)
                conf_all_views.append(conf)

            # Calculate majority vote for the current model
            target = subj_targ_all_views[key]
            joint_pred, joint_ci, num_views_agree = majority_vote(preds_all_views, conf_all_views, method='conf')
            if num_views_agree == 0:
                cnt_all_disagree += 1
            if args.verbose and num_views_agree == 3 and joint_pred != target:
                print(f'All models wrong for: subj id {key}')
                print(f't={target}, p={joint_pred}, all_preds: {preds_all_views}, all confs: {conf_all_views}')

            joint_subj_preds.append(joint_pred)
            joint_subj_conf.append(joint_ci)
            joint_subj_targets.append(subj_targ_all_views[key])  # target is always the same

        subj_res = get_metric_dict(joint_subj_targets, joint_subj_preds, probs=None, binary=False, subset='val',
                                   prefix='Video ', tb=False, conf=joint_subj_conf)
        for metric, val in subj_res.items():
            all_res[metric].append(val)

    if args.verbose:
        print('no. videos where all disagree (aggregated over all folds)', cnt_all_disagree)

    for metric, metric_values in all_res.items():
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        metric_str = f'{mean:.2f} (std: {std:.2f})'
        print(metric, ':', metric_str)


if __name__ == '__main__':
    main()
