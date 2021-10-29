import torch
import numpy as np
import os
import pandas as pd
import csv
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report


def get_save_classification_report(targets, preds, file_name, metric_res_dir='results', epochs=None):
    report = classification_report(targets, preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['support'] = df['support'].astype(int)
    file_name = os.path.join(metric_res_dir, 'classification_reports', file_name)
    df.to_csv(file_name, float_format='%.3f')
    if epochs is not None: # Add also epochs info
        with open(file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(['epochs'] + epochs)
    print(classification_report(targets, preds))


def get_scores_for_fold(fold_targets, fold_preds, fold_samples):
    frame_roc_auc = roc_auc_score(fold_targets, fold_preds)

    # Get scores per video
    res_per_video = {}  # format is {'vid_id': target, [predicted frames]}
    for t, p, s in zip(fold_targets, fold_preds, fold_samples):
        vid_id = s.split('_')[0]
        if vid_id in res_per_video:
            res_per_video[vid_id][1].append(p)  # append all predictions for the given video
        else:
            res_per_video[vid_id] = (t, [])  # first initialise it
    fold_targets_per_video = []
    fold_preds_per_video = []
    # Get a single prediction per video
    for res in res_per_video.values():
        ratio_pred_1 = np.sum(res[1]) / len(res[1])
        pred = 1 if ratio_pred_1 >= 0.5 else 0  # Change to a single pred value per video
        fold_preds_per_video.append(pred)
        fold_targets_per_video.append(res[0])
    video_roc_auc = roc_auc_score(fold_targets_per_video, fold_preds_per_video)
    return frame_roc_auc, video_roc_auc


def get_all_results(res_base_dir='raw_results', metric_res_dir='results', classification_report=True):
    os.makedirs(os.path.join(metric_res_dir, 'classification_reports'), exist_ok=True)
    all_runs = os.listdir(res_base_dir)
    val_data = [[] for _ in range(len(all_runs))]  # list of lists, for each run
    train_data = [[] for _ in range(len(all_runs))]  # list of lists, for each run
    for i, run_name in enumerate(all_runs):
        val_preds = []
        val_targets = []
        train_preds = []
        train_targets = []
        val_metrics = {
            'Frame ROC_AUC': [],
            'Video ROC_AUC': [],
        }
        train_metrics = {
            'Frame ROC_AUC': [],
            'Video ROC_AUC': [],
        }
        epochs = []
        for fold_model_name in os.listdir(os.path.join(res_base_dir, run_name)):
            epoch = int(fold_model_name.rsplit('_e', 1)[-1])
            epochs.append(epoch)
            fold_dir = os.path.join(res_base_dir, run_name, fold_model_name)
            fold_val_preds = np.load(os.path.join(fold_dir, 'val_preds.npy'))
            fold_val_targets = np.load(os.path.join(fold_dir, 'val_targets.npy'))
            fold_val_samples = np.load(os.path.join(fold_dir, 'val_samples.npy'))
            fold_train_preds = np.load(os.path.join(fold_dir, 'train_preds.npy'))
            fold_train_targets = np.load(os.path.join(fold_dir, 'train_targets.npy'))
            fold_train_samples = np.load(os.path.join(fold_dir, 'train_samples.npy'))

            frame_roc_auc, video_roc_auc = get_scores_for_fold(fold_val_targets, fold_val_preds, fold_val_samples)
            val_metrics['Frame ROC_AUC'].append(frame_roc_auc)
            val_metrics['Video ROC_AUC'].append(video_roc_auc)
            val_preds.extend(fold_val_preds)
            val_targets.extend(fold_val_targets)

            frame_roc_auc, video_roc_auc = get_scores_for_fold(fold_train_targets, fold_train_preds, fold_train_samples)
            train_metrics['Frame ROC_AUC'].append(frame_roc_auc)
            train_metrics['Video ROC_AUC'].append(video_roc_auc)
            train_preds.extend(fold_train_preds)
            train_targets.extend(fold_train_targets)

        # Save Results
        if classification_report:
            get_save_classification_report(val_targets, val_preds, f'val_report_{run_name}.csv')
            get_save_classification_report(train_targets, train_preds, f'train_report_{run_name}.csv')

        for metric_values in val_metrics.values():
            mean = np.mean(metric_values)
            std = np.std(metric_values)
            metric_str = f'{mean:.3f} (std: {std:.3f})'
            val_data[i].append(metric_str)

        for metric_values in train_metrics.values():
            mean = np.mean(metric_values)
            std = np.std(metric_values)
            metric_str = f'{mean:.3f} (std: {std:.3f})'
            train_data[i].append(metric_str)

    df_val = pd.DataFrame(val_data, index=os.listdir(res_base_dir), columns=val_metrics.keys())
    df_train = pd.DataFrame(train_data, index=os.listdir(res_base_dir), columns=train_metrics.keys())
    df_final = pd.concat([df_val, df_train], keys=['val', 'train'], axis=1)
    df_final.to_csv(os.path.join(metric_res_dir, 'summary.csv'), float_format='%.3f')


get_all_results('raw_results')
