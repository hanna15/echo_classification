import numpy as np
import torch
import os
import math
from torch import cuda, device
from torch import nn, optim, no_grad
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import wandb
from echo_ph.data.echo_dataset import EchoDataset
from echo_ph.models.conv_nets import ConvNet, SimpleConvNet
from echo_ph.models.my_resnet import resnet_simpler
from echo_ph.data.ph_labels import long_label_type_to_short
from utils.transforms import get_augment_transforms, get_base_transforms
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
import warnings

"""
This script trains a basic pre-trained resnet-50 and performs image classification on the first frame of each 
newborn echocardiography video (KAPAP view). 
"""

parser = ArgumentParser(
    description='Train a Machine Learning model for classifying newborn echocardiography. Please make sure to have '
                'already generated label files, placed in project_root/label_files, and valid/train index files, placed'
                'in project_root/index_files',
    formatter_class=ArgumentDefaultsHelpFormatter)
# Paths, file name, model names, etc
parser.add_argument('--videos_dir', default=None,
                    help='Path to the directory containing the raw videos - if work on raw videos')
parser.add_argument('--cache_dir', default=None,
                    help='Path to the directory containing the cached and processed numpy videos - if work on those')
parser.add_argument('--label_type', default='2class', choices=['2class', '2class_drop_ambiguous', '3class'],
                    help='How many classes for the labels, and in some cases also variations of dropping ambiguous '
                         'labels. Will be used to fetch the correct label file and train and valid index files')
parser.add_argument('--k', default=None, type=int,
                    help='In case of k-fold cross-validation, set the current k (fold) for this training.'
                         'Will be used to fetch the relevant k-th train and valid index file')
parser.add_argument('--model_name', type=str, default=None,
                    help='Set the name of the model you want to load or train. If None, use the model name as assigned'
                         'by the function get_run_name(), using selected arguments, and optionally unique_run_id')
parser.add_argument('--run_id', type=str, default='',
                    help='Set a unique_run_id, to identify run if arguments alone are not enough to identify (e.g. when'
                         'running on same settings multiple times). Id will be pre-pended to the run name derived '
                         'from arguments. Default is empty string, i.e. only identify run with arguments.')
# Data parameters
parser.add_argument('--scaling_factor', default=0.25, help='How much to scale (down) the videos, as a ratio of original '
                                                          'size. Also determines the cache sub-folder')
parser.add_argument('--num_workers', type=int, default=4, help='The number of workers for loading data')
parser.add_argument('--max_p', type=float, default=90, help='Percentile for max expansion frames')
parser.add_argument('--augment', action='store_true',
                    help='set this flag to apply ALL augmentation transformations to training data')
parser.add_argument('--noise', action='store_true',
                    help='Apply random noise augmentation')
parser.add_argument('--intensity', action='store_true',
                    help='Apply random intensity augmentation')
parser.add_argument('--rand_resize', action='store_true',
                    help='Apply random resizing augmentation')
parser.add_argument('--rotate', action='store_true',
                    help='Apply random rotating augmentation')
parser.add_argument('--translate', action='store_true',
                    help='Apply random translating augmentation')
parser.add_argument('--hist_eq', action='store_true',
                    help='set this flag to apply histogram equalisation to training and validation data')
# Class imbalance
parser.add_argument('--class_balance_per_epoch', action='store_true',
                    help='set this flag to have ca. equal no. samples of each class per epoch / oversampling')
parser.add_argument('--weight_loss', action='store_true',
                    help='set this flag to weight loss, according to class imbalance')
# Training & models parameters
parser.add_argument('--load_model', action='store_true',
                    help='Set this flag to load an already trained model to predict only, instead of training it.'
                         'If args.model_name is set, load model from that path. Otherwise, get model name acc. to'
                         'function get_run_name(), and load the corresponding model')
parser.add_argument('--model', default='resnet', choices=['resnet', 'res_simple', 'conv', 'simple_conv'],
                    help='What model architecture to use.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout value for those model who use dropout')
parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw'], help='What optimizer to use.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--max_epochs', type=int, default=300, help='Max number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--decay_factor', type=float, default=0.0, help='Decay lr by this factor for decay on plateau')
parser.add_argument('--decay_patience', type=int, default=1000,
                    help='Number of epochs to decay lr for decay on plateau')
parser.add_argument('--min_lr', type=float, default=0.0, help='Min learning rate for reducing lr')
parser.add_argument('--cooldown', type=float, default=0, help='cool-down for reducing lr on plateau')
parser.add_argument('--early_stop', type=int, default=100,
                    help='Patience (in no. epochs) for early stopping due to no improvement of valid f1 score')
parser.add_argument('--pretrained', action='store_true', help='Set this flag to use pre-trained resnet')
parser.add_argument('--eval_metric', default='f1/valid', choices=['f1/valid', 'loss/valid', 'f1/train', 'loss/train'],
                    help='Set this the metric you want to use for early stopping')

# General parameters
parser.add_argument('--debug', action='store_true', help='set this flag when debugging, to not connect to wandb, etc')
parser.add_argument('--visualise_frames', action='store_true', help='set this flag to visualise frames')
parser.add_argument('--log_freq', type=int, default=2,
                    help='How often to log to tensorboard and w&B.')
parser.add_argument('--tb_dir', type=str, default='tb_runs_cv',
                    help='Tensorboard directory - where tensorboard logs are stored.')

MAX_NO_FOLDS = 5
BASE_RES_DIR = 'results'
BASE_MODEL_DIR = 'models'
TORCH_SEED = 0


def get_run_name():
    """
    Returns a 'semi'-unique name according to most important arguments.
    Can be used for model name and tb log names
    :return:
    """
    if args.run_id == '':
        run_id = ''
    else:
        run_id = args.run_id + '_'
    if args.k is None:
        k = ''
    else:
        k = '.k' + str(args.k)
    run_name = run_id + args.model + '_' + args.optimizer + '_lt_' + long_label_type_to_short[args.label_type]\
               + k + '.lr_' + str(args.lr) + '.batch_' + str(args.batch_size)
    if args.decay_factor > 0.0:
        run_name += str(args.decay_factor)  # only add to description if not default
    if args.decay_patience < 1000:
        run_name += str(args.decay_patience)  # only add to description if not default
    if args.pretrained:
        run_name += '_pre'
    if args.augment:
        run_name += '_aug'
    if args.class_balance_per_epoch:
        run_name += '_bal'
    if args.weight_loss:
        run_name += '_w'
    if args.hist_eq:
        run_name += '_hist'
    if args.noise:
        run_name += '_n'
    if args.intensity:
        run_name += '_i'
    if args.rand_resize:
        run_name += '_re'
    if args.rotate:
        run_name += '_rot'
    if args.rotate:
        run_name += '_t'
    if args.model != 'resnet':
        run_name += '_drop' + str(args.dropout)
    if args.max_p != 90:
        run_name += '_p' + str(args.max_p)
    return run_name


def get_metrics(outputs, targets, prefix='', binary=False):
    """
    Get metrics per batch
    :param outputs: Model outputs (before max) OR model predictions (after max) - as a tensor OR list
    :param targets: Targets / true label - as a tensor OR list
    :param prefix: What to prefix the metric with - set to 'train' or 'valid' or 'test'
    :param binary: Set to true if this is for binary classification
    :return: Dictionary containing the metrics and the model predictions (arg maxed outputs)
    """
    if torch.is_tensor(outputs):
        outputs = outputs.cpu()
        targets = targets.cpu()
    if np.shape(outputs) != np.shape(targets):
        preds = np.max(outputs, dim=1)
    else:
        preds = outputs

    if binary:
        avg = 'binary'
    else:
        avg = 'micro'  # For imbalanced multi-class, micro is better than macro
    metrics = {'f1' + '/' + prefix: f1_score(targets, preds, average=avg),
               'accuracy' + '/' + prefix: accuracy_score(targets, preds),
               'b-accuracy' + '/' + prefix: balanced_accuracy_score(targets, preds),
               'roc_auc' + '/' + prefix: roc_auc_score(targets, preds)
               }
    return metrics


def run_batch(batch, model, criterion=None, binary=False):
    """
    Run a single batch
    :param batch: The data for this batch
    :param model: The seq2seq model
    :param criterion: The criterion for the loss. Set to None during evaluation (no training).
    :param binary: Set to True if this is binary classification.
    :return: The required metrics for this batch, as well as the predictions and targets
    """
    dev = device('cuda' if cuda.is_available() else 'cpu')
    input = batch["frame"].to(dev)  # batch_size, num_channels, w, h
    targets = batch["label"].to(dev)
    sample_names = batch["sample_name"]
    outputs = model(input)
    # Get loss, if we are training
    if criterion:
        if binary:
            # Convert to one-hot encoding, and convert to float, bc the binary loss supports prob. labels (“soft” labels).
            one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=2).float()
            loss = criterion(outputs, one_hot_targets)
        else:
            loss = criterion(outputs, targets)
    else:  # when just evaluating, no loss
        loss = None
    return loss, outputs, targets, sample_names


def evaluate(model, model_name, train_loader, valid_loader, data_len, valid_len, binary=False):
    model_path = os.path.join(BASE_MODEL_DIR, model_name)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    valid_num_batches = math.ceil(valid_len / args.batch_size)
    num_batches = math.ceil(data_len / args.batch_size)
    epoch_metrics = {'f1/train': 0, 'accuracy/train': 0, 'b-accuracy/train': 0}  # , 'roc_auc': 0}
    epoch_valid_metrics = {'f1/val': 0, 'accuracy/val': 0, 'b-accuracy/val': 0}  # , 'roc_auc': 0
    epoch_targets = []
    epoch_preds = []
    epoch_valid_targets = []
    epoch_valid_preds = []

    with torch.no_grad():
        model.eval()
        for train_batch in train_loader:
            _, pred, targets, metrics = run_batch(train_batch, model, binary=binary, metric_prefix='train')
            epoch_targets.extend(targets)
            epoch_preds.extend(pred)
            for metric in metrics:
                epoch_metrics[metric] += metrics[metric]
        for valid_batch in valid_loader:
            _, val_pred, val_targets, val_metrics = run_batch(valid_batch, model, binary=binary, metric_prefix='valid')
            epoch_valid_targets.extend(val_targets)
            epoch_valid_preds.extend(val_pred)
            for metric in val_metrics:
                epoch_valid_metrics[metric] += val_metrics[metric]

    for metric in epoch_metrics:
        epoch_metrics[metric] /= num_batches
        print(metric, ":", epoch_metrics[metric])
    for metric in epoch_valid_metrics:
        epoch_valid_metrics[metric] /= valid_num_batches
        print(metric, ":", epoch_valid_metrics[metric])


def save_model_and_res(model, run_name, target_lst, pred_lst, val_target_lst, val_pred_lst, sample_names,
                       val_sample_names, epoch=None, k=None):
    """
    Save the given model, as well as the outputs, targets & metrics
    :param model: The model to save
    :param run_name: Descriptive name of this run / model
    :param target_lst: List of targets for training data
    :param pred_lst: List of predictions for training data
    :param sample_names: List of sample names for training data (sample + frame nr)
    :param val_target_lst: List of targets for validation data
    :param val_pred_lst: List of predictions for validation data
    :param val_sample_names: List of sample names for validation data (sample + frame nr)
    :param epoch: Current epoch for the given model
    :param k: If cross-validation is being used, this is the current fold
    """
    if epoch is None:
        base_name = run_name + '_final'
    else:  # Append epoch name to the model name
        base_name = run_name + '_e' + str(epoch)

    if k is None:  # If not k-fold cross validation, save results in base dirs
        res_dir = os.path.join(BASE_RES_DIR, base_name)
        model_dir = BASE_MODEL_DIR
    else:
        res_dir = os.path.join(BASE_RES_DIR, 'fold' + str(k), base_name)
        model_dir = os.path.join(BASE_MODEL_DIR, 'fold' + str(k))

    os.makedirs(res_dir, exist_ok=True)  # create sub-directory for this base model name

    model_file_name = base_name + '.pt'
    targ_file_name = 'targets_' + base_name + '.npy'
    pred_file_name = 'preds_' + base_name + '.npy'
    sample_file_names = 'samples_' + base_name + '.npy'

    torch.save(model.state_dict(), os.path.join(model_dir, model_file_name))
    np.save(os.path.join(res_dir, 'train_' + targ_file_name), target_lst)
    np.save(os.path.join(res_dir, 'train_' + pred_file_name), pred_lst)
    np.save(os.path.join(res_dir, 'train_' + sample_file_names), sample_names)
    np.save(os.path.join(res_dir, 'val_' + targ_file_name), val_target_lst)
    np.save(os.path.join(res_dir, 'val_' + pred_file_name), val_pred_lst)
    np.save(os.path.join(res_dir, 'val_' + sample_file_names), val_sample_names)


def train(model, train_loader, valid_loader, data_len, valid_len, tb_writer, run_name, optimizer, weights=None,
          binary=False, use_wandb=False):

    # Set training loss, optimizer and training parameters
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if binary:
        criterion = nn.BCEWithLogitsLoss(weight=weights)  # if weights is None, no weighting is performed
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)  # if weights is None, no weighting is performed
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_factor, patience=args.decay_patience,
                                                     min_lr=args.min_lr, cooldown=args.cooldown)
    best_early_stop = float("inf") if 'loss' in args.eval_metric else -1  # if loss metric, then minimize, else maximize
    num_val_fails = 0
    print("Start training on", data_len, "training samples, and", valid_len, "validation samples")
    for epoch in range(args.max_epochs):
        epoch_loss = epoch_valid_loss = 0
        epoch_targets = []
        epoch_preds = []
        epoch_samples = []
        epoch_valid_targets = []
        epoch_valid_preds = []
        epoch_valid_samples = []

        # TRAIN
        model.train()
        for train_batch in train_loader:
            loss, out, targets, sample_names = run_batch(train_batch, model, criterion, binary)
            epoch_samples.extend(sample_names)
            epoch_targets.extend(targets)
            epoch_preds.extend(torch.max(out, dim=1)[1])
            epoch_loss += loss.item() * args.batch_size
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # VALIDATE
        with no_grad():
            model.eval()
            for valid_batch in valid_loader:
                val_loss, val_out, val_targets, val_sample_names = run_batch(valid_batch, model, criterion, binary)
                epoch_valid_samples.extend(val_sample_names)
                epoch_valid_targets.extend(val_targets)
                epoch_valid_preds.extend(torch.max(val_out, dim=1)[1])
                epoch_valid_loss += val_loss.item() * args.batch_size

        scheduler.step(epoch_valid_loss / valid_len)  # Update learning rate scheduler

        if epoch % args.log_freq == 0:  # log every xth epoch
            target_lst = [t.item() for t in epoch_targets]
            pred_lst = [t.item() for t in epoch_preds]
            targ_lst_valid = [t.item() for t in epoch_valid_targets]
            pred_lst_valid = [t.item() for t in epoch_valid_preds]
            epoch_metrics = get_metrics(pred_lst, target_lst, prefix='train', binary=binary)
            epoch_valid_metrics = get_metrics(pred_lst_valid, targ_lst_valid, prefix='valid', binary=binary)
            print('*** epoch:', epoch, '***')
            print('train_loss:', epoch_loss / data_len)
            print('valid loss:', epoch_valid_loss / valid_len)

            for metric in epoch_metrics:
                print(metric, ":", epoch_metrics[metric])
            for metric in epoch_valid_metrics:
                print(metric, ":", epoch_valid_metrics[metric])

            if not args.debug:  # log and save results
                log_dict = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr'],  # Actual learning rate (changes because of scheduler)
                    "loss/valid": epoch_valid_loss / valid_len,
                    "loss/train": epoch_loss / data_len
                }
                log_dict.update(epoch_metrics)
                log_dict.update(epoch_valid_metrics)
                if use_wandb:
                    wandb.log(log_dict)
                for metric_key in log_dict:
                    step = int(epoch / args.log_freq)
                    tb_writer.add_scalar(metric_key, log_dict[metric_key], step)

                if args.early_stop:
                    if 'loss' in args.eval_metric:  # smaller value is better
                        better_res = log_dict[args.eval_metric] < best_early_stop
                    else:  # larger value is better
                        better_res = log_dict[args.eval_metric] > best_early_stop

                    if better_res:
                        best_early_stop = log_dict[args.eval_metric]
                        num_val_fails = 0
                        save_model_and_res(model, run_name, target_lst, pred_lst, targ_lst_valid, pred_lst_valid,
                                           epoch_samples, epoch_valid_samples, epoch=epoch, k=args.k)
                    else:
                        num_val_fails += 1

                    if num_val_fails >= args.early_stop:
                        print('== Early stop training after', num_val_fails,
                              'epochs without validation loss improvement')
                        break
            else:
                target_lst = [t.item() for t in epoch_targets]
                vals, cnts = np.unique(target_lst, return_counts=True)
                print('epoch target distribution')
                for val, cnt in zip(vals, cnts):
                    print(val, ':', cnt)

    if not args.debug:
        tb_writer.close()


def get_resnet(num_classes=3):
    model = models.resnet18(pretrained=args.pretrained)
    in_channels = 1
    # Change the input layer to take Grayscale image, instead of RGB images (set in_channels as 1)
    # original definition of the first layer on the ResNet class
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Change the output layer to output 3 classes instead of 1000 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def main():
    torch.manual_seed(TORCH_SEED)  # Fix a seed, to increase reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Will be training on device', device)
    run_name = get_run_name()
    use_wandb = False  # Set this to false for now as can't seem to use on cluster
    if not args.debug:
        warnings.simplefilter("ignore")  # Ignore warnings, so they don't fill output log files
        # Initialize  logging if not debug phase
        if use_wandb:
            wandb.init(project='echo_classification', entity='hragnarsd', config={}, mode="offline")
            wandb.config.update(args)
        tb_writer = SummaryWriter(log_dir=os.path.join(args.tb_dir, run_name))
    else:
        tb_writer = None

    binary = True if args.label_type.startswith('2class') else False
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    idx_file_end = '' if args.k is None else '_' + str(args.k)
    train_index_file_path = os.path.join('index_files', 'train_samples_' + args.label_type + idx_file_end + '.npy')
    valid_index_file_path = os.path.join('index_files', 'valid_samples_' + args.label_type + idx_file_end + '.npy')

    # Data & Transforms
    if args.augment and not args.load_model:  # All augmentations
        train_transforms = get_augment_transforms(hist_eq=args.hist_eq)  # all other default true
    else:
        individual_augments = [args.hist_eq, args.noise, args.intensity, args.rand_resize, args.rotate, args.translate]
        if any(individual_augments):  # Only some specific augmentations
            train_transforms = get_augment_transforms(individual_augments)
        else:  # No augmentation
            train_transforms = get_base_transforms(hist_eq=args.hist_eq)
    valid_transforms = get_base_transforms(hist_eq=args.hist_eq)

    train_dataset = EchoDataset(train_index_file_path, label_path, videos_dir=args.videos_dir,
                                cache_dir=args.cache_dir,
                                transform=train_transforms, scaling_factor=args.scaling_factor,
                                procs=args.num_workers, visualise_frames=args.visualise_frames,
                                percentile=args.max_p)
    if args.weight_loss:
        class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float).to(device)
    else:
        class_weights = None
    valid_dataset = EchoDataset(valid_index_file_path, label_path, videos_dir=args.videos_dir, cache_dir=args.cache_dir,
                                transform=valid_transforms, scaling_factor=args.scaling_factor, procs=args.num_workers,
                                visualise_frames=args.visualise_frames, percentile=args.max_p)
    # For the data loader, if only use 1 worker, set it to 0, so data is loaded on the main process
    num_workers = (0 if args.num_workers == 1 else args.num_workers)

    # Model & Optimizers
    if args.model == 'resnet':
        model = get_resnet(num_classes=len(train_dataset.labels)).to(device)
    elif args.model == 'res_simple':
        model = resnet_simpler(num_classes=len(train_dataset.labels), drop_prob=args.dropout).to(device)
    elif args.model == 'conv':
        model = ConvNet(num_classes=len(train_dataset.labels), dropout_val=args.dropout).to(device)
    else:
        model = SimpleConvNet(num_classes=len(train_dataset.labels)).to(device)
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)  # create model results dir, if not exists
    os.makedirs(BASE_RES_DIR, exist_ok=True)  # create results dir, if not exists
    for i in range(MAX_NO_FOLDS):
        os.makedirs(os.path.join(BASE_MODEL_DIR, 'fold' + str(i)), exist_ok=True)  # create model res dir for each fold
        os.makedirs(os.path.join(BASE_RES_DIR, 'fold' + str(i)), exist_ok=True)  # create res dir for each fold

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:  # optimizer = adamw
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)  # weight-decay provided by default
    if args.load_model:  # Create eval datasets (no shuffle) and evaluate model
        eval_loader_train = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=num_workers)
        eval_loader_valid = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=num_workers)
        model_name = run_name if args.model_name is None else args.model_name
        evaluate(model, model_name, eval_loader_train, eval_loader_valid)
    else:  # Create training datasets (with shuffling or sampler) and train
        if args.class_balance_per_epoch:
            sampler = WeightedRandomSampler(train_dataset.example_weights, train_dataset.num_samples)
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=num_workers,
                                      sampler=sampler)  # Sampler is mutually exclusive with shuffle
        else:
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=num_workers)

        train(model, train_loader, valid_loader, len(train_dataset), len(valid_dataset), tb_writer, run_name, optimizer,
              weights=class_weights, binary=binary, use_wandb=use_wandb)


if __name__ == "__main__":
    args = parser.parse_args()
    main()
