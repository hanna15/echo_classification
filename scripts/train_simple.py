import numpy as np
import torch
import os
import math
from torch import cuda, device
from torch import nn, optim, no_grad
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import wandb
from echo_ph.data.echo_dataset import EchoDataset
from utils.transforms import Normalize, RandomResize, AugmentSimpleIntensityOnly
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
import warnings

"""
This script trains a basic pre-trained resnet-50 and performs image classification on the first frame of each 
newborn echocardiography video (KAPAP view). 
"""

parser = ArgumentParser(
    description='Train a Machine Learning model for classifying newborn echocardiography. Please make sure to have '
                'already generated label files, placed in project_root/label_files, and test/train index files, placed'
                'in project_root/index_files',
    formatter_class=ArgumentDefaultsHelpFormatter)
# Paths
parser.add_argument('--videos_dir', default=None,
                    help='Path to the directory containing the raw videos - if work on raw videos')
parser.add_argument('--cache_dir', default=None,
                    help='Path to the directory containing the cached and processed numpy videos - if work on those')
parser.add_argument('--label_type', default='2class', choices=['2class', '2class_drop_ambiguous', '3class'],
                    help='How many classes for the labels, and in some cases also variations of dropping ambiguous '
                         'labels. Will be used to fetch the correct label file and train and test index files')
# Data parameters
parser.add_argument('--scaling_factor', default=0.5, help='How much to scale (down) the videos, as a ratio of original '
                                                          'size. Also determines the cache sub-folder')
parser.add_argument('--num_workers', type=int, default=3, help='The number of workers for loading data')

# Training parameters
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--max_epochs', type=int, default=200, help='Max number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--decay_factor', type=float, default=0.0, help='Decay lr by this factor for decay on plateau')
parser.add_argument('--decay_patience', type=float, default=1000,
                    help='Number of epochs to decay lr for decay on plateau')
parser.add_argument('--min_lr', type=float, default=0.0, help='Min learning rate for reducing lr')
parser.add_argument('--cooldown', type=float, default=0, help='cool-down for reducing lr on plateau')
parser.add_argument('--pretrained', action='store_true', help='Set this flag to use pre-trained resnet')

# Class imbalance
parser.add_argument('-b', '--class_balance_per_epoch', action='store_true',
                    help='set this flag to have ca. equal no. samples of each class per epoch / oversampling')
parser.add_argument('--weight_loss', action='store_true',
                    help='set this flag to weight loss, according to class imbalance')
parser.add_argument('--augment', action='store_true',
                    help='set this flag to apply augmentation transformations to training data')
# General parameters
parser.add_argument('--debug', action='store_true', help='set this flag when debugging, to not connect to wandb, etc')
parser.add_argument('--visualise_frames', action='store_true', help='set this flag to visualise frames')
parser.add_argument('--log_freq', type=int, default=5,
                    help='How often to log to tensorboard and w&B, and save models. Save logs every log_freq th epoch, '
                         'but save models every (log_freq * 2) th epoch.')


def get_metrics(outputs, targets, prefix='', binary=False):
    out = outputs.cpu()
    tar = targets.cpu()
    _, preds = torch.max(out, dim=1)
    # Determine averaging strategies for f1-score
    if binary:
        avg = 'binary'
    else:
        avg = 'micro'  # For imbalanced multi-class, micro is better than macro
    metrics = {  # zero_div=0, sets f1 to 1 (corr), when all targets and preds are negative & NOT give warning.
                 # default is 'warn', which sets f1 to 0, and further raises a warning
                'f1' + '/' + prefix: f1_score(tar, preds, average=avg, zero_division=1),
                'accuracy' + '/' + prefix: accuracy_score(tar, preds),
                'b-accuracy' + '/' + prefix: balanced_accuracy_score(tar, preds)  # ,
                # prefix + 'roc_auc': roc_auc_score(tar, preds)
                # Todo: roc_auc is undefined if batch gt has only 1 class
                # TODO: => thus change all metrics to be calculated per-EPOCH (!), instead of per-batch
               }
    return metrics, preds


def run_batch(batch, model, criterion, binary=False, metric_prefix=''):
    """
    Run a single batch
    :param batch: The data for this batch
    :param model: The seq2seq model
    :param criterion: The criterion for the loss. Set to None during evaluation (no training).
    :param binary: Set to True if this is binary classification.
    :param metric_prefix: Set to a string to prefix each metric key in metric-dict, if desired.
    :return: The required metrics for this batch, as well as the predictions and targets
    """
    dev = device('cuda' if cuda.is_available() else 'cpu')
    input = batch["frames"].to(dev)  # batch_size, num_channels, w, h
    targets = batch["label"].to(dev)
    outputs = model(input)
    if binary:
        # Convert to one-hot encoding, and convert to float, bc the binary loss supports prob. labels (“soft” labels).
        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=2).float()
        loss = criterion(outputs, one_hot_targets)
    else:
        loss = criterion(outputs, targets)
    # return original targets (not one-hot)
    metrics, predictions = get_metrics(outputs, targets, metric_prefix, binary)
    return loss, predictions, targets, metrics


def get_run_name():
    run_name = 'lt_' + args.label_type + '.lr_' + str(args.lr) + '.batch_' + str(args.batch_size)
    if args.pretrained:
        run_name += '_pre.'
    if args.augment:
        run_name += '_aug.'
    if args.class_balance_per_epoch:
        run_name += '_bal.'
    if args.weight_loss:
        run_name += '_weight.'
    return run_name


def train(model, train_loader, valid_loader, data_len, valid_len, weights=None, binary=False):
    # Initialize weights & biases logging
    if not args.debug:
        # wandb.init(project='echo_classification', entity='hragnarsd', config={}, mode="offline", sync_tensorboard=True)
        # wandb.config.update(args)
        run_name = get_run_name()
        writer = SummaryWriter(log_dir=os.path.join('tb_runs', run_name))
    # Set training loss, optimizer and training parameters
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if binary:
        criterion = nn.BCEWithLogitsLoss(weight=weights)  # if weights is None, no weighting is performed
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)  # if weights is None, no weighting is performed
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_factor, patience=args.decay_patience,
                                                     min_lr=args.min_lr, cooldown=args.cooldown)

    valid_num_batches = math.ceil(valid_len / args.batch_size)
    num_batches = math.ceil(data_len / args.batch_size)
    print("Start training on", data_len, "training samples, and", valid_len, "validation samples")
    for epoch in range(args.max_epochs):
        epoch_loss = 0
        epoch_valid_loss = 0
        epoch_targets = []
        epoch_preds = []
        epoch_valid_targets = []
        epoch_valid_preds = []
        epoch_metrics = {'f1/train': 0, 'accuracy/train': 0, 'b-accuracy/train': 0}  #, 'roc_auc': 0}
        epoch_valid_metrics = {'f1/val': 0, 'accuracy/val': 0, 'b-accuracy/val': 0}  #, 'roc_auc': 0}

        # TRAIN
        model.train()
        # if not args.debug:
            # wandb.watch(model)
        for train_batch in train_loader:
            loss, pred, targets, metrics = run_batch(train_batch, model, criterion, binary=binary, metric_prefix='train')
            epoch_targets.extend(targets)
            epoch_preds.extend(pred)
            epoch_loss += loss.item() * args.batch_size
            for metric in metrics:
                epoch_metrics[metric] += metrics[metric]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # VALIDATE
        with no_grad():
            model.eval()
            for valid_batch in valid_loader:
                val_loss, val_pred, val_targets, val_metrics = run_batch(valid_batch, model, criterion, binary=binary,
                                                                         metric_prefix='val')
                epoch_valid_targets.extend(val_targets)
                epoch_valid_preds.extend(val_pred)
                epoch_valid_loss += val_loss.item() * args.batch_size
                for metric in val_metrics:
                    epoch_valid_metrics[metric] += val_metrics[metric]

        scheduler.step(epoch_valid_loss / valid_len)  # Update learning rate scheduler

        if epoch % args.log_freq == 0:  # log every 10th epoch
            print('*** epoch:', epoch, '***')
            print('train_loss:', epoch_loss / data_len)
            print('valid loss:', epoch_valid_loss / valid_len)

            for metric in epoch_metrics:
                epoch_metrics[metric] /= num_batches
                print(metric, ":", epoch_metrics[metric])
            for metric in epoch_valid_metrics:
                epoch_valid_metrics[metric] /= valid_num_batches
                print(metric, ":", epoch_valid_metrics[metric])

            # Todo: Create a metric dictionary that can be updated with more metrics.
            if not args.debug:  # log and save results
                log_dict = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr'],  # Actual learning rate (changes because of scheduler)
                    "loss/valid": epoch_valid_loss / valid_len,
                    "loss/train": epoch_loss / data_len
                }
                log_dict.update(epoch_metrics)
                log_dict.update(epoch_valid_metrics)
                # wandb.log(log_dict)
                for metric_key in log_dict:
                    step = int(epoch / args.log_freq)
                    writer.add_scalar(metric_key, log_dict[metric_key], step)

            if epoch % (2 * args.log_freq) == 0:  # save model checkpoints at 2x lower resolution than saving logs
                torch.save(model.state_dict(), os.path.join('models', run_name + '.pt'))

                # writer.add_hparams(
                #     {"init_lr": args.lr, "bsize": args.batch_size, "augment": args.augment, "pretrained": args.pretrained},
                #     log_dict
                # )

            else:
                target_lst = [t.item() for t in epoch_targets]
                vals, cnts = np.unique(target_lst, return_counts=True)
                print('epoch target distribution')
                for val, cnt in zip(vals, cnts):
                    print(val, ':', cnt)
    if not args.debug:
        writer.close()


def get_resnet(num_classes=3):
    model = models.resnet18(pretrained=args.pretrained)  # TODO: Later move  to resnet-50 ==> better to start small.
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
    if not args.debug:
        warnings.simplefilter("ignore")  # ignore warnings, so they don't fill output log files
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('start training on device', device)
    binary = False
    if args.label_type.startswith('2class'):
        binary = True
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    train_index_file_path = os.path.join('index_files', 'train_samples_' + args.label_type + '.npy')
    test_index_file_path = os.path.join('index_files', 'test_samples_' + args.label_type + '.npy')
    # Data
    # First resize, then normalize (!)
    base_transforms = [transforms.ToPILImage(), transforms.Resize(size=(128, 128),
                                                                  interpolation=InterpolationMode.BICUBIC),
                       transforms.ToTensor(), Normalize()]
    transform_list_train = base_transforms
    if args.augment:
        # In 65% cases, apply intensity augment. These have 3 transformations, each applied with 50% prob - so in 87.5%
        # of calling it, some augmentation is performed. Thus, total% of intensity augments is: 0.875 * 0.65 = 56.8 %
        intesity_aug = transforms.RandomApply([AugmentSimpleIntensityOnly()], 0.65)
        # In 50% cases perform random resizing (each time image is EITHER padded and made smaller or zoomed in)
        resize = transforms.RandomApply([RandomResize()], 0.5)
        transform_list_train.extend([intesity_aug, resize])
    transforms_train = transforms.Compose(transform_list_train)
    transforms_valid = transforms.Compose(base_transforms)

    train_dataset = EchoDataset(train_index_file_path, label_path, videos_dir=args.videos_dir, cache_dir=args.cache_dir,
                                transform=transforms_train, scaling_factor=args.scaling_factor, procs=args.num_workers,
                                visualise_frames=args.visualise_frames)
    if args.weight_loss:
        class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float)
    else:
        class_weights = None
    valid_dataset = EchoDataset(test_index_file_path, label_path, videos_dir=args.videos_dir, cache_dir=args.cache_dir,
                                transform=transforms_valid, scaling_factor=args.scaling_factor, procs=args.num_workers,
                                visualise_frames=args.visualise_frames)
    # For the data loader, if only use 1 worker, set it to 0, so data is loaded on the main process
    num_workers = (0 if args.num_workers == 1 else args.num_workers)

    if args.class_balance_per_epoch:
        sampler = WeightedRandomSampler(train_dataset.example_weights, train_dataset.num_samples)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=num_workers,
                                  sampler=sampler)  # Sampler is mutually exclusive with shuffle
    else:
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=num_workers)

    # Model
    model = get_resnet(num_classes=len(train_dataset.labels)).to(device)
    os.makedirs('models', exist_ok=True) # create model results dir, if not exists
    train(model, train_loader, valid_loader, len(train_dataset), len(valid_dataset), weights=class_weights,
          binary=binary)


if __name__ == "__main__":
    args = parser.parse_args()
    main()
