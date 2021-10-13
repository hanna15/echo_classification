import torch
from torch import cuda, device
from torch import nn, optim, no_grad
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import wandb
from echo_ph.data.echo_dataset import EchoDataset
from utils.transforms import HistEq, RandomNoise, Augment, Normalize, ConvertToTensor, AugmentSimpleIntensityOnly

"""
This script trains a basic pre-trained resnet-50 and performs image classification on the first frame of each 
newborn echocardiography video (KAPAP view). 
"""

parser = ArgumentParser(
    description='Train a Machine Learning model for classifying newborn echocardiography',
    formatter_class=ArgumentDefaultsHelpFormatter)
# Paths
parser.add_argument('--videos_dir', default=None,
                    help='Path to the directory containing the raw videos - if work on raw videos')
parser.add_argument('--cache_dir', default=None,
                    help='Path to the directory containing the cached and processed numpy videos - if work on those')
parser.add_argument('--train_file_list_path', default='train_samples.npy',
                    help='Path to a file containing list of samples to use for training')
parser.add_argument('--valid_file_list_path', default='test_samples.npy',
                    help='Path to a file containing list of samples to use for validation')
parser.add_argument('--label_path', default='labels3.pkl',
                    help='Path to a file containing labels of all training and test data')
# Data parameters
parser.add_argument('--scaling_factor', default=0.5, help='How much to scale (down) the videos, as a ratio of original '
                                                          'size. Also determines the cache sub-folder')
parser.add_argument('--num_workers', type=int, default=3, help='The number of workers for loading data')
parser.add_argument('--intensity_augments', action='store_true', help='set this flag to apply intensity augmentations '
                                                                      'to training data')
parser.add_argument('--class_balance_per_epoch', action='store_true', help='set this flag to use sampler, to have equal '
                                                                           'number of samples of each class in each epoch')

# Training parameters
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
parser.add_argument('--max_epochs', type=int, default=200, help='Max number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--decay_factor', type=float, default=0.0, help='Decay lr by this factor for decay on plateau')
parser.add_argument('--decay_patience', type=float, default=1000,
                    help='Number of epochs to decay lr for decay on plateau')
parser.add_argument('--min_lr', type=float, default=0.0, help='Min learning rate for reducing lr.')
parser.add_argument('--cooldown', type=float, default=0, help='cool-down for reducing lr on plateau')
parser.add_argument('--weight_loss', action='store_true', help='set this flag to weight loss, according'
                                                                         'to class imbalance')
# General parameters
parser.add_argument('--debug', action='store_true', help='set this flag when debugging, to not connect to wandb, etc')


def run_batch(batch, model, criterion):
    """
    Run a single batch
    :param batch: The data for this batch
    :param model: The seq2seq model
    :param criterion: The criterion for the loss. Set to None during evaluation (no training).
    :return: The required metrics for this batch, as well as the predictions and targets
    """
    dev = device('cuda' if cuda.is_available() and not args.load_model else 'cpu')
    data = batch["frames"].to(dev)  # batch_size, num_channels, w, h
    labels = batch["label"].to(dev)
    outputs = model(data)
    loss = criterion(outputs, labels)
    return loss, outputs, labels


def train(model, train_loader, valid_loader, data_len, valid_len, weights=None):
    # Initialize weights & biases logging
    if not args.debug:
        wandb.init(project='echo_classification', entity='hragnarsd', config={})
        wandb.config.update(args)

    # Set training loss, optimizer and training parameters
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if weights is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_factor, patience=args.decay_patience,
                                                     min_lr=args.min_lr, cooldown=args.cooldown)

    print("Start training on", data_len, "training samples, and", valid_len, "validation samples")

    for epoch in range(args.max_epochs):
        epoch_loss = 0
        epoch_valid_loss = 0

        # TRAIN
        model.train()
        if not args.debug:
            wandb.watch(model)
        for train_batch in train_loader:
            loss, pred, true = run_batch(train_batch, model, criterion)
            # print('* train * loss:', loss.item(), 'pred', pred.argmax(), 'true', true)
            epoch_loss += loss.item() * args.batch_size
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # VALIDATE
        with no_grad():
            model.eval()
            for valid_batch in valid_loader:
                valid_loss, valid_pred, valid_true = run_batch(valid_batch, model, criterion)
                # print('* valid * loss:', valid_loss.item(), 'pred', valid_pred.argmax(), 'true', valid_true)
                epoch_valid_loss += valid_loss.item() * args.batch_size

        print('epoch:', epoch)
        print('train_loss:', epoch_loss / data_len)
        print('valid loss:', epoch_valid_loss / valid_len)

        # Todo: Create a metric dictionary that can be updated with more metrics.
        if not args.debug:
            wandb.log({
                "valid loss": epoch_valid_loss / valid_len,
                "train loss": epoch_loss / data_len
            })
        scheduler.step(epoch_valid_loss / valid_len)  # Update learning rate scheduler


def get_resnet(num_classes=3):
    model = models.resnet50(pretrained=True)
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
    # Data
    # First resize, then normalize (!)
    base_transforms = [transforms.ToPILImage(),
                        transforms.Resize(size=(128, 128), interpolation=InterpolationMode.BICUBIC),
                        transforms.ToTensor(), Normalize()]
    transform_list_train = base_transforms
    if args.intensity_augments:
        # 50 % of samples will have random intensity augments (4 different augments, each applied with 30% prob)
        intensity_aug = transforms.RandomApply([AugmentSimpleIntensityOnly(orig_img_scale=0.5, size=128)], 0.5)
        transform_list_train.append(intensity_aug)
    transforms_train = transforms.Compose(transform_list_train)
    transforms_valid = transforms.Compose(base_transforms)
    # TODO: Try to have normalization and hist_eq and resize on all data, but try to have randomNoise() and AugmentSimpleIntensityOnly only on some data (add more data with it).
    # transform = transforms.Compose(
    #     [
    #         #VideoSubsample(num_video_frames) if num_video_frames is not None else Identity(),
    #         # HistEq(),
    #         ConvertToTensor(),
    #         Normalize(),
    #         #CropToCorners(orig_img_scale=dataset_orig_img_scale) if crop_to_corner else Identity(),
    #         #ShapeEqualization(resize, orig_img_scale=dataset_orig_img_scale) if shape_equalization else Identity(),
    #         #             GaussianSmoothing(),
    #         #Resize(resize, return_pid=(with_pid or augment)) if not shape_equalization else Identity(),
    #         transforms.Resize(size=(128, 128),
    #                           interpolation=InterpolationMode.BICUBIC),
    #         # Augment(orig_img_scale=0.5, size=128),
    #         # ==> AugmentSimpleIntensityOnly(orig_img_scale=0.5, size=128), => Try also with this (!)
    #         # RandomNoise(),
    #         #RandomMask(resize=resize,
    #         #           orig_img_scale=dataset_orig_img_scale) if mask and not shape_equalization else Identity(),
    #         #MinMask(resize=resize,
    #         #        orig_img_scale=dataset_orig_img_scale) if min_mask and not shape_equalization else Identity(),
    #     ]
    # )

    train_dataset = EchoDataset(videos_dir=args.videos_dir, cache_dir=args.cache_dir,
                                file_list_path=args.train_file_list_path, label_file_path=args.label_path,
                                transform=transforms_train, scaling_factor=args.scaling_factor, procs=args.num_workers)
    if args.weight_loss:
        class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float)
    else:
        class_weights = None
    valid_dataset = EchoDataset(videos_dir=args.videos_dir, cache_dir=args.cache_dir,
                                file_list_path=args.valid_file_list_path, label_file_path=args.label_path,
                                transform=transforms_valid, scaling_factor=args.scaling_factor,
                                procs=args.num_workers)

    # For the data loader, if only use 1 worker, set it to 0, so data is loaded on the main process
    num_workers = (0 if args.num_workers == 1 else args.num_workers)

    if args.class_balance_per_epoch:
        sampler = WeightedRandomSampler(train_dataset.example_weights, len(train_dataset.targets))
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=num_workers,
                                  sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=num_workers)

    # Model
    model = get_resnet(num_classes=len(train_dataset.labels))
    train(model, train_loader, valid_loader, len(train_dataset), len(valid_dataset), weights=class_weights)


if __name__ == "__main__":
    args = parser.parse_args()
    main()
