import torch
from torch import cuda, device
from torch import nn, optim, no_grad
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import wandb
from data.echo_dataset import EchoDataset

"""
This script trains a basic pre-trained resnet-50 and performs image classification on the first frame of each 
newborn echocardiography video (KAPAP view). 
"""

parser = ArgumentParser(
    description='Train a Machine Learning model for classifying newborn echocardiography',
    formatter_class=ArgumentDefaultsHelpFormatter)
# Data parameters
parser.add_argument('--videos_dir', default='/Users/hragnarsd/Documents/masters/videos/Heart_Echo',
                    help='Path to the directory containing the videos')
parser.add_argument('--cache_dir', default='~/.heart_echo',
                    help='Path to the directory containing the cached and processed numpy videos')
parser.add_argument('--scaling_factor', default=0.5, help='How much to scale (down) the videos, as a ratio of original '
                                                          'size. Also determines the cache sub-folder')
parser.add_argument('--train_file_list_path', default='train_samples.npy',
                    help='Path to a file containing list of samples to use for training')
parser.add_argument('--valid_file_list_path', default='test_samples.npy',
                    help='Path to a file containing list of samples to use for validation')
parser.add_argument('--num_workers', type=int, default=4, help='The number of workers for loading data')

# Training parameters
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
parser.add_argument('--max_epochs', type=int, default=200, help='Max number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--decay_factor', type=float, default=0.0, help='Decay lr by this factor for decay on plateau')
parser.add_argument('--decay_patience', type=float, default=1000,
                    help='Number of epochs to decay lr for decay on plateau')
parser.add_argument('--min_lr', type=float, default=0.0, help='Min learning rate for reducing lr.')
parser.add_argument('--cooldown', type=float, default=0, help='cool-down for reducing lr on plateau')


def run_batch(batch, model, criterion):
    """
    Run a single batch
    :param batch: The data for this batch
    :param model: The seq2seq model
    :param criterion: The criterion for the loss. Set to None during evaluation (no training).
    :return: The required metrics for this batch, as well as the predictions and targets
    """
    dev = device('cuda' if cuda.is_available() and not args.load_model else 'cpu')
    data = batch["frame"].to(dev)  # batch_size, num_channels, w, h
    labels = batch["label"].to(dev)
    outputs = model(data)
    loss = criterion(outputs, labels)
    return loss, outputs, labels


def train(model, train_loader, valid_loader, data_len, valid_len, weights=None):
    # Initialize weights & biases logging
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
        wandb.log({
            "valid loss": epoch_valid_loss / valid_len,
            "train loss": epoch_loss / data_len
        })
        scheduler.step(epoch_valid_loss / valid_len)  # Update learning rate scheduler


def get_resnet():
    model = models.resnet50(pretrained=True)
    in_channels = 1
    # Change the input layer to take Grayscale image, instead of RGB images (set in_channels as 1)
    # original definition of the first layer on the ResNet class
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Change the output layer to output 3 classes instead of 1000 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    return model


def main():
    # Model
    model = get_resnet()

    # Data
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(size=(128, 128),
                                                      interpolation=InterpolationMode.BICUBIC),
                                    transforms.ToTensor()])

    train_dataset = EchoDataset(videos_dir=args.videos_dir, cache_dir=args.cache_dir,
                                scaling_factor=args.scaling_factor,
                                file_list_path=args.train_file_list_path,
                                transform=transform, procs=3)  # Todo: Have procs be a parameters
    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float)
    valid_dataset = EchoDataset(videos_dir=args.videos_dir, cache_dir=args.cache_dir,
                                scaling_factor=args.scaling_factor,
                                file_list_path=args.valid_file_list_path,
                                transform=transform)
    # For the data loader, if only use 1 worker, set it to 0, so data is loaded on the main process
    num_workers = (0 if args.num_workers == 1 else args.num_workers)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=num_workers)
    train(model, train_loader, valid_loader, len(train_dataset), len(valid_dataset), weights=class_weights)


if __name__ == "__main__":
    args = parser.parse_args()
    main()
