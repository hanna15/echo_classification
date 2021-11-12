import torch
from torchvision import models
import os


def get_resnet3d_50(num_classes=2, pretrained=True):
    download_model_path = os.path.expanduser('~/.cache/torch/hub/facebookresearch_pytorchvideo_main')
    model = torch.hub.load(download_model_path, 'slow_r50', source='local', pretrained=pretrained)
    in_channels = 1  # Grayscale
    model.blocks[0].conv = torch.nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                           bias=False)
    num_ftrs = model.blocks[5].proj.in_features
    model.blocks[5].proj = torch.nn.Linear(num_ftrs, num_classes)
    return model


def get_resnet3d_18(num_classes=2, model_type='r2plus1d_18', pretrained=True):
    """

    :param num_classes: 2 or 3
    :param model_type: One of r2plus1d_18, mc3_18, r3d_18
    :param pretrained: True or False
    :return:
    """
    model = models.video.__dict__[model_type](pretrained=pretrained)
    in_channels = 1
    model.stem[0] = torch.nn.Conv3d(in_channels, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                    bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # model.fc.bias.data[0] = 55.6
    return model
