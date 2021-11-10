import torch
import os


def get_resnet3d(num_classes=2, pretrained=True):
    download_model_path = os.path.expanduser('~/.cache/torch/hub/facebookresearch_pytorchvideo_main')
    model = torch.hub.load(download_model_path, 'slow_r50', source='local', pretrained=pretrained)
    in_channels = 1  # Grayscale
    model.blocks[0].conv = torch.nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                           bias=False)
    num_ftrs = model.blocks[5].proj.in_features
    model.blocks[5].proj = torch.nn.Linear(num_ftrs, num_classes)
    return model
