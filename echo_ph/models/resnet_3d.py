import torch


def get_resnet3d(num_classes=2, pretrained=True):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrained)
    in_channels = 1  # Grayscale
    model.blocks[0].conv = torch.nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                           bias=False)
    num_ftrs = model.blocks[5].proj.in_features
    model.blocks[5].proj = torch.nn.Linear(num_ftrs, num_classes)
    return model
