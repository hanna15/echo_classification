import torch
from torch import nn
from torchvision import models
from echo_ph.models.conv_lstm import ConvLSTM
from echo_ph.models.non_local import NLBlockND, MapBasedAtt
import os


def get_resnet3d_50(num_classes=2, pretrained=True):
    download_model_path = os.path.expanduser('~/.cache/torch/hub/facebookresearch_pytorchvideo_main')
    model = torch.hub.load(download_model_path, 'slow_r50', source='local', pretrained=pretrained)
    in_channels = 1  # Grayscale
    model.blocks[0].conv = torch.nn.Conv3d(in_channels, model.blocks[0].conv.out_channels, kernel_size=(1, 7, 7),
                                           stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    num_ftrs = model.blocks[5].proj.in_features
    model.blocks[5].proj = torch.nn.Linear(num_ftrs, num_classes)
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_resnet3d_18(num_classes=2, model_type='r2plus1d_18', pretrained=True):
    """

    :param num_classes: 2 or 3
    :param model_type: One of r2plus1d_18, mc3_18, r3d_18
    :param pretrained: True or False
    :return:
    """
    model = models.video.__dict__[model_type](pretrained=pretrained)
    in_channels = 1
    model.stem[0] = torch.nn.Conv3d(in_channels, model.stem[0].out_channels, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                    padding=(0, 3, 3), bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


class DCNN3D_ConvLSTM(nn.Module):
    def __init__(self):
        super(DCNN3D_ConvLSTM, self).__init__()
        model = models.video.__dict__['r3d_18'](pretrained=True)
        in_channels = 1
        model.stem[0] = torch.nn.Conv3d(in_channels, model.stem[0].out_channels, kernel_size=(1, 7, 7),
                                        stride=(1, 2, 2),
                                        padding=(0, 3, 3), bias=False)
        self.cnn_3d = nn.Sequential(*[model.stem, model.layer1, model.layer2])
        self.conv_lstm = ConvLSTM(input_dim=3, hidden_dim=[64, 64, 128], kernel_size=(3, 3),
                                  num_layers=3, batch_first=True, bias=True, return_all_layers=False)

    def forward(self, x):
        x = self.cnn_3d(x)
        x = self.conv_lstm(x)  # Input: A tensor of size B, T, C, H, W
        return x


class Res3DAttention(nn.Module):
    def __init__(self, num_classes=2, ch=1, w=112, h=112, t=6, att_type='self'):
        super(Res3DAttention, self).__init__()
        model = models.video.__dict__['r3d_18'](pretrained=True)
        in_channels = 1
        model.stem[0] = torch.nn.Conv3d(in_channels, model.stem[0].out_channels, kernel_size=(1, 7, 7),
                                        stride=(1, 2, 2),
                                        padding=(0, 3, 3), bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        self.base_model = model
        if att_type == 'self':
            self.att_block = NLBlockND(in_channels=ch * w * h, dimension=1)
        else:  # 'map-based attention'
            self.att_block = MapBasedAtt(in_channels=ch * w * h, time_steps=t)

    def forward(self, x):
        batch_size, ch, clip_len, w, h = x.shape
        x = x.reshape((batch_size, ch * w * h, clip_len))  # reshape to batch_size, w*h, clip-length
        x, att = self.att_block(x)
        x = x.reshape((batch_size, ch, clip_len, w, h))  # reshape back to 'normal'
        x = self.base_model(x)  # Input: A tensor of size B, T, C, H, W
        return x, att