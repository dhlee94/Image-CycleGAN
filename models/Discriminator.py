import torch.nn as nn
import torch
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        #Change first channels
        hidden_outchannel = vgg19_model.features[2].weight.shape[0]
        vgg19_model.features[0] = nn.Conv2d(in_channels=in_channel, out_channels=hidden_outchannel, kernel_size=3, stride=1, padding=1)
        #Change avgpool
        vgg19_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #Change class
        hidden_inchannel = vgg19_model.classifier[-4].weight.shape[0]
        vgg19_model.classifier[-1] = nn.Linear(in_features=hidden_inchannel, out_features=out_channel)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children()))

    def forward(self, img):
        return self.feature_extractor(img)
