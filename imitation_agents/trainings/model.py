import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ActionModel(nn.Module):
    def __init__(self, throttle_max = 0.75, steer_max = 1.0):
        super().__init__()

        self.throttle_max = throttle_max
        self.steer_max = steer_max

        # front RGB part import ResNet-50
        self.front_rgb_backbone = torchvision.models.resnet50(pretrained=True)

        # remove last layer of front RGB of ResNet-50
        self.front_rgb_backbone.fc = nn.Linear(2048, 512, bias=True)

        # encoder for fused inputs
        self.fused_encoder = nn.Linear(3, 128, bias=True)

        # throttle-brake network
        self.accel_brake_network = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # steering network
        self.steer_network = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def normalize_rgb(self, x):
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def forward(self, fronts, fused_input):
        # pre-trained ResNet backbone
        front_rgb_features = torch.relu(self.front_rgb_backbone(fronts))
        
        # fuse velocity and relative far waypoints
        fused_features = torch.relu(self.fused_encoder(fused_input.float()))

        # concatenate rgb and fused features
        mid_features = torch.cat((front_rgb_features, fused_features), dim=1)

        accel_brake = self.accel_brake_network(mid_features)
        steer = self.steer_network(mid_features)

        return accel_brake, steer
