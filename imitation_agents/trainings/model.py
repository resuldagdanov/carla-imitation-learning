import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


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


class OffsetModel(nn.Module):
    def __init__(self):
        super().__init__()

        # front RGB part import ResNet-50
        self.front_rgb_backbone = torchvision.models.resnet50(pretrained=True)

        # remove last layer of front RGB of ResNet-50
        self.front_rgb_backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU()
        )

        # encoder for fused inputs
        self.waypoint_fuser = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU()
        )

        # encoder part will be freezed during RL training
        self.mlp_encoder_network = nn.Sequential(
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # output networks -> will be unfreezed in RL training and pre-trained again in RL part
        self.brake_classifier_out = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.waypoint_offset_out = nn.Sequential(
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
        front_rgb_features = self.front_rgb_backbone(fronts)
        
        # fuse velocity and relative far waypoints
        fused_features = self.waypoint_fuser(fused_input.float())

        # concatenate rgb and fused features
        mid_features = torch.cat((front_rgb_features, fused_features), dim=1)

        # state space of RL agent
        features_out = self.mlp_encoder_network(mid_features)

        dnn_brake = self.brake_classifier_out(features_out)
        offset_amount = self.waypoint_offset_out(features_out)

        return dnn_brake, offset_amount
