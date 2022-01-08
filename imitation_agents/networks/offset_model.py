import torch
import torch.nn as nn
import torchvision
import numpy as np


class OffsetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def inference(self, image, fused_inputs):
        # convert width height channel to channel width height
        image = np.array(image.transpose((2, 0, 1)), np.float32)
        
        # BGRA to BGR
        image = image[:3, :, :]
        # BGR to RGB
        image = image[::-1, :, :]
        
        # normalize to 0 - 1
        image = image / 255
        
        # to tensor and unsquueze for batch dimension
        image_torch = torch.from_numpy(image.copy()).unsqueeze(0)

        # normalize input image
        image_torch = self.normalize_rgb(image_torch)

        # fused inputs to torch
        fused_inputs = np.array(fused_inputs, np.float32)
        fused_inputs_torch = torch.from_numpy(fused_inputs.copy()).unsqueeze(0)

        # inference
        with torch.no_grad():
            dnn_brake, offset_amount = self.forward(image_torch, fused_inputs_torch)
        
        # torch control to CPU numpy array
        dnn_brake = dnn_brake.squeeze(0).cpu().detach().numpy()[0]
        offset_amount = offset_amount.squeeze(0).cpu().detach().numpy()[0]

        brake = np.where(dnn_brake < 0.5, 0.0, 1.0)
        offset = offset_amount * 3

        return brake, offset
