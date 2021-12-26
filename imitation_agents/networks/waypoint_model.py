import torch
import torch.nn as nn
import torchvision
import numpy as np


class WaypointModel(nn.Module):
    def __init__(self, throttle_max = 0.75, steer_max = 1.0):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        #image_torch = image_torch.to(self.device)

        # fused inputs to torch
        fused_inputs = np.array(fused_inputs, np.float32)
        fused_inputs_torch = torch.from_numpy(fused_inputs.copy()).unsqueeze(0) #.to(self.device)

        # inference
        with torch.no_grad():
            accel_brake, steer = self.forward(image_torch, fused_inputs_torch)
        
        # torch control to CPU numpy array
        accel_brake = accel_brake.squeeze(0).cpu().detach().numpy()
        steer = steer.squeeze(0).cpu().detach().numpy()

        return accel_brake, steer
