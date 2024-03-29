import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet34

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, backbone=resnet34(pretrained=True), dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = backbone   #num_classes=dim, zero_init_residual=True
        # build a 3-layer projector
        prev_dim =self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Linear(prev_dim, dim, bias=False)
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True), # first layer
                self.encoder.fc,
                nn.BatchNorm1d(dim, affine=False)) # output layer
        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(inplace=True), # hidden layer
                nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z = self.encoder(x) # NxC
        p = self.predictor(z) # NxC

        return z, p