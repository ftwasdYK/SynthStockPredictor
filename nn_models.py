from torch import nn
from torch.nn import functional as F
import torch
from pathlib import Path


__all__ = ['TsNet']

class TsNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.extractor = FeatureExctractor(num_classes=num_classes)
        self.flat = nn.Flatten()
        self.fc = nn.LazyLinear(num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.extractor(x.unsqueeze(1))
        x = self.fc(self.flat(x))
        return self.sig(x)

    @staticmethod
    def load_checkpoint(model:nn.Module, checkpoint_path:Path):
        checkpoint = torch.load(checkpoint_path)
        return model.load_state_dict(checkpoint['model_state_dict'])

class FeatureExctractor(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batchnorm = nn.LazyBatchNorm1d(num_features=16)
    
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = F.relu(self.batchnorm(x))
        return x
        