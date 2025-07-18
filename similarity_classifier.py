import torch.nn as nn
import torch

class SimilarityNetwork(nn.Module):
    def __init__(self):
        super(SimilarityNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        diff = torch.abs(x1 - x2)
        return self.fc(diff)