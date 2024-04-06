import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)

        return x

class DropoutBlock(nn.Module):
    def __init__(self, in_units: int, out_units: int, rate: float):
        super(DropoutBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_units, out_units),
            nn.BatchNorm1d(out_units),
            nn.ReLU(),
            nn.Dropout(rate)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class Model(nn.Module):
    def __init__(self, n_action: int, num_hidden: int, num_resblock:int,
                 rate:float, row:int, col: int, device: str):
        super(Model, self).__init__()

        # Bottom layer
        self.initial_block = nn.Sequential(
            nn.Conv2d(4, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        ).to(device)

        self.res_blocks = nn.Sequential(
            *[ResNetBlock(num_hidden) for _ in range(num_resblock)]
        ).to(device)

        self.dropout_model = nn.Sequential(
            DropoutBlock(num_hidden*row*col, 200, rate),
            DropoutBlock(200, 100, rate)
        )

        self.model = nn.Sequential(
            self.initial_block,
            self.res_blocks,
            nn.Flatten(),
            self.dropout_model
        )

        self.policy_head = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_action),
        ).to(device)

        self.value_head = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Tanh()
        ).to(device)

        self.to(device)

        self.device = device

        # Losses
        # Mean Square Error for minimizing the difference between estimated value and target value
        self.mse_loss = nn.MSELoss()

        # Cross entropy loss to evaluate the correct policy as compared to target policy
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        value = self.value_head(x)
        policy = self.policy_head(x)

        return value, policy

    # Perform the loss calculation
    def get_loss(self, pred_val, pred_policy, true_val, true_policy):
        val_loss = self.mse_loss(pred_val, true_val)
        policy_loss = self.ce_loss(pred_policy, true_policy)

        final_loss = val_loss + policy_loss
        return {
            'total_loss': final_loss,
            'value_loss': val_loss,
            'policy_loss': policy_loss
        }