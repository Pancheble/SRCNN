import torch
from torch import nn

class SRCNN(nn.Module):
    def __init__(self, kernel_list, filters_list, num_channels=3):
        super(SRCNN, self).__init__()

        f1, f2, f3 = kernel_list
        n1, n2, n3 = filters_list

        self.conv1 = nn.Conv2d(num_channels, n1, kernel_size=f1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2)
        self.conv3 = nn.Conv2d(n2, num_channels, kernel_size=f3)
        self.relu = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)