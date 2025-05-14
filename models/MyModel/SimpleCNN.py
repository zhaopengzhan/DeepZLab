from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim):
        super(SimpleCNN, self).__init__()
        nn.ModuleList()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_dim),

            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
