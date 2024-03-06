import torch.nn as nn


class CustomResNet(nn.Module):
    def __init__(self, name, in_channels=0, num_classes=0):
        super(CustomResNet, self).__init__()
        self.name = name
        self.num_classes = num_classes

        # Define convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.GroupNorm(32, 64)
        self.tanh1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, 128)
        self.tanh2 = nn.ReLU()

        # Define residual blocks
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 128),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 512),
        )

        # Define final classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh2(x)

        # Residual blocks
        residual = x
        x = self.res1(x)
        x += residual
        x = self.res2(x)
        x = self.res3(x)

        # Final classification layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CNN(nn.Module):
    def __init__(self, name, in_channels=0, num_classes=0):
        super(CNN, self).__init__()
        self.name = name

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(1, 1)),
            nn.GroupNorm(16, 16),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1)),
            nn.GroupNorm(16, 16),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
