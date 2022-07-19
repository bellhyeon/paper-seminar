from model.modeling import Model
from torch import nn


@Model.register("alexnet")
class AlexNet(Model):
    """
    AlexNet Pytorch Implementation.
    It follows `ImageNet Classification with Deep Convolutional Neural Networks`.

    Structure: 5 convolutional layers + 3 fully-connected layers
    """

    def __init__(self, **kwargs):
        super(AlexNet, self).__init__()
        self.num_features = kwargs["num_features"]
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4
            ),  # (bs, 3, 227, 227) → (bs, 96, 55, 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(
                kernel_size=3, stride=2
            ),  # (bs, 96, 55, 55) → (bs, 96, 27, 27)
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),  # (bs, 96, 27, 27) → (bs, 256, 27, 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(
                kernel_size=3, stride=2
            ),  # (bs, 256, 27, 27) → (bs, 256, 13, 13)
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),  # (bs, 256, 13, 13) → (bs, 384, 13, 13)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),  # (bs, 384, 13, 13) → (bs, 384, 13, 13)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (bs, 384, 13, 13), (bs, 256, 13, 13)
            nn.MaxPool2d(
                kernel_size=3, stride=2
            ),  # (bs, 256, 13, 13) → (bs, 256, 6, 6)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),  # same as x.view(-1, 256 * 6 * 6)
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=self.num_features),
        )

    def init_weights(self):
        for layer_num, layer in enumerate(self.conv):
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01),
                nn.init.zeros_(layer.bias)
                if (
                    layer_num == 4 or layer_num == 10 or layer_num == 12
                ):  # second, fourth, fifth conv layers
                    nn.init.ones_(layer.bias)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01),
                nn.init.ones_(layer.bias)

    def forward(self, inp):
        conv_result = self.conv(inp)
        output = self.fc(conv_result)
        return output
