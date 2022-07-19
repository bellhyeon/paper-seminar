from model.modeling import Model
from torch import nn
from torchsummary import summary
import torch


@Model.register("googlenet")
class GoogLeNet(Model):
    """
    GoogLeNet (Inception_V1) Pytorch Implementation.
    It follows `Going Deeper with Convolutions`.

    Structure: 22 convolutional layers + 2 auxiliary classifier
    """

    def __init__(self, **kwargs):
        super(GoogLeNet, self).__init__()
        self.num_features = kwargs["num_features"]

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # (bs, 480, 28, 28) → (bs, 480, 14, 14)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, self.num_features)
        self.softmax = nn.Softmax(dim=-1)

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3,
            ),  # (bs, 3, 224, 224) → (bs, 64, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            ),  # (bs, 64, 112, 112) → (bs, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1), # (bs, 64, 56, 56) → (bs, 192, 56, 56),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # (bs, 192, 56, 56) → (bs, 192, 28, 28)
        )

        # inception block
        self.inception_3a = Inception_Block(192, 64, 96, 128, 16, 32, 32) # (bs, 192, 28, 28) → (bs, 256, 28, 28)
        self.inception_3b = Inception_Block(256, 128, 128, 192, 32, 96, 64) # (bs, 256, 28, 28), (bs, 480, 28, 28)
        self.inception_4a = Inception_Block(480, 192, 96, 208, 16, 48, 64) # (bs, 480, 14, 14) → (bs, 512, 14, 14)
        self.inception_4b = Inception_Block(512, 160, 112, 224, 24, 64, 64) # (bs, 512, 14, 14) → (bs, 512, 14, 14)
        self.inception_4c = Inception_Block(512, 128, 128, 256, 24, 64, 64) # (bs, 512, 14, 14) → (bs, 512, 14, 14)
        self.inception_4d = Inception_Block(512, 112, 144, 288, 32, 64, 64) # (bs, 512, 14, 14) → (bs, 528, 14, 14)
        self.inception_4e = Inception_Block(528, 256, 160, 320, 32, 128, 128) # (bs, 528, 14, 14) → (bs, 832, 14, 14)
        self.inception_5a = Inception_Block(832, 256, 160, 320, 32, 128, 128) # (bs, 832, 7, 7) → (bs, 832, 7, 7)
        self.inception_5b = Inception_Block(832, 384, 192, 384, 48, 128, 128) # (bs, 832, 7, 7) → (bs, 1024, 7, 7)



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        x = self.stem(inp)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.max_pool(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.gap(x)
        x = x.view([-1, x.shape[1]])

        x = self.dropout(x)
        x = self.fc(x)
        output = self.softmax(x)
        return output

class Inception_Block(nn.Module):
    def __init__(self, in_channels, _1x1, _3x3reduce, _3x3, _5x5reduce, _5x5, pool_proj):
        super(Inception_Block, self).__init__()
        self.in_channels = in_channels
        self._1x1 = _1x1
        self._3x3reduce = _3x3reduce
        self._3x3 = _3x3
        self._5x5reduce = _5x5reduce
        self._5x5 = _5x5
        self.pool_proj = pool_proj

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self._1x1, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self._3x3reduce, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self._3x3reduce, out_channels=self._3x3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self._5x5reduce, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self._5x5reduce, out_channels=self._5x5, kernel_size=5, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.pool_proj, kernel_size=1, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, inp):
        branch1_output = self.branch1(inp)
        branch2_output = self.branch2(inp)
        branch3_output = self.branch3(inp)
        branch4_output = self.branch4(inp)

        output = torch.cat([branch1_output, branch2_output, branch3_output, branch4_output], dim=1)

        return output
