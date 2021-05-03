from torch.nn import Module, Sequential, Linear, Softmax
from lib_resnet import ResNet_50


class Classifier(Module):
    def __init__(self, in_chans, n_class, img_size):
        super().__init__()

        self.encoder = ResNet_50(img_size, in_channel=in_chans)
        self.flatten_size = 512

        self.linear_prob = Sequential(
            Linear(self.flatten_size, n_class),
            Softmax(dim=-1),
        )

    def forward(self, x):
        # x: (N, C, H, W)
        feat = self.encoder(x)
        output = self.linear_prob(feat)

        return output

    def device(self):
        return next(self.parameters()).device


if __name__ == '__main__':
    pass
