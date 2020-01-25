import torch.nn as nn
import torchvision.models as models
import torch

class feature_extractor(nn.Module):

    def __init__(self):

        super(feature_extractor, self).__init__()

        resnet101 = models.resnet101(pretrained=True)
        modules = list(resnet101.children())[:-1]
        self.net = nn.Sequential(*modules)

        self.ft_size = 2048

    def forward(self, x):
        features = self.net(x)
        features = features.view(-1, self.ft_size)
        return features

class class_classifier(nn.Module):
    def __init__(self, num_classes = 345):
        super(class_classifier, self).__init__()

        self.num_classes = num_classes

        self.main = nn.Sequential(

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            #nn.Dropout(),

            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(1024, num_classes)
        )

    def forward(self, x):

        x = self.main(x)

        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Linear(2048, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            nn.Dropout(),

            nn.Linear(64, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        x = self.main(x)

        return x


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)