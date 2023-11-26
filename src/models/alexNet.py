import torch
import torch.nn as nn

# Implementação dos Layers =====================================================================
def alex_conv_layer1(in_ch, out_ch, kernel, stride, padding):
    layer = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = kernel, stride = 2)
    )
    return layer

def alex_conv_layer2(in_ch, out_ch, kernel, stride, padding):
    layer = nn.Sequential(
        nn.Con2d(in_ch, out_ch, kernel, stride, padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )
    return layer

def alex_fc_layer1(in_ch, out_ch):
    layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_ch, out_ch),
        nn.ReLU()
    )
    return layer

def alex_fc_layer2(in_ch, out_ch):
    layer = nn.Sequential(
        nn.Linear(in_ch, out_ch),
    )
    return layer

# Implementação da Rede AlexNet ================================================================
class MY_AlexNet(nn.Module):
    def __init__(self, n_classes):
        super(MY_AlexNet, self).__init__()

        self.layer1 = alex_conv_layer1(3, 96, 11, 4, 0)
        self.layer2 = alex_conv_layer1(96, 256, 5, 1, 2)
        self.layer3 = alex_conv_layer2(256, 384, 3, 1, 1)
        self.layer4 = alex_conv_layer2(384, 384, 3, 1, 1)
        self.layer5 = alex_conv_layer1(384, 256, 3, 1, 1)

        self.layer6 = alex_fc_layer1(9216, 4096)
        self.layer7 = alex_fc_layer1(4096, 4096)

        self.layer8 = alex_fc_layer2(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        alexNet_features = self.layer5(out)
        out = alexNet_features.view(out.size(0), -1)

        out = self.layer6(out)
        out = self.layer7(out)

        out = self.layer8(out)

        return alexNet_features, out
