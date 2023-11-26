import torch
import torch.nn as nn

# Implementação baseada no repistório abaixo:
# https://github.com/msyim/VGG16/blob/master/VGG16.py

# Implementação dos Layers =====================================================================
def vgg_conv_layer(in_ch, out_ch, kernel, padding):
    layer = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )
    return layer

def vgg_fc_layer(in_ch, out_ch):
    layer = nn.Sequential(
        nn.Linear(in_ch, out_ch),
        nn.BatchNorm1d(out_ch),
        nn.ReLU()
    )
    return layer

# Implementação dos Blocos de Camadas ==========================================================
def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = []

    for i in range(len(in_list)):
        layers.append(vgg_conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]))
    layers += [nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]

    return nn.Sequential(*layers)

# Implementação da Rede VGG16 ==================================================================
class MY_VGG16(nn.Module):
    def __init__(self, n_classes):
        super(MY_VGG16, self).__init__()

        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        self.layer6 = vgg_fc_layer(7*7*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out