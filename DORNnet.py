# coding=utf-8
"""
@Author: jin
@Email:
"""
import os
import torch
import torch.nn as nn
import torchvision.models
import collections
import math


def weights_init(m):
    """
    模型权重参数初始化
    :param m:模型网络层层序号，layers_num
    :return:m.bias.data.zero_()，模型偏置归零
    """
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ResNet(nn.Module):
    """
    模型网络类声明
    """
    def __init__(self, in_channels=3, pretrained=True):
        """
        网络初始化
        :param in_channels: 输入图像通道数
        :param pretrained:预训练模式开启
        """
        super(ResNet, self).__init__()
        # dorn 模型基础模型设置为resnet101模型
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(101)](pretrained=pretrained)
        # pretrained_model = torchvision.models.__dict__['resnet{}'.format(18)](pretrained=pretrained)

        self.channel = in_channels
        # 输入size(3, 257, 353)
        # 图像数据的矩阵变换和激活,BatchNorm2d 归一化操作，ReLU 激活函数为分段激活函数，Conv2d 卷积，卷积核数
        # 构建顺序容器
        self.conv1 = nn.Sequential(collections.OrderedDict([
            ('conv1_1', nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)),  # (64, 129, 177)
            ('bn1_1', nn.BatchNorm2d(64)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),  # (64, 129, 177)
            ('bn_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),  # (128, 129, 177)
            ('bn1_3', nn.BatchNorm2d(128)),
            ('relu1_3', nn.ReLU(inplace=True))
        ]))

        self.bn1 = nn.BatchNorm2d(128)

        # print(pretrained_model._modules['layer1'][0].conv1)

        self.relu = pretrained_model._modules['relu']
        #构建池化层
        # 池化层矩阵变换
        self.maxpool = pretrained_model._modules['maxpool']  # (128, 65, 89)
        self.layer1 = pretrained_model._modules['layer1']
        self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.layer2 = pretrained_model._modules['layer2']

        self.layer3 = pretrained_model._modules['layer3']
        self.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3[0].downsample[0] = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.layer4 = pretrained_model._modules['layer4']
        self.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # 清理预训练显存
        del pretrained_model
        # 权重参数初始化
        weights_init(self.conv1)
        weights_init(self.bn1)
        weights_init(self.layer1[0].conv1)
        weights_init(self.layer1[0].downsample[0])
        weights_init(self.layer3[0].conv2)
        weights_init(self.layer3[0].downsample[0])
        weights_init(self.layer4[0].conv2)
        weights_init(self.layer4[0].downsample[0])

    def forward(self, x):
        """
        前向传播
        :param x: 自身前向传播函数 X(C,H,W) 维度矩阵封装
        :return:
        """
        # print(pretrained_model._modules)
        x = self.conv1(x)  # (128, 129, 177)
        x = self.bn1(x)
        x = self.relu(x)
        # print('conv1:', x.size())
        x = self.maxpool(x)  # (128, 65, 89)
        # print('pool:', x.size())
        x1 = self.layer1(x)  # (256, 65, 89)
        # print('layer1 size:', x1.size())
        x2 = self.layer2(x1)  # (512, 33, 45)
        # print('layer2 size:', x2.size())
        x3 = self.layer3(x2)  # (1024, 33, 45)
        # print('layer3 size:', x3.size())
        x4 = self.layer4(x3)  # (2048, 33, 45)
        # print('layer4 size:', x4.size())
        return x4


class FullImageEncoder(nn.Module):
    # 图像处理模块
    def __init__(self):
        """
        全连接接编码器
        """
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(8, stride=8, padding=(4, 2))  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(2048 * 6 * 5, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积
        self.upsample = nn.UpsamplingBilinear2d(size=(33, 45))  # KITTI 49X65 NYU 33X45

    def forward(self, x):
        """
        :param x: 图像维度矩阵
        :return:
        """
        # input size (b, 2048, 33, 45)
        x1 = self.global_pooling(x)  # (b, 2048, 5, 6)
        # print('# x1 size:', x1.size())
        # 全连接层加dropout层,防止模型过拟合
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * 6 * 5)
        x4 = self.relu(self.global_fc(x3))  # (b, 512)
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)  # (b, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)  # (b, 512, 1, 1)
        # print('x5:',x5.size())
        out = self.upsample(x5)  # (b, 512, 33, 45) 和COPY的效果一样
        return out


class SceneUnderstandingModule(nn.Module):
    # 场景理解处理模块
    def __init__(self):
        super(SceneUnderstandingModule, self).__init__()
        # 调用图像处理模块
        self.encoder = FullImageEncoder()
        # 定义多个aspp基础网络
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=18, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, 136, 1),  # KITTI 142 NYU 136 In paper, K = 80 is best, so use 160 is good!
            # nn.UpsamplingBilinear2d(scale_factor=8)
            nn.UpsamplingBilinear2d(size=(257, 353))
        )

    def forward(self, x):
        """
        :param x: 传输矩阵
        :return:
        """
        # x input_size (2048, 33, 45)
        # 编码器
        x1 = self.encoder(x)
        # 调用多个aspp网络
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)
        # 网络拼接
        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)  # (b, 2560, 33, 45)
        # print('cat x6 size:', x6.size())
        # 张量存储
        out = self.concat_process(x6)  # (b, 136, 257, 353)
        return out


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        # 前向传播
        # (b, 136, 257, 353)
        """
        :param x: N x H x W x C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, H, W = x.size()
        ord_num = C // 2
        # 模型浮点运算
        if torch.cuda.is_available():
            decode_label = torch.zeros((N, 1, H, W), dtype=torch.float32).cuda()
            ord_labels = torch.zeros((N, C // 2, H, W), dtype=torch.float32).cuda()
        else:
            decode_label = torch.zeros((N, 1, H, W), dtype=torch.float32)
            ord_labels = torch.zeros((N, C // 2, H, W), dtype=torch.float32)
        # print('#1 decode size:', decode_label.size())
        # ord_num = C // 2
        # for i in range(ord_num):
        #     ord_i = x[:, 2 * i:2 * i + 2, :, :]
        #     ord_i = nn.functional.softmax(ord_i, dim=1)  # compute P(w, h) in paper
        #     ord_i = ord_i[:, 1, :, :]
        #     ord_labels[:, i, :, :] = ord_i
        #     # print('ord_i >= 0.5 size:', (ord_i >= 0.5).size())
        #     decode_label += (ord_i >= 0.5).view(N, 1, H, W).float()  # sum(n(p_k >= 0.5))

        """
        replace iter with matrix operation
        fast speed methods
        """
        A = x[:, ::2, :, :].clone()
        B = x[:, 1::2, :, :].clone()

        A = A.view(N, 1, ord_num * H * W)
        B = B.view(N, 1, ord_num * H * W)

        C = torch.cat((A, B), dim=1)  # (b, 2, 6169028)
        # C = torch.clamp(C, min = 1e-8, max = 1e8) # prevent nans
        # print('C size:', C.size())

        ord_c = nn.functional.softmax(C, dim=1)  # (b, 2, 6169028)
        # print('ord_c size:', ord_c.size())

        ord_c1 = ord_c[:, 1, :].clone()  # (b, 1, 6169028)
        # print('ord_c1_1:', ord_c1.size())

        ord_c1 = ord_c1.view(-1, ord_num, H, W)  # (b, 68, 257, 353)
        # print('ord_c1_2:', ord_c1.size())

        decode_c = torch.sum(ord_c1, dim=1).view(-1, 1, H, W)  # (b, 1, 257, 353)
        # print('decode_c:', decode_c.size())

        return decode_c, ord_c1


class DORN(nn.Module):
    def __init__(self, output_size=(257, 353), channel=3):
        """
        模型网络初始化
        :param output_size:
        :param channel:
        """
        super(DORN, self).__init__()

        self.output_size = output_size
        self.channel = channel
        self.feature_extractor = ResNet(in_channels=channel, pretrained=True)
        self.aspp_module = SceneUnderstandingModule()
        self.orl = OrdinalRegressionLayer()

    def forward(self, x):
        x1 = self.feature_extractor(x)  # (b, 2048, 33, 45)
        # print(x1.size())
        x2 = self.aspp_module(x1)  # (b, 136, 257, 353)
        # print('DORN x2 size:', x2.size())
        depth_labels, ord_labels = self.orl(x2)
        return depth_labels, ord_labels


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 默认使用GPU 0

if __name__ == "__main__":
    # 模型网络层构建
    # 模块初始化
    resnet = ResNet().cuda().eval()
    aspp = SceneUnderstandingModule().cuda().eval()
    full = FullImageEncoder().cuda().eval()
    ord = OrdinalRegressionLayer().cuda().eval()

    image = torch.randn(1, 3, 257, 353)
    image = image.cuda()
    r1 = resnet(image)
    r2 = aspp(r1)
    r3_1, r3_2 = ord(r2)
    print(r2.size())
    os.system('pause')
    model = DORN().cuda().eval()
    with torch.no_grad():
        out0, out1 = model(image)
    print('out0 size:', out0.size())
    print('out1 size:', out1.size())
