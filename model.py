'''
multi-scale convolutional Siamese network
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from DropBlock import DropBlock1D

class Channel_Att_1D(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att_1D, self).__init__()
        self.channels = channels
        self.bn1 = nn.BatchNorm1d(self.channels, affine=True)
    def forward(self, x):
        residual = x
        x = self.bn1(x)
        # 计算权重，即Mc的计算
        weight_bn1 = self.bn1.weight.data.abs()
        weight_bn2 = torch.sum(self.bn1.weight.data.abs())
        weight_bn = weight_bn1 / weight_bn2
        weight_bn = weight_bn.unsqueeze(0).unsqueeze(-1)
        #x = x.permute(0, 2, 1).contiguous()
        x = torch.mul(weight_bn, x)
        #x = x.permute(0, 2, 1).contiguous()
        x = torch.sigmoid(x) * residual
        return x


class EmbeddingNet(nn.Module):
    def __init__(self, verbose=True):
        super(EmbeddingNet, self).__init__()
        self.nam1 = Channel_Att_1D(64)
        self.nam2 = Channel_Att_1D(64)

        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.Conv1d(64, 64, 3, 1, 3,dilation=3),
            nn.Conv1d(64, 64, 3, 1, 5, dilation=5),
            nn.Conv1d(64, 64, 3, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),#数字分别代表输入通道数，输出通道数，卷积核大小，步长，padding大小
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
        )
        self.conv22 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            DropBlock1D(),
        )
        self.conv222 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            DropBlock1D(),
        )
        self.conv2222 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            DropBlock1D(),
        )

        self.sigmoid = nn.Sigmoid()  # sigmoid激活函数
        self.Drop1 = DropBlock1D()
        self.Drop2 = DropBlock1D()
        self.Drop3 = DropBlock1D()
        self.Drop4 = DropBlock1D()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, padding=1)

        if verbose:
            print(f'{self._get_name()} - Number of parameters: {self.count_params()}  \n')  # 打印网络参数个数

    def count_params(self):  # 计算网络参数个数
        return sum(p.numel() for p in self.parameters())  # p.numel()返回p中元素个数

    def forward(self, x):
        x = x.unsqueeze(1)
        xquan = self.conv1d(x)
        x11 = self.conv1(x)
        x12 = self.conv3(x)
        x13 = self.conv4(x)
        x1 = torch.cat((x11, x12, x13), dim=1)  # 使用cat拼接,在通道维度上拼接,即将x11,x12,x13的通道数相加,得到x1,通道数为32+16+16=64,其余维度不变
        x1 = self.Drop1(x1)

        x2 = self.conv2(x1)
        x2 = self.Drop2(x2)
        x3 = self.conv22(x2)
        x3 = self.Drop2(x3)
        x3 = x1 + self.nam1(x3)

        x4 = self.conv222(x3)
        x4 = self.Drop3(x4)
        x4 = self.conv2222(x4)
        x4 = self.Drop4(x4)
        x5 = x1 + x3 + self.nam2(x4)

        x5 = self.maxpool1(x5)
        x5 = x5 + xquan
        out = self.sigmoid(x5)  # 激活函数，作用是将数据映射到0-1之间
        return out


class SiameseNet(nn.Module):#孪生网络,它的作用是将两个输入映射到同一个空间，然后计算两个输入的相似度
    def __init__(self,embedding_net):
        super(SiameseNet,self).__init__()
        self.embedding_net = embedding_net

    def forward(self,x1,x2):
        return self.embedding_net(x1),self.embedding_net(x2)#

    def get_embedding(self,x):
        return self.embedding_net(x)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, output1, output2, label, size_average=True):
        distances = F.cosine_similarity(output1,output2,dim=1)
        loss1= (label) * torch.pow(distances, 2) + (1-label) * torch.pow(torch.clamp(
            self.margin - distances, min=0.0), 2)
        label = label.repeat(1, 63)
        lambda_val = 10
        distances = distances.cpu().detach().numpy()
        weight = 1 - np.exp(-lambda_val * (distances-0.6))
        weight = np.maximum(weight, 0)
        distances = distances * weight
        distances = torch.from_numpy(distances).to(label.device)
        bce = nn.BCELoss(reduction='mean')(distances, 1-label)
        losses = 0.9 * loss1 + 0.1* bce

        return losses.mean() if size_average else losses.sum()


