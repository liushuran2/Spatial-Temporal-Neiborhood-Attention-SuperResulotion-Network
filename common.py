import torch
import torch.nn as nn


def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    return x * cdf

def fft2d(input, gamma=0.8):
    temp = input.permute(0, 3, 1, 2)
    fft = torch.fft.fftn(torch.complex(temp, torch.zeros_like(temp)))
    absfft = torch.pow(torch.abs(fft) + 1e-8, gamma)
    output = absfft.permute(0, 2, 3, 1)
    return output

def fftshift2d(input, size_psc=128):
    bs, h, w, ch = input.size()
    fs11 = input[:, -h // 2:h, -w // 2:w, :]
    fs12 = input[:, -h // 2:h, 0:w // 2, :]
    fs21 = input[:, 0:h // 2, -w // 2:w, :]
    fs22 = input[:, 0:h // 2, 0:w // 2, :]
    output = torch.cat([torch.cat([fs11, fs21], dim=1), torch.cat([fs12, fs22], dim=1)], dim=2)
    output = torch.nn.functional.interpolate(output, size=(size_psc, size_psc), mode='bilinear', align_corners=True)
    return output

def global_average_pooling2d(layer_in):
    return torch.mean(layer_in, dim=(2, 3), keepdim=True)

class FCALayer(nn.Module):
    def __init__(self, channel, reduction=16, size_psc=128):
        super().__init__()
        self.fft = fft2d
        self.fft2d = fftshift2d
        self.conv0 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.avg_pool = global_average_pooling2d
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        w = self.fft(x)
        w = self.fft2d(w)
        w = self.relu(self.conv0(w))
        w = self.conv1(self.avg_pool(w))
        w = self.relu(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return x * w
    
class FCAB(nn.Module):
    def __init__(self, channel, size_psc=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.gelu = gelu
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.fca = FCALayer(channel, reduction=16, size_psc=size_psc)
        self.dropout = nn.Dropout(p=0.8)

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.gelu(conv)
        conv = self.dropout(conv)
        conv = self.conv2(conv)
        conv = self.gelu(conv)
        att = self.fca(conv)
        return att + x
    
class FCABs(nn.Module):
    def __init__(self, channel, size_psc=128, n_RCAB=4):
        super(FCABs, self).__init__()
        self.n_RCAB = n_RCAB
        self.FCABs = nn.ModuleList([FCAB(channel, size_psc) for _ in range(self.n_RCAB)])
        
    def forward(self, input):
        conv = input
        for i in range(self.n_RCAB):
            conv = self.FCABs[i](conv)
        conv = conv + input
        return conv




        
        