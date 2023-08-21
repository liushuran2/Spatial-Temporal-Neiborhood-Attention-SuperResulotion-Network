import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from common import FCABs
from natten import NeighborhoodNonlocalAttention2D
def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class STNA(nn.Module):
    def __init__(self, config, is_training=True):
        super(STNA, self).__init__()
        self.in_channels = config['in_channels']
        self.inter_channels = self.in_channels // 2
        self.is_training=is_training
        self.num_input_frame = config['num_input_frame']
        self.scale = config['scale']
                
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Conv2d(in_channels=self.inter_channels*self.num_input_frame, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True)
        
        self.W_z1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W_z1.weight, 0)
        nn.init.constant_(self.W_z1.bias, 0)
        self.mb = torch.nn.Parameter(torch.randn(self.inter_channels, 256))

        self.att = NeighborhoodNonlocalAttention2D(dim = self.inter_channels, num_heads=1,kernel_size=3)
    
    def forward(self, x):
        b, t, c, h, w = x.size()
        q = x[:, int((self.num_input_frame - 1) / 2), :, :, :]
        

        reshaped_x = x.view(b*t , c, h, w).contiguous()
        h_ = self.norm(reshaped_x)
        q_=self.norm(q)

        g_x = self.g(h_).view(b, t, self.inter_channels, h,w).contiguous()
        theta_x = self.theta(h_).view(b, t, self.inter_channels,  h,w).contiguous()

        phi_x = torch.unsqueeze(self.phi(q_),dim=1)
        phi_x  =phi_x.permute(0,1,3,4,2).contiguous()
        phi_x_for_quant = phi_x.permute(0,1,4,2,3).view(b,self.inter_channels, -1).permute(0,2,1)

        corr_l = []
        for i in range(t):
            theta = theta_x[:, i, :, :, :]
            g = g_x[:, i, :, :, :]

            g = torch.unsqueeze(g.permute(0,2,3,1).contiguous(),dim=1)
            theta = torch.unsqueeze(theta.permute(0,2,3,1).contiguous(),dim=1)
            
            if self.is_training:
                f = self.att(phi_x, theta, g)
            else: 
                f = self.att(phi_x, theta, g)
            y = f.view(b,self.inter_channels,h,w)
            corr_l.append(y)

        corr_prob = torch.cat(corr_l, dim=1).view(b, -1, h, w)
        W_y = self.W_z(corr_prob)
        
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = torch.matmul(phi_x_for_quant, mbg)
        f_div_C1 = F.softmax(f1 * (int(self.inter_channels) ** (-0.5)), dim=-1)
        y1 = torch.matmul(f_div_C1, mbg.permute(0, 2, 1))
        qloss=torch.mean(torch.abs(phi_x_for_quant-y1))
        y1 = y1.permute(0, 2, 1).view(b, self.inter_channels, h, w).contiguous()
        W_y1 = self.W_z1(y1)
        
        z = W_y + q+W_y1

        return z, qloss


class mana(nn.Module):
    def __init__(self, config,is_training):
        super(mana, self).__init__()
        self.in_channels = config['in_channels']
        self.conv_first = nn.Conv2d(1, self.in_channels, 3, 1, 1)
        self.conv_input = nn.Conv2d(1, 64, 3, 1, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.scale = config['scale']
        self.num_input_frame = config['num_input_frame']

        self.encoder = make_layer(FCABs, config['encoder_nblocks'], channel=self.in_channels)
        self.decoder = make_layer(FCABs, config['decoder_nblocks'], channel=self.in_channels)

        self.temporal_spatial = STNA(config, is_training)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sig = nn.Sigmoid()

        # upsample
        self.upconv1 = nn.Conv2d(self.in_channels, 64 * self.scale * self.scale, 3, 1, 1)
        self.upconv2 = nn.Conv2d(self.in_channels, 64 * self.scale * self.scale, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(self.scale)
        self.conv_hr1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_last1 = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_last2 = nn.Conv2d(64, 1, 3, 1, 1)


    def forward(self, x):
        b, t, c, h, w = x.size()
        lx = F.interpolate(x[:, int((self.num_input_frame - 1) / 2), :, :, :], scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        encode = self.encoder((self.lrelu(self.conv_first(x.view(-1, c, h, w)))))
        mem=encode.view(b,t,self.in_channels,h,w).contiguous()
        res,qloss = self.temporal_spatial(mem)
        out = self.decoder(res)
        
        out1 = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out2 = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out1 = self.lrelu(self.conv_hr1(out1))
        out2 = self.lrelu(self.conv_hr2(out2))

        feature_input = self.conv_input(lx)
        out1 = out1 + feature_input
        out2 = out2 + feature_input

        out1 = self.conv_last1(out1)
        out1 = lx + out1

        out2 = self.conv_last2(out2)
        out2 = self.sig(out2)

        oup = torch.cat((out1, out2), 1)
        return oup,qloss