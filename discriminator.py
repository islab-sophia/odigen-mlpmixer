import torch
import torch.nn as nn
import torch.nn.functional as F
from mlpmixer import MLPBlock
from augment import DiffAugment

class Discriminator(nn.Module):
    def __init__(self,in_channel=3,channel_size=256,patch_size=(16,16),img_size=(256,512),layer=8,dropout=0.3,augment=False):
        super(Discriminator, self).__init__()
        p_h,p_w = patch_size
        img_h,img_w = img_size
        self.h,self.w = img_h//p_h,img_w//p_w
        self.conv =  nn.Conv2d(in_channel,channel_size,patch_size,patch_size)
        layers = [MLPBlock(channel_size,self.h*self.w,dropout=dropout) for i in range(layer)]
        self.layers = nn.Sequential(*layers)
        self.local_linear = nn.Linear(channel_size,1)
        self.channel_linear = nn.Linear(self.h*self.w,1)
        self.upconv = nn.ConvTranspose2d(channel_size,in_channel,patch_size,patch_size)
        self.augment = augment

    def forward(self,x,*args):
        if self.augment:
            x = DiffAugment(x, 'color,translation,cutout')
        inp = x
        x = self.conv(x)
        x = x.view(x.size(0),x.size(1),-1)
        x = self.layers(x)
        ss = x.reshape(x.size(0),x.size(1),self.h,self.w)
        channel_pred = self.channel_linear(x)
        x = x.transpose(1,2)
        local_pred = self.local_linear(x)
        ss = self.upconv(ss)
        ss = F.l1_loss(inp,ss) # dis rec loss
        return local_pred,channel_pred,ss

if __name__ == '__main__':
    inp = torch.randn(5,3,256,512).cuda()
    module = Discriminator().cuda()
    out = module(inp)
    print(out[0].shape)
    print(out[1].shape)