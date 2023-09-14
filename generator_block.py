import torch
import torch.nn as nn
import torch.nn.functional as F
from mlpmixer_cbn import ConditionalNorm,MLPCBNMixer,circle_padding

class DepthwiseModule(nn.Module):
    def __init__(self,channel,kernel=3):
        super(DepthwiseModule, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel, 1, 0, groups=channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, 1, 0, groups=channel)
        self.padding = kernel//2
    def forward(self,x):
        x = circle_padding(x,self.padding)
        x = self.conv1(x)
        x = circle_padding(x,self.padding)
        x = self.conv2(x)
        return x

class MLPCBNDepthWiseBlock(nn.Module):
    def __init__(self,channel,kernel=3,**kwargs):
        super(MLPCBNDepthWiseBlock, self).__init__()
        channel_hidden = channel*2
        self.norm1 = ConditionalNorm(channel,dim=2)
        self.norm2 = ConditionalNorm(channel,dim=2)
        self.module1 = DepthwiseModule(channel,kernel)
        self.module2 = nn.Sequential(
            nn.Conv2d(channel,channel_hidden,1,1,0),
            nn.GELU(),
            nn.Conv2d(channel_hidden,channel,1,1,0),)

    def forward(self,x,label):
        skip = x
        x = self.norm1(x,label)
        x = self.module1(x)
        x = x + skip
        skip = x
        x = self.norm2(x,label)
        x = self.module2(x)
        x = x + skip
        return x

class MLPCBNSingleDepthWiseBlock(nn.Module):
    def __init__(self,channel,kernel=3,**kwargs):
        super(MLPCBNSingleDepthWiseBlock, self).__init__()
        channel_hidden = channel*2
        self.norm1 = ConditionalNorm(channel,dim=2)
        self.norm2 = ConditionalNorm(channel,dim=2)
        self.conv1 = nn.Conv2d(channel, channel, kernel, 1, 0, groups=channel)
        self.module2 = nn.Sequential(
            nn.Conv2d(channel,channel_hidden,1,1,0),
            nn.GELU(),
            nn.Conv2d(channel_hidden,channel,1,1,0),)
        self.padding = kernel // 2

    def forward(self,x,label):
        # x : (batch,channel,h,w)
        skip = x
        x = self.norm1(x,label)
        x = circle_padding(x, self.padding)
        x = self.conv1(x)
        x = x + skip
        skip = x
        x = self.norm2(x,label)
        x = self.module2(x)
        x = x + skip
        return x

class MLPCBNDepthWiseMixer(nn.Module):
    def __init__(self,channel_size=256,layer=8,kernel=3,single=False):
        super(MLPCBNDepthWiseMixer, self).__init__()
        block = MLPCBNSingleDepthWiseBlock if single else MLPCBNDepthWiseBlock
        layers = [block(channel_size,kernel) for i in range(layer)]
        self.layers = nn.ModuleList(layers)
    def forward(self,x,label):
        for i in self.layers:
            x = i(x,label)
        return x

class RGB_module(nn.Module):
    def __init__(self,in_channel,out_channel,output_resolution=(256,512),up=True):
        super(RGB_module, self).__init__()
        if up:
            self.conv = nn.ConvTranspose2d(in_channel,out_channel,2,2)
        self.to_rgb = nn.Conv2d(in_channel,3,1,1)
        self.resolution = output_resolution

    def forward(self,x):
        rgb = self.to_rgb(x)
        rgb = F.interpolate(rgb,self.resolution)
        if hasattr(self,"conv"):
            x = self.conv(x)
        return x,rgb

class MLPMixerBlock(nn.Module):
    def __init__(self,channel=512,patch_size=16,layer=4,output_reduction=4):
        super(MLPMixerBlock, self).__init__()
        self.conv = nn.Conv2d(3,channel,(patch_size,patch_size),(patch_size,patch_size))
        self.mixer = MLPCBNMixer(layer=layer, dropout=0, noise=False, skip=False, channel_size=channel,down=False)
        self.to_rgb = RGB_module(channel,channel//output_reduction,up=True)
        self.squeeze = nn.Linear(channel*2,channel)

    def forward(self,img,noise,label):
        x = self.conv(img)
        b, c, h, w = x.shape
        x = x.view(b,c,-1)
        x = torch.cat([x, noise], dim=1)
        x = self.squeeze(x.transpose(1,2)).transpose(1,2)
        x = self.mixer(x, label)
        x = x.reshape(b, c, h, w)
        x, rgb = self.to_rgb(x)
        return x,rgb

class DepthwiseConvBlock(nn.Module):
    def __init__(self,channel=128,patch_size=8,layer=4,output_reduction=4,is_last=False,dw_kernel=5):
        super(DepthwiseConvBlock, self).__init__()
        self.conv = nn.Conv2d(3,channel,(patch_size,patch_size),(patch_size,patch_size))
        self.mixer = MLPCBNDepthWiseMixer(layer=layer, channel_size=channel, kernel=dw_kernel, single=False)
        self.to_rgb = RGB_module(channel,channel//output_reduction,up=not is_last)

    def forward(self,img,x,label):
        x = self.conv(img) + x
        x = self.mixer(x, label)
        x, rgb = self.to_rgb(x)
        return x,rgb

class Generator(nn.Module):
    def __init__(self,channel=512,layer=[4,4,4,4,4],activation=True,dw_kernel=5):
        super(Generator, self).__init__()
        self.layer1 = MLPMixerBlock(channel, patch_size=16, layer=layer[0], output_reduction=4)
        self.layer2 = DepthwiseConvBlock(channel // 4, patch_size=8, layer=layer[1], output_reduction=4,
                                         is_last=False, dw_kernel=dw_kernel)
        self.layer3 = DepthwiseConvBlock(channel // 16, patch_size=4, layer=layer[2], output_reduction=4,
                                         is_last=False, dw_kernel=dw_kernel)
        self.layer4 = DepthwiseConvBlock(channel // 64, patch_size=2, layer=layer[3], output_reduction=2,
                                         is_last=False, dw_kernel=dw_kernel)
        self.layer5 = DepthwiseConvBlock(channel // 128, patch_size=1, layer=layer[4],
                                         is_last=True, dw_kernel=dw_kernel)
        self.act = activation

    def forward(self,img,noise,label):
        x, rgb1 = self.layer1(img,noise,label)
        x, rgb2 = self.layer2(img,x,label)
        x, rgb3 = self.layer3(img,x,label)
        x, rgb4 = self.layer4(img,x,label)
        _, rgb5 = self.layer5(img,x,label)
        img_out = rgb1 + rgb2 + rgb3 + rgb4 + rgb5
        if self.act:
            img_out = torch.tanh(img_out)
        return img_out

if __name__ == '__main__':
    inp = torch.randn(5,3,256,512)
    noise = torch.randn(5,512,512)
    label = torch.randn(5,24)
    module = Generator()
    out = module(inp,noise,label)
    print(out.shape)