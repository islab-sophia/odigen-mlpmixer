import torch
import torch.nn as nn
import torch.nn.functional as F

from mlpmixer_cbn import MLPCBNMixer,MLPCBNDepthWiseMixer

class MLPLayerBlock(nn.Module):
    def __init__(self,in_channel,layer,kernel=3,use_patch=False,single=False):
        super(MLPLayerBlock, self).__init__()
        if not use_patch:
            self.mixer = MLPCBNMixer(layer=layer, dropout=0, channel_size=in_channel)
        else:
            self.mixer = MLPCBNDepthWiseMixer(layer=layer, channel_size=in_channel,kernel=kernel,single=single)
    def forward(self,x,label):
        x = self.mixer(x,label)
        return x

class RGB_module(nn.Module):
    def __init__(self,in_channel,out_channel,h,w,output_resolution=(256,512),up=True,kernel=1):
        super(RGB_module, self).__init__()
        if up:
            self.conv = nn.ConvTranspose2d(in_channel,out_channel,2,2)
        self.to_rgb = nn.Conv2d(in_channel,3,kernel,padding=(kernel-1)//2)
        self.resolution = output_resolution
        self.h = h
        self.w = w


    def forward(self,x):
        rgb = self.to_rgb(x)
        rgb = F.interpolate(rgb,self.resolution)
        if hasattr(self,"conv"):
            x = self.conv(x)
        return x,rgb

class Generator(nn.Module):
    def __init__(self,channel=512,layer=[4,4,4,4,4],activation=True,dw_kernel=5,single=False,out_kernel=1):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3,channel,(16,16),(16,16))
        self.conv2 = nn.Conv2d(3,channel//4,(8,8),(8,8))
        self.conv3 = nn.Conv2d(3,channel//16,(4,4),(4,4))
        self.conv4 = nn.Conv2d(3,channel//64,(2,2),(2,2))
        self.conv5 = nn.Conv2d(3,channel//128,(1,1),(1,1))
        self.squeeze_conv = nn.Conv2d(channel * 2, channel, 1, 1)
        self.mixer1 = MLPLayerBlock(channel, layer[0])
        self.mixer2 = MLPLayerBlock(channel // 4, layer[1], use_patch=True, kernel=dw_kernel,single=single)
        self.mixer3 = MLPLayerBlock(channel // 16, layer[2], use_patch=True, kernel=dw_kernel,single=single)
        self.mixer4 = MLPLayerBlock(channel // 64, layer[3], use_patch=True, kernel=dw_kernel,single=single)
        self.mixer5 = MLPLayerBlock(channel // 128, layer[4], use_patch=True)
        self.activation = activation
        self.to_rgb1 = RGB_module(channel,channel//4,16,32,kernel=out_kernel)
        self.to_rgb2 = RGB_module(channel//4,channel//16,32,64,kernel=out_kernel)
        self.to_rgb3 = RGB_module(channel//16,channel//64,64,128,kernel=out_kernel)
        self.to_rgb4 = RGB_module(channel//64,channel//128,128,256,kernel=out_kernel)
        self.to_rgb5 = RGB_module(channel//128,None,256,512,up=False,kernel=out_kernel)

    def forward(self,x,label):
        img = x
        x = self.conv1(x)
        b,c,h,w = x.shape
        seed_l = torch.randn(b,c,h,w,device=x.device)
        x = torch.cat([x,seed_l], dim=1)
        x = self.squeeze_conv(x)
        x = self.mixer1(x,label)
        x = x.reshape(b,c,h,w)
        x,rgb1 = self.to_rgb1(x)
        x += self.conv2(img)
        x = self.mixer2(x,label)
        x,rgb2 = self.to_rgb2(x)
        x += self.conv3(img)
        x = self.mixer3(x,label)
        x,rgb3 = self.to_rgb3(x)
        x += self.conv4(img)
        x = self.mixer4(x,label)
        x,rgb4 = self.to_rgb4(x)
        x += self.conv5(img)
        x = self.mixer5(x,label)
        _,rgb5 = self.to_rgb5(x)
        img = rgb1 + rgb2 + rgb3+ rgb4+ rgb5
        if self.activation:
            img = torch.tanh(img)
        return img
if __name__ == '__main__':
    inp = torch.randn(5,3,256,512).cuda()
    label = torch.randn(5,24).cuda()
    module = Generator().cuda()
    out = module(inp,label)
    print(out.shape)