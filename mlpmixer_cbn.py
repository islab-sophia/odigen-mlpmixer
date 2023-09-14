import torch
import torch.nn as nn
import torch.nn.functional as F

def circle_padding(tensor, pad=1):
    tensor = F.pad(tensor, (pad, pad, 0, 0), mode="circular")
    tensor = F.pad(tensor, (0, 0, pad, pad))
    return tensor

class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition=24,dim=3):
        super().__init__()
        if dim==1:
            self.bn = nn.BatchNorm1d(in_channel, affine=False)
        elif dim==2:
            self.bn = nn.BatchNorm2d(in_channel, affine=False)
        elif dim==3:
            self.bn = nn.BatchNorm3d(in_channel, affine=False)
        else:
            raise NotImplementedError
        self.embed = nn.Linear(n_condition, in_channel* 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0
        self.dim= dim

    def forward(self, input, class_id):
        out = self.bn(input)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        for i in range(2,self.dim+2):
            gamma = gamma.unsqueeze(i)
            beta = beta.unsqueeze(i)
        out = gamma * out + beta
        return out

class MLPCBNBlock(nn.Module):
    def __init__(self,channel,token,channel_hidden=2048,token_hidden=1024,dropout=0.3,**kwargs):
        super(MLPCBNBlock, self).__init__()
        self.norm1 = ConditionalNorm(channel,dim=1)
        self.norm2 = ConditionalNorm(channel,dim=1)
        self.MLP1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channel,channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden,channel))
        self.MLP2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(token,token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden,token))

    def forward(self,x,class_id):
        skip = x
        x = self.norm1(x,class_id).permute(0,2,1)
        x = self.MLP1(x).permute(0,2,1)
        x = x + skip
        skip = x
        x = self.norm2(x,class_id)
        x = self.MLP2(x)
        x = x + skip
        return x


class MLPCBNPatchBlock(nn.Module):
    def __init__(self,channel,h,w,patch=16,dropout=0.3,noise=False,**kwargs):
        super(MLPCBNPatchBlock, self).__init__()
        self.h_patch = h//patch
        self.w_patch = w//patch
        channel_hidden = channel*2
        token_hidden = patch*patch*2
        self.norm1 = ConditionalNorm(channel)
        self.norm2 = ConditionalNorm(channel)
        self.MLP1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channel,channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden,channel))
        self.MLP2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(patch**2,token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden,patch**2))
        self.patch = patch
        self.noise = noise

    def forward(self,x,class_id):
        # x : (batch,channel,patch_h,patch_w,patch*patch)
        skip = x
        if self.noise:
            x += 0.1*torch.randn_like(x)
        x = self.norm1(x,class_id).transpose(1,4)
        x = self.MLP1(x).transpose(1,4)
        x = x + skip
        skip = x
        x = self.norm2(x,class_id)
        x = self.MLP2(x)
        x = x + skip
        return x

class MLPCBNMixer(nn.Module):
    def __init__(self,channel_size=256,patch_size=(16,16),img_size=(256,512),layer=8,dropout=0.3,
                 noise=False):
        super(MLPCBNMixer, self).__init__()
        p_h,p_w = patch_size
        img_h,img_w = img_size
        h,w = img_h//p_h,img_w//p_w
        layers = [MLPCBNBlock(channel_size,h*w,dropout=dropout,noise=noise) for i in range(layer)]
        self.layers = nn.ModuleList(layers)
    def forward(self,x,label):
        if hasattr(self,"conv"):
            x = self.conv(x)
        x = x.view(x.size(0),x.size(1),-1)
        for i in self.layers:
            x = i(x,label)
        if type(x) is tuple:
            x = x[0]
        return x

class DepthwiseConvModule(nn.Module):
    def __init__(self,channel,kernel=3,single=False):
        super(DepthwiseConvModule, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel, 1, 0, groups=channel)
        if not single:
            self.conv2 = nn.Conv2d(channel, channel, kernel, 1, 0, groups=channel)
        self.padding = kernel//2
        self.single = single
    def forward(self,x):
        x = circle_padding(x,self.padding)
        x = self.conv1(x)
        if not self.single:
            x = circle_padding(x,self.padding)
            x = self.conv2(x)
        return x

class MLPCBNDoubleDepthWiseBlock(nn.Module):
    def __init__(self,channel,kernel=3,**kwargs):
        super(MLPCBNDoubleDepthWiseBlock, self).__init__()
        channel_hidden = channel*2
        self.norm1 = ConditionalNorm(channel,dim=2)
        self.norm2 = ConditionalNorm(channel,dim=2)
        self.module1 = DepthwiseConvModule(channel,kernel)
        self.module2 = nn.Sequential(
            nn.Conv2d(channel,channel_hidden,1,1,0),
            nn.GELU(),
            nn.Conv2d(channel_hidden,channel,1,1,0),)

    def forward(self,x,class_id):
        # x : (batch,channel,h,w)
        skip = x
        x = self.norm1(x,class_id)
        x = self.module1(x)
        x = x + skip
        skip = x
        x = self.norm2(x,class_id)
        x = self.module2(x)
        x = x + skip
        return x

class MLPCBNDepthWiseMixer(nn.Module):
    def __init__(self,channel_size=256,layer=8,kernel=3,single=False):
        super(MLPCBNDepthWiseMixer, self).__init__()
        block = MLPCBNDoubleDepthWiseBlock
        layers = [block(channel_size,kernel,single=single) for i in range(layer)]
        self.layers = nn.ModuleList(layers)
    def forward(self,x,label):
        for i in self.layers:
            x = i(x,label)
        return x

if __name__ == '__main__':
    inp = torch.randn((4,16,256,512)).cuda()
    id = torch.randn((4,24)).cuda()
    module = MLPCBNDepthWiseMixer(16,layer=4).cuda()
    out = module(inp,id)
    print(out[0].shape)