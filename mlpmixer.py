import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self,channel,token,channel_hidden=2048,token_hidden=1024,dropout=0.3,**kwargs):
        super(MLPBlock, self).__init__()
        self.ln1 = nn.LayerNorm((token,))
        self.ln2 = nn.LayerNorm((token,))
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
    def forward(self,x):
        skip = x
        x = self.ln1(x).permute(0,2,1)
        x = self.MLP1(x).permute(0,2,1)
        x = x + skip
        skip = x
        x = self.ln2(x)
        x = self.MLP2(x)
        x = x + skip
        return x

class MLPMixer(nn.Module):
    def __init__(self,in_channel=3,channel_size=256,patch_size=(16,16),img_size=(256,512),layer=8,dropout=0.3,
                 noise=False,down=True):
        super(MLPMixer, self).__init__()
        p_h,p_w = patch_size
        img_h,img_w = img_size
        h,w = img_h//p_h,img_w//p_w
        if down:
            self.conv = nn.Conv2d(in_channel,channel_size,patch_size,patch_size,0)
        block = MLPBlock
        layers = [block(channel_size,h*w,dropout=dropout,noise=noise) for i in range(layer)]
        self.layers = nn.Sequential(*layers)
        self.down = down
        self.token_size = h*w
        self.channel = channel_size
    def forward(self,x):
        if self.down:
            x = self.conv(x)
        x = x.view(x.size(0),x.size(1),-1)
        x = self.layers(x)
        if type(x) is tuple:
            x = x[0]
        return x



if __name__ == '__main__':
    inp = torch.randn((4,16,256,512)).cuda()
    module = MLPMixer(16,256).cuda()
    out = module(inp)
    print(out.shape)