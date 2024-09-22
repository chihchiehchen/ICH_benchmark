# modified from https://github.com/jeya-maria-jose/UNeXt-pytorch/blob/main/archs.py
import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from .coordconv import CoordConv2d
__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
import pdb
import torchvision.transforms.functional as F_vision


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
#     def shift(x, dim):
#         x = F.pad(x, "constant", 0)
#         x = torch.chunk(x, shift_size, 1)
#         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
#         x = torch.cat(x, 1)
#         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x



class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  out_ch = 6, in_ch=3, deep_supervision=False,img_size=512, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(in_ch, 16, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(128, 32, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        
        self.final = nn.Conv2d(16, out_ch, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)





class hourglass_mid(nn.Module):
    #rewrite the bottleneck layers of UNeXt
    def __init__(self, num_heads= 1,embed_dims=[ 128, 160, 256],qkv_bias=False, norm_layer=nn.LayerNorm,qk_scale=None,attn_drop_rate=0.,drop_rate=0.,sr_ratios=8,dpr=0):
        super(hourglass_mid,self).__init__()
        self.block1 = shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads, mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
            sr_ratio=sr_ratios)

        self.block2 = shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads, mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
            sr_ratio=sr_ratios)

        self.dblock1 = shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads, mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
            sr_ratio=sr_ratios)

        self.dblock2 = shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads, mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
            sr_ratio=sr_ratios)

        self.patch_embed3 = OverlapPatchEmbed( patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed( patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(embed_dims[2], embed_dims[1], 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(embed_dims[1], embed_dims[0], 3, stride=1, padding=1)  

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])
        self.dbn1 = nn.BatchNorm2d(embed_dims[1])
        self.dbn2 = nn.BatchNorm2d(embed_dims[0])
    def forward(self,x):
        B = x.shape[0]
        t3 = x

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(x)
        
        out = self.block1(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        
        out = self.block2(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        out = self.dblock1(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        
        out = self.dblock2(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return out


class hourglass_iter(nn.Module):
    def __init__(self, in_chans,embed_dims):
        super(hourglass_iter,self).__init__()
        self.l = len(in_chans)
        assert self.l >= 2
        
        self.enc_list = nn.ModuleList([ nn.Conv2d(in_chans[i], in_chans[i+1], 3, stride=1, padding=1) for i in range(self.l-1)])
         
        self.dec_list = nn.ModuleList([ nn.Conv2d(in_chans[i+1], in_chans[i], 3, stride=1, padding=1) for i in range(self.l-1)])  #nn.Conv2d(in_chans[1], in_chans[0], 3, stride=1, padding=1)
       
        self.ebn_list = nn.ModuleList([ nn.BatchNorm2d(in_chans[i+1]) for i in range(self.l-1)])
        self.dbn_list =  nn.ModuleList([ nn.BatchNorm2d(in_chans[i+1]) for i in range(self.l-2)])
        self.in_chans = in_chans
        self.in_conv = nn.Conv2d(in_channels=in_chans[-1], out_channels=embed_dims[0], kernel_size=1, stride=1, padding=0, dilation=1)
        self.out_conv = nn.Conv2d(in_channels=embed_dims[0], out_channels=in_chans[-1], kernel_size=1, stride=1, padding=0, dilation=1)
        
        self.bottleneck = hourglass_mid(embed_dims=embed_dims)
        
    def forward(self,x ):
        out = x 
        down_list = [] 
        for i in range(self.l -1):
            out = F.relu(F.max_pool2d(self.ebn_list[i](self.enc_list[i](out)),2,2))
            down_list.append(out)

        out = self.bottleneck(out)
        for i in range(self.l -2):
            out = F.relu(F.interpolate(self.dbn_list[self.l-3-i](self.dec_list[self.l-2-i](out)),scale_factor=(2,2),mode ='bilinear'))
        
            out = torch.add(out,down_list[self.l-3-i])
        out = F.relu(F.interpolate((self.dec_list[0](out)),scale_factor=(2,2),mode ='bilinear'))
              
        return out 



class UTM(nn.Module):
    def __init__(self,out_ch,embed_dims=[ 128, 128, 128,256],in_chans =[64,128] ):
        super(UTM,self).__init__()
        self.multires = hourglass_iter(in_chans =in_chans,embed_dims=embed_dims)
        self.norm = nn.BatchNorm2d(in_chans[0])

        self.select = nn.Conv2d(in_chans[0],out_ch,kernel_size = 1)
        self.skip = nn.Conv2d(in_chans[0],out_ch,kernel_size = 1)
        self.act = nn.GELU()
    def forward(self, x):
        out = self.select(self.norm(self.multires(x)))
        out += self.skip(x)
        return self.act(out)

class UTM_sup(nn.Module):
    def __init__(self,out_ch,embed_dims=[ 128, 192, 256],in_chans =[32,64,128] ,num_classes = 6, rev = True):
        super(UTM_sup,self).__init__()
        self.multires = hourglass_iter(embed_dims=embed_dims,in_chans =in_chans)
        self.norm = nn.BatchNorm2d(in_chans[0])
        self.final = nn.Conv2d(out_ch, num_classes, kernel_size=1)
        if rev:
            self.mid = nn.GELU()
            self.reverse = nn.Conv2d(num_classes,out_ch, kernel_size=1)
        self.select = nn.Conv2d(in_chans[0],out_ch,kernel_size = 1)
        self.skip = nn.Conv2d(in_chans[0],out_ch,kernel_size = 1)
        self.act = nn.GELU()
        self.rev = rev
    def forward(self, x):
        out = self.multires(x)
        if self.rev:
        
            seg_out = self.final(out)
            out = self.select(self.norm(out))
            out += self.skip(x)
        
            out += self.reverse(self.mid(seg_out))
        else:
            out = self.select(self.norm(out))
            out += self.skip(x)
            seg_out = self.final(out)

        return self.act(out),seg_out



class SUTM(nn.Module):
    def __init__(self,out_ch = 1, in_ch=3, b_l =3,embed_dims=[  128,256,512],in_chans =[32,32,64,128]): # embed_dims=[ 128, 128,256, 512],in_chans =[32,64,96,128]  [ 128, 192, 256] # embed_dims=[ 128, 128,256, 256],in_chans =[32,32,64,128]
        super(SUTM,self).__init__()
        #assert in_chans[-1] == embed_dims[0]
        self.coord = CoordConv2d(in_channels=in_ch, out_channels=in_chans[0],kernel_size = 3) #nn.Conv2d(in_ch,in_chans[0],kernel_size = 3,padding=1) #
        self.backbone = nn.ModuleList( [UTM(in_chans = in_chans, out_ch = in_chans[0],embed_dims=embed_dims) for i in range(b_l)] ) #,in_chans =[32,64,128]
        self.l = b_l
        self.final = nn.Conv2d(in_chans[0], out_ch, kernel_size=1)
    def forward(self, x):
        
        out = self.coord(x)
        for i in range(self.l):
            out = self.backbone[i](out)
        out = self.final(out)

        return out
#EOF

class SUTM_I(nn.Module):
    def __init__(self,out_ch = 6, in_ch=3, b_l =3,embed_dims=[  128,256,512],in_chans =[32,32,64,128],num_classes = 6):
        super(SUTM_I,self).__init__()
        self.coord = CoordConv2d(in_channels=in_ch, out_channels=in_chans[0],kernel_size = 3)
        self.backbone = nn.ModuleList( [UTM_sup(in_chans = in_chans,out_ch = 32,embed_dims=embed_dims,num_classes= num_classes,rev = True) for i in range(b_l-1)] +[UTM_sup(in_chans = in_chans, out_ch = 32,embed_dims=embed_dims,num_classes= num_classes,rev = False) ] )
        
        self.l = b_l
        

    def forward(self, x):
        
        out = self.coord(x)
        finallist = []
        for i in range(self.l):
            out,seg_out = self.backbone[i](out)
            finallist.append(seg_out)
        
        return finallist
