import torch
from torch import nn
import math
from einops import rearrange
import math
from inspect import isfunction
import pandas as pd
import numpy as np

def exists(x):

    #  If x is not None, it returns True; otherwise, it returns False. 
    return x is not None


def default(val, d):

    if exists(val):
        return val
    
    # isfunction returns True when passed a function, and False when passed a non-function
    return d() if isfunction(d) else d


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()

        self.fn = fn


    def forward(self, x, *args, **kwargs):

        return self.fn(x, *args, **kwargs) + x

class m_Linear(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out

        # 使用 nn.Linear 实现全连接层
        self.linear = nn.Linear(size_in, size_out)

    def forward(self, x):
        # 直接使用全连接层进行前向传播
        h = self.linear(x)
        return h

class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5 
        self.heads = heads
        hidden_dim = dim_head * heads   #32*4
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)


    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) -> b h c', h=self.heads), qkv)
        q = q * self.scale  #torch.Size([256, 4, 32])

        qk = torch.einsum('b h d, b h e -> b h d e', q, k)
        score = qk.softmax(dim=-1)
        out = torch.einsum('b h d e, b h e -> b h d', score, v)  # torch.Size([256, 4, 32])
        out = rearrange(out, 'b h c -> b (h c)', h=self.heads)

        return self.to_out(out)


class SinusoidalPositionEmbeddings(nn.Module):
    """
        Based on transformer-like embedding from 'Attention is all you need'
        Note: 10,000 corresponds to the maximum sequence length
    """

    def __init__(self, dim):
        super().__init__()
        assert dim >= 2, "dim must be at least 2"
        self.dim = dim


    def forward(self, time):
        assert len(time.shape) == 1

        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        # dim are channels 
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim))
        self.b = nn.Parameter(torch.zeros(1, dim))


    def forward(self, x):
        # computes the mean and variance of the input x along the channel dimension (dim=1)
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)

        # normalizes x
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        # self.norm = LayerNorm(dim)
        self.norm=nn.LayerNorm(dim) if isfunction(fn) else fn

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class newComplexTimeBlock(nn.Module):
    def __init__(self,length_in,length_out,time_dim=8,norm=True):
        super().__init__()
        self.leng_in = length_in
        self.leng_out = length_out
        self.flatten_linear=nn.Linear(length_in, length_in)  # [256,2*4] 空间和特征信息相关联  
        
        self.net = nn.Sequential(
            nn.Identity(),  # No normalization applied
            nn.Linear(length_in, length_in*2),
            nn.GELU(),
            nn.Linear(length_in*2, length_out)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, length_in) if time_dim!=length_in else nn.Identity(),
            nn.GELU(),
        )
        self.res_linear = nn.Linear(length_in, length_out) if length_in != length_out else nn.Identity()
        
    def forward(self,x,time_emb=None):
        h=self.flatten_linear(x)
        if time_emb!=None:
            assert exists(time_emb), 'time emb must be passed in'
            time_condition =self.time_mlp(time_emb)
            h=h+time_condition
        h=self.net(h)
        res_x=self.res_linear(x)    
        return h + res_x
        

##### Main Model #####
class UnetComplexBlock(nn.Module):
    #todo: 卷积核的size可能要再根据实际情况定夺一下
    def __init__(self, 
                 dim,   #4
                 loc_dim=7,
                 channels=2,
                 dim_mults=(1, 2, 4, 8),signal_feature_dim=8):
        super().__init__()
        self.p_tx=2
        self.leng = dim # input_dim
        time_dim = dim*2    # todo 写到yml里

        # if dim = 4, [2, 4, 8,16,32] 2是channels
        dims = [channels, *map(lambda m: dim * m, dim_mults)]

        # [(2, 4), (4, 8), (8, 16), (16, 32)]每一层的输入输出维度
        in_out = list(zip(dims[:-1], dims[1:]))

        #，用于处理时间嵌入。它将时间步长嵌入转换为特征向量，以便在模型的不同层中使用。
        #每个时间步长对应一个 16 维的嵌入向量。
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim*2),
                nn.Linear(dim * 2, dim*4),
                nn.GELU(),
                nn.Linear(dim * 4, dim*2)
            )

        self.class_emb = nn.Sequential(
                nn.LayerNorm(loc_dim),
                nn.Linear(loc_dim, dim), 
                nn.GELU(),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )


        self.downs = nn.ModuleList([] )
        self.ups = nn.ModuleList([])

        in_out=[(8,16),(16,32),(32,64),(64,128),(128,256)] #todo 后续也写到yml
        for ind, (length_in, length_out) in enumerate(in_out):

            self.downs.append(nn.ModuleList([
                newComplexTimeBlock(length_in, length_out, time_dim=time_dim, norm=(ind!=0)),
                newComplexTimeBlock(length_out, length_out, time_dim=time_dim),
                Residual(PreNorm(length_out, LinearAttention(length_out)))]))

        mid_dim = in_out[-1][1] 
        self.mid_block1 = newComplexTimeBlock(mid_dim, mid_dim, time_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = newComplexTimeBlock(mid_dim, mid_dim, time_dim=time_dim)

        # [(0, (64, 128)), (1, (32, 64)), (2, (16, 32)), (3, (8, 16))]
        for ind, (length_in, length_out) in enumerate(reversed(in_out[0:])):  

            self.ups.append(nn.ModuleList([
                newComplexTimeBlock(length_out, length_in, time_dim=time_dim),
                newComplexTimeBlock(length_in, length_in, time_dim=time_dim),
                Residual(PreNorm(length_in, LinearAttention(length_in)))]))

        # 将(batch_size,8)变为(batch_size,1,4)
        self.final_linear = nn.Sequential(
            newComplexTimeBlock(dim*2, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim),  # 8-> 4
            nn.Unflatten(1, (1, dim))  # Reshape to (batch_size, 1, dim // 2)
        )

    def forward(self, feature_x, time, location,sf,tp,true_distance):
        # feature_x: torch.Size([@, 1, 4])
        # time: torch.Size([@])
        # location: torch.Size([@, 3])
        time_2=time
        t = self.time_mlp(time_2)                                # (@ , ) => (@ , featuren_dim)
        class_cond = self.class_emb(location)                    # (@, 3) => (@, feature_dim)
        class_cond = class_cond.unsqueeze(dim=1)                 # (@, dim) => (@, 1, feature_dim)
        # feature_x=self.signal_linear(feature_x)                  # (@ , 1, dim) => (@, 1, signal_feature_dim)
        x = torch.cat((feature_x, class_cond), dim=1)            # (@, 1, signal_feature_dim) => (@, 2, signal_feature_dim)
        x=x.reshape(x.size(0), -1)                               # (@, 2, signal_feature_dim) => (@, 2*signal_feature_dim)   
        # (256,8)
        h = []
        for convnext, convnext2, attn in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)

        x = self.mid_block1(x, t)   # (@ ,128)
        x = self.mid_attn(x)        # (@ ,128)
        x = self.mid_block2(x, t)   # (@ ,128)
        for convnext, convnext2, attn in self.ups:
            skip_connection = h.pop()
            x = x + skip_connection  # Combine the skip connection with the current layer output
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)

        out = self.final_linear(x)

        return out
