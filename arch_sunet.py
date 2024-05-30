import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile


#将IGMSA替换为IFA进行特征融合
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from SS2D_arch import SS2D
from IFA_arch import IFA
import sys
print(sys.path)


# 这两个函数 _no_grad_trunc_normal_ 和 trunc_normal_ 都用于初始化神经网络中的权重参数。它们将权重初始化为遵循截断正态分布的值，这种方法有助于在训练开始时改善网络的性能和稳定性。
def _no_grad_trunc_normal_(tensor, mean, std, a, b):#这是一个内部函数，用于实际执行初始化过程。
    """
    用截断正态分布初始化张量。

    在给定的范围 [a, b] 内，根据指定的均值 (mean) 和标准差 (std) 截断正态分布，
    并用这个分布来填充输入的张
    量。
    

    参数:
        tensor (Tensor): 需要被初始化的张量。
        mean (float): 截断正态分布的均值。
        std (float): 截断正态分布的标准差。
        a (float): 分布的下限。
        b (float): 分布的上限。

    返回:
        Tensor: 填充后的张量，其值遵循指定的截断正态分布。
    
    功能：
        这个函数主要用于深度学习中的权重初始化，尤其是在需要限制权重范围以避免激活值过大或过小时非常有用。通过截断正态分布进行初始化，可以帮助神经网络更快地收敛并提高模型的稳定性。
    """
    # 计算正态分布的累积分布函数（CDF）值
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # 检查均值是否在截断区间外的两个标准差范围内
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():  # 确保以下操作不计算梯度
        # 计算截断区间的累积分布函数值
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)
        # 以这个区间初始化张量
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()  # 计算逆误差函数，以获得正态分布的值
        tensor.mul_(std * math.sqrt(2.))  # 缩放
        tensor.add_(mean)  # 加上均值
        tensor.clamp_(min=a, max=b)  # 截断到 [a, b]

        return tensor  # 返回初始化后的张量

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):#这是一个公开的接口函数，通常由外部调用来初始化权重。
    """
    将输入张量初始化为截断正态分布。

    这个函数是`_no_grad_trunc_normal_`的公开接口，用于初始化张量，使其值遵循指定均值和标准差的截断正态分布。

    参数:
        tensor (torch.Tensor): 需要初始化的张量。
        mean (float, 可选): 正态分布的均值，默认为 0。
        std (float, 可选): 正态分布的标准差，默认为 1。
        a (float, 可选): 分布的下限，默认为 -2。
        b (float, 可选): 分布的上限，默认为 2。

    返回:
        torch.Tensor: 初始化后的张量，其值遵循指定的截断正态分布。
    
    功能：
        这个函数通常用于神经网络的权重初始化过程中，特别是当我们需要确保权重在特定范围内且遵循正态分布时。通过截断分布，我们可以避免极端值的影响，从而帮助提高模型训练的稳定性和效率。
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    """
    对张量进行变量缩放初始化。

    这个函数根据张量的维度(fan_in, fan_out)和给定的参数，对张量进行初始化，
    以确保初始化的张量具有一定的方差，这有助于控制网络层输出的方差在训练初期保持稳定。

    参数:
        tensor (torch.Tensor): 需要初始化的张量。
        scale (float): 缩放因子,用于调整方差,默认为1.0。
        mode (str): 指定使用张量的哪部分维度来计算方差。
                    'fan_in'：使用输入维度，
                    'fan_out'：使用输出维度，
                    'fan_avg'：使用输入和输出维度的平均值。
                    默认为'fan_in'。
        distribution (str): 初始化的分布类型，可以是
                    'truncated_normal'：截断正态分布，
                    'normal'：正态分布，
                    'uniform'：均匀分布。
                    默认为'normal'。

    返回:
        None，直接在输入的张量上进行修改。
    
    功能：
        这种初始化方法通常用于深度神经网络中，可以帮助保持激活函数输入的方差在训练过程中保持稳定，从而有助于避免梯度消失或爆炸问题。通过调整scale、mode和distribution参数，可以进一步优化模型的初始化过程。
    """

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)  # 计算张量的输入和输出维度大小 

    # 根据mode确定分母
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom  # 计算方差

    # 根据指定的分布进行初始化
    if distribution == "truncated_normal":
        # 使用截断正态分布进行初始化，std是标准差
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        # 使用正态分布进行初始化
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        # 使用均匀分布进行初始化
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        # 如果提供了无效的分布类型，抛出异常
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    """
    使用 LeCun 正态初始化方法对张量进行初始化。

    LeCun 初始化是一种变量缩放方法，特别适用于带有S型激活函数（如sigmoid或tanh）的深度学习模型。
    它根据层的输入数量（fan_in）来调整权重的缩放，从而使网络的训练更加稳定。

    参数:
        tensor (torch.Tensor): 需要初始化的张量。

    返回:
        None，直接在输入的张量上进行修改。
    """
    # 调用 variance_scaling_ 函数，使用'fan_in'模式和'normal'分布进行初始化
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class PreNorm(nn.Module):
    """
    预归一化模块，通常用于Transformer架构中。

    在执行具体的功能（如自注意力或前馈网络）之前先进行层归一化，
    这有助于稳定训练过程并提高模型性能。

    属性:
        dim: 输入特征的维度。
        fn: 要在归一化后应用的模块或函数。
    """

    def __init__(self, dim, fn):
        """
        初始化预归一化模块。

        参数:
            dim (int): 输入特征的维度，也是层归一化的维度。
            fn (callable): 在归一化之后应用的模块或函数。
        """
        super().__init__()  # 初始化基类 nn.Module
        self.fn = fn  # 存储要应用的函数或模块
        self.norm = nn.LayerNorm(dim)  # 创建层归一化模块

    def forward(self, x, *args, **kwargs):
        """
        对输入数据进行前向传播。

        参数:
            x (Tensor): 输入到模块的数据。
            *args, **kwargs: 传递给self.fn的额外参数。

        返回:
            Tensor: self.fn的输出，其输入是归一化后的x。
        """
        x = self.norm(x)  # 首先对输入x进行层归一化
        return self.fn(x, *args, **kwargs)  # 将归一化的数据传递给self.fn，并执行

class GELU(nn.Module):
    """
    GELU激活函数的封装。

    GELU (Gaussian Error Linear Unit) 是一种非线性激活函数，
    它被广泛用于自然语言处理和深度学习中的其他领域。
    这个函数结合了ReLU和正态分布的性质。
    """

    def forward(self, x):
        """
        在输入数据上应用GELU激活函数。

        参数:
            x (Tensor): 输入到激活函数的数据。

        返回:
            Tensor: 经过GELU激活函数处理后的数据。
        """
        return F.gelu(x)  # 使用PyTorch的函数实现GELU激活

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    """
    创建并返回一个二维卷积层。

    参数:
    in_channels (int): 输入特征图的通道数。
    out_channels (int): 输出特征图的通道数。
    kernel_size (int): 卷积核的大小，卷积核是正方形的。
    bias (bool, 可选): 是否在卷积层中添加偏置项。默认为 False。
    padding (int, 可选): 在输入特征图周围添加的零填充的层数。默认为 1。
    stride (int, 可选): 卷积的步长。默认为 1。

    返回:
    nn.Conv2d: 配置好的二维卷积层。
    """
    # 创建并返回一个二维卷积层，具体配置参数包括输入通道数、输出通道数、卷积核大小等。
    # padding 参数设置为卷积核大小的一半，确保输出特征图的大小与步长和填充有关，但如果步长为 1，则与输入相同。
    return nn.Conv2d(
        in_channels,  # 输入特征的通道数
        out_channels,  # 输出特征的通道数
        kernel_size,  # 卷积核的大小
        padding=(kernel_size // 2),  # 自动计算填充大小以保持特征图的空间尺寸
        bias=bias,  # 是否添加偏置项
        stride=stride  # 卷积的步长
    )


class FeedForward(nn.Module):
    """
    实现一个基于卷积的前馈网络模块，通常用于视觉Transformer结构中。
    这个模块使用1x1卷积扩展特征维度，然后通过3x3卷积在这个扩展的维度上进行处理，最后使用1x1卷积将特征维度降回原来的大小。

    参数:
        dim (int): 输入和输出特征的维度。
        mult (int): 特征维度扩展的倍数，默认为4。
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),  # 使用1x1卷积提升特征维度
            GELU(),  # 使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),  # 分组卷积处理，维持特征维度不变，增加特征的局部相关性
            GELU(),  # 再次使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),  # 使用1x1卷积降低特征维度回到原始大小
        )

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
        x (tensor): 输入特征，形状为 [b, h, w, c]，其中b是批次大小，h和w是空间维度，c是通道数。

        返回:
        out (tensor): 输出特征，形状与输入相同。
        """
        # 由于PyTorch的卷积期望的输入形状为[b, c, h, w]，需要将通道数从最后一个维度移到第二个维度
        out = self.net(x.permute(0, 3, 1, 2).contiguous())  # 调整输入张量的维度
        return out.permute(0, 2, 3, 1)  # 将输出张量的维度调整回[b, h, w, c]格式

class AttenBlock(nn.Module):
    """
    实现一个基于2D SSM 的 Attention Block，该块包含多个注意力和前馈网络层。
    每个块循环地执行自注意力和前馈网络操作。
    参数:
        dim (int): 特征维度，对应于每个输入/输出通道的数量。
        dim_head (int): 每个注意力头的维度。
        heads (int): 注意力机制的头数。
        num_blocks (int): 该层中重复模块的数量。
    """
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2,d_state = 16):

        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IFA(dim_2=dim,dim = dim,num_heads = heads,ffn_expansion_factor=2.66 ,bias=True, LayerNorm_type='WithBias'),
                SS2D(d_model=dim,dropout=0,d_state =d_state),   # 加入Mamba的SS2D模块，输入参数都为源代码指定参数
                PreNorm(dim, FeedForward(dim=dim))  # 预归一化层，后跟前馈网络,这一部分相当于LN+FFN
            ]))

    def forward(self, x, hw):
        """
        前向传播函数处理输入特征和光照特征。
        参数:
            new x (Tensor): 输入特征张量，形状为 [b, L, c]，其中 b 是批次大小，c 是通道数。
            old x (Tensor): 输入特征张量，形状为 [b, c, h, w]，其中 b 是批次大小，c 是通道数，h 和 w 是高度和宽度。
        返回:
            Tensor: 输出特征张量，形状为 [B, L, C]。
        """
        h, w = hw
        x = x.reshape(x.shape[0], h, w, x.shape[-1]) # b h, w, c
        x = x.permute(0, 3, 1, 2) # b, c, h, w
        for (trans,ss2d,ff) in self.blocks:
            y=trans(x, x).permute(0, 2, 3, 1)  # 调整张量维度以匹配预期的输入格式[b, h, w, c]
            #应用SS2D模块并进行残差连接
            x=ss2d(y)+ x.permute(0, 2, 3, 1)  #当我创建了一个类之后,如果继承nn并且自己定义了一个forward,那么nn会把hook挂到对象中,直接用对象(forward的参数)就能调用forward函数
            # print("经过ss2d的特征大小",x.shape)
            # 应用前馈网络并进行残差连接
            x = ff(x) + x# bhwc
            x = x.permute(0, 3, 1, 2) #bchw
        x = x.permute(0,2,3,1).view(x.shape[0], int(x.shape[2]*x.shape[3]), x.shape[1]) # [B, L, C]

        return x
    
'''Ref Paper: SUNet: Swin Transformer with UNet for Image Denoising (https://github.com/FanChiMao/SUNet)'''

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: None (or nn.LayerNorm)
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if norm_layer is not None:
            self.norm1 = norm_layer(dim)
        else:
            self.norm1 = None 
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if norm_layer is not None:
            self.norm2 = norm_layer(dim)
        else: 
            self.norm2 = None 
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1

            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: None (or nn.LayerNorm)
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
                
        if norm_layer is not None:
             self.norm = norm_layer(4 * dim)
        else:
            self.norm = None 



    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
                
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


# Dual up-sample
class UpSample(nn.Module):
    def __init__(self, input_resolution, in_channels, scale_factor):
        super(UpSample, self).__init__()
        self.input_resolution = input_resolution
        self.factor = scale_factor


        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2*in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels//2, in_channels//2, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))
        elif self.factor == 4:
            self.conv = nn.Conv2d(2*in_channels, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 16 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        """
        x: B, L = H*W, C
        """
        if type(self.input_resolution) == int:
            H = self.input_resolution
            W = self.input_resolution

        elif type(self.input_resolution) == tuple:
            H, W = self.input_resolution

        B, L, C = x.shape
        x = x.view(B, H, W, C)  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear
        out = self.conv(torch.cat([x_p, x_b], dim=1))
        out = out.permute(0, 2, 3, 1)  # B, H, W, C
        if self.factor == 2:
            out = out.view(B, -1, C // 2)

        
        return out


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: None (or nn.LayerNorm)
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=None, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            AttenBlock(dim=dim, dim_head=64, heads=num_heads, num_blocks=1, d_state = 16)
            for i in range(depth)])
        # self.blocks = nn.ModuleList([
        #     SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
        #                          num_heads=num_heads, window_size=window_size,
        #                          shift_size=0 if (i % 2 == 0) else window_size // 2,
        #                          mlp_ratio=mlp_ratio,
        #                          qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                          drop=drop, attn_drop=attn_drop,
        #                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #                          norm_layer=norm_layer)
            # for i in range(depth)])

        # patch merging layer
        if downsample is not None:
                self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, hw):
        for blk in self.blocks:
            x = blk(x, hw)
        if self.downsample is not None:
            # b, c = x.shape[0], x.shape[1]
            # x = x.reshape(b, c, -1).transpose(1,2) # B, L, C
            x = self.downsample(x)
            hw = (hw[0]//2, hw[1]//2)
        return x, hw

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: None (or nn.LayerNorm)
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=None, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint


        # build blocks
        self.blocks = nn.ModuleList([
            AttenBlock(dim=dim, dim_head=64, heads=num_heads, num_blocks=1, d_state=16)
            for i in range(depth)])
        
        # # build blocks
        # self.blocks = nn.ModuleList([
        #     SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
        #                          num_heads=num_heads, window_size=window_size,
        #                          shift_size=0 if (i % 2 == 0) else window_size // 2,
        #                          mlp_ratio=mlp_ratio,
        #                          qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                          drop=drop, attn_drop=attn_drop,
        #                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #                          norm_layer=norm_layer)
        #     for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = UpSample(input_resolution, in_channels=dim, scale_factor=2)
        else:
            self.upsample = None

    def forward(self, x, hw):
        for blk in self.blocks:
                x = blk(x, hw)

        if self.upsample is not None:
            # b, c = x.shape[0], x.shape[1]
            # x = x.reshape(b, c, -1).transpose(1,2) # B, L, C
            x = self.upsample(x)
            hw = ( int(hw[0]*2), int(hw[1]*2) )
        return x, hw


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        b, c, ph, pw = x.shape
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, (ph,pw)

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

