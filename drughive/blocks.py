import torch
import torch.nn as nn
import torch.nn.functional as F

from .gridutils import get_gaussian_kernel


INTERPOLATION = 'trilinear'

def activation(x):
    return F.gelu(x)

def norm(x, dim):
    return torch.sqrt(torch.sum(x ** 2, dim))

@torch.jit.script
def jit_normalize_weight(log_wn, weight):
    eps = 1e-5
    a = torch.exp(log_wn)  # weight norm param
    wn = torch.sqrt(torch.sum(weight ** 2, dim=[1, 2, 3, 4])).reshape(-1, 1, 1, 1, 1) + eps # norm of weight
    return a * weight / (wn)


class ConvLayerWeightNorm(nn.Conv3d):
    """A ConvLayer layer with weight normalization."""

    def __init__(self, c_in, c_out, kernel_size, stride=1, padding='same', dilation=1, bias=False, weight_norm=True):
        super(ConvLayerWeightNorm, self).__init__(c_in, c_out, kernel_size, stride, padding, bias)

        self.weight_norm = weight_norm
        self.log_wn = None
        if self.weight_norm:
            self.init_weight_norm()

        self.weight_n = self.normalize_weight()
        self.dilation = dilation


    def init_weight_norm(self):
        w = norm(self.weight, dim=[1, 2, 3, 4]).reshape(-1, 1, 1, 1, 1) + 1e-2
        self.log_wn = nn.Parameter(torch.log(w), requires_grad=True)


    def forward(self, x):
        self.weight_n = self.normalize_weight()
        return F.conv3d(x, self.weight_n, self.bias, self.stride, self.padding, self.dilation, self.groups)


    def normalize_weight(self):
        w = self.weight
        if self.weight_norm:
            w = jit_normalize_weight(self.log_wn, w)
        return w


class ConvLayer(ConvLayerWeightNorm):
    pass


class BNActConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
        super(BNActConv, self).__init__()
        stride = abs(stride)
        self.bn = nn.BatchNorm3d(c_in, momentum=0.05)
        self.conv = ConvLayer(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.bn(x)
        y = activation(x)
        y = self.conv(y)
        return y


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1):
        super(ConvBNAct, self).__init__()

        self.conv = ConvLayer(c_in, c_out, kernel_size, stride, padding='same', bias=False, weight_norm=False) # No weight_norm because BN follows
        self.bn = nn.BatchNorm3d(c_out, momentum=0.05)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return activation(y)


class FactorizedReduce(nn.Module):
    def __init__(self, c_in, c_out):
        super(FactorizedReduce, self).__init__()
        assert c_out % 2 == 0
        cout = max(1, c_out // 8)

        self.conv_1 = ConvLayer(c_in, cout, kernel_size=1, stride=2, padding=0, bias=True)
        self.conv_2 = ConvLayer(c_in, cout, kernel_size=1, stride=2, padding=0, bias=True)
        self.conv_3 = ConvLayer(c_in, cout, kernel_size=1, stride=2, padding=0, bias=True)
        self.conv_4 = ConvLayer(c_in, cout, kernel_size=1, stride=2, padding=0, bias=True)
        self.conv_5 = ConvLayer(c_in, cout, kernel_size=1, stride=2, padding=0, bias=True)
        self.conv_6 = ConvLayer(c_in, cout, kernel_size=1, stride=2, padding=0, bias=True)
        self.conv_7 = ConvLayer(c_in, cout, kernel_size=1, stride=2, padding=0, bias=True)
        
        if c_out < 8:
            self.conv_out = ConvLayer(8, c_out, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_8 = ConvLayer(c_in, cout, kernel_size=1, stride=2, padding=0, bias=True)
        else:
            self.conv_out = nn.Identity()
            self.conv_8 = ConvLayer(c_in, c_out - (7 * cout), kernel_size=1, stride=2, padding=0, bias=True)


    def forward(self, x):
        y = activation(x)
        conv1 = self.conv_1(y)
        conv2 = self.conv_2(y[:, :, 1:, 1:, 1:])
        conv3 = self.conv_3(y[:, :, :, 1:, 1:])
        conv4 = self.conv_4(y[:, :, 1:, :, 1:])
        conv5 = self.conv_5(y[:, :, 1:, 1:, :])
        conv6 = self.conv_6(y[:, :, 1:, :, :])
        conv7 = self.conv_7(y[:, :, :, 1:, :])
        conv8 = self.conv_8(y[:, :, :, :, 1:])
        y = torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8], dim=1)
        y = self.conv_out(y)
        return y


class Downsample(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super(Downsample, self).__init__()
        self.layers = FactorizedReduce(c_in, c_out)
    
    def forward(self, x):
        return self.layers(x)


class Upsample(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super(Upsample, self).__init__()
        self.channels_in = c_in
        self.channels_out = c_out
        self.layers = nn.Sequential()
        self.layers.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
        self.layers.append(ConvLayer(c_in, c_out, kernel_size=1))

    def forward(self, x):
        return self.layers(x)


class InputBlock(nn.Module):
    def __init__(self, c_in, c_out, **conv_kwargs):
        super(InputBlock, self).__init__()
        self.channels_in = c_in
        self.channels_out = c_out
        self.layers = nn.Sequential(ConvLayer(c_in, c_out, kernel_size=3, padding=1, bias=True, **conv_kwargs))

    def forward(self, x):
        return self.layers(x)


class OutputBlock(nn.Module):
    def __init__(self, c_in, c_out, **conv_kwargs):
        super(OutputBlock, self).__init__()
        self.channels_in = c_in
        self.channels_out = c_out
        self.layers = nn.Sequential(nn.ELU(), 
                                     ConvLayer(c_in, c_out, kernel_size=3, padding=1, bias=True, **conv_kwargs))

    def forward(self, x):
        return self.layers(x)


class SamplerBlock(nn.Module):
    def __init__(self, c_in, c_out, act=True, **conv_kwargs):
        super(SamplerBlock, self).__init__()
        self.channels_in = c_in
        self.channels_out = c_out
        self.act = act
        self.layers = nn.Sequential()
        if self.act:
            self.layers.append(nn.ELU())
        self.layers.append(ConvLayer(c_in, c_out, kernel_size=1, padding=0, bias=True))

    def forward(self, x):
        return self.layers(x)


class EncoderModule(nn.Module):
    def __init__(self, c_in, c_out, n_blocks, kernel_size=3, n_block_convs=2, downsample=True):
        super(EncoderModule, self).__init__()
        self.channels_in = c_in
        self.channels_out = c_out
        self.downsample = downsample
        self.blocks = nn.ModuleList()            
        for i in range(n_blocks):
            c_out = c_in if (i < n_blocks - 1) else self.channels_out
            ds = True if (i == (n_blocks - 1)) and self.downsample else False
            self.blocks.append(EncoderBlock(c_in, c_out, kernel_size, ds, n_block_convs))
            c_in = c_out
    
    def forward(self, x):
        for mod in self.blocks:
            x = mod(x)
        return x


class DecoderModule(nn.Module):
    def __init__(self, c_in, c_out, n_blocks, kernel_size=3, n_block_convs=2, upsample=True):
        super(DecoderModule, self).__init__()
        self.channels_in = c_in
        self.channels_out = c_out
        self.upsample = upsample
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            c_out = c_in if (i < n_blocks - 1) else self.channels_out
            up = True if (i == 0) and self.upsample else False
            self.blocks.append(DecoderBlock(c_in, c_out, kernel_size, up, n_block_convs))
            c_in = c_out

    def forward(self, x):
        for mod in self.blocks:
            x = mod(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, downsample=False, n_convs=2):
        super(EncoderBlock, self).__init__()
        self.channels_in = c_in
        self.channels_out = c_out
        self.downsample = downsample
        self.layers = nn.Sequential()
        for i in range(n_convs - 1):
            self.layers.append(BNActConv(c_in, c_in, kernel_size, padding=1))
        self.layers.append(BNActConv(c_in, c_out, kernel_size, padding=1))

        if downsample:
            self.skip = Downsample(c_in, c_out)
            self.layers.append(Downsample(c_out, c_out))
        elif c_in == c_out:
            self.skip = nn.Identity()
        else:
            self.skip = ConvLayer(c_in, c_out, kernel_size=1)

    def forward(self, x):
        return self.skip(x) + 0.1 * self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, upsample=False, n_convs=1):
        super(DecoderBlock, self).__init__()
        self.channels_in = c_in
        self.channels_out = c_out
        self.upsample = upsample

        self.layers = nn.Sequential()
        if upsample:
            self.layers.append(Upsample(c_in, c_out))
            self.skip = Upsample(c_in, c_out)
            c_in = c_out
        elif c_in == c_out:
            self.skip = nn.Identity()
        else:
            self.skip = ConvLayer(c_in, c_out, kernel_size=1)

        self.layers.append(nn.BatchNorm3d(c_in, momentum=0.05))
        for _ in range(n_convs):
            self.layers.append(ConvBNAct(c_in, c_out, stride=1, kernel_size=kernel_size))
            c_in = c_out
        self.layers.append(nn.BatchNorm3d(c_in, momentum=0.05))

    def forward(self, x):
        return self.skip(x) + 0.1 * self.layers(x)


class ConvAddBlock(nn.Module):
    def __init__(self, c_conv, c_in2):
        super(ConvAddBlock, self).__init__()
        assert c_conv == c_in2, f'Inputs must have same number of channels. Received {c_conv} and {c_in2}'
        self.conv = ConvLayer(c_conv, c_conv, kernel_size=1, stride=1, padding=0, bias=True)
        self.channels_in = c_conv + c_in2
        self.channels_out = c_conv
        
    def forward(self, xconv, x2):
        xconv = self.conv(xconv)
        y = xconv + x2
        return y


class ConcatConvBlock(nn.Module):
    def __init__(self, c_in1, c_in2):
        super(ConcatConvBlock, self).__init__()
        self.conv = ConvLayer(c_in1 + c_in2, c_in1, kernel_size=1, stride=1, padding=0, bias=True)
        self.channels_in = c_in1 + c_in2
        self.channels_out = c_in1

    def forward(self, x1, x2):
        y = torch.cat([x1, x2], dim=1)
        y = self.conv(y)
        return y
    

class EncoderMixBlock(ConvAddBlock):
    pass

class EncoderMixBlockProt(ConcatConvBlock):
    pass
    
class DecoderMixBlock(ConcatConvBlock):
    pass

class DecoderMixBlockProt(ConcatConvBlock):
    pass
      
    
class GaussConvLayer(nn.Conv3d):
    def __init__(self, var=0.7, trunc=1.5, resolution=0.5, n_channels=1, normalized=False):
        self.var=var
        self.trunc=trunc
        self.resolution=resolution
        self.n_channels=n_channels
        self.normalized = normalized
        self.init_kernel(normalized)
        
        super().__init__(in_channels=n_channels,
                         out_channels=n_channels, 
                         kernel_size=len(self.kernel), 
                         stride=1, 
                         padding='same', 
                         dilation=1, 
                         groups=n_channels, 
                         bias=False, 
                         padding_mode='zeros')
        
        self.update_kernel()

    def re_init(self, var=None, trunc=None, resolution=None, n_channels=None, normalized=None):
        '''Re-initializes GaussConvLayer with input parameters changed.'''
        if not hasattr(self, 'normalized'):
            self.normalized = False
        var = var if var is not None else self.var
        trunc = trunc if trunc is not None else self.trunc
        resolution = resolution if resolution is not None else self.resolution
        n_channels = n_channels if n_channels is not None else self.n_channels
        normalized = normalized if normalized is not None else self.normalized
        return GaussConvLayer(var, trunc, resolution, n_channels, normalized)

    def init_kernel(self, normalized=False):
        self.kernel = get_gaussian_kernel(self.var, self.trunc, self.resolution)
        if normalized:
            self.kernel /= self.kernel.sum()
        
    def update_kernel(self):
        self.weight.data.copy_(torch.from_numpy(self.kernel))
        for param in self.parameters():
            param.requires_grad = False
    

class MeanFilter3d(nn.Conv3d):
    def __init__(self, kernel_size, n_channels):
        self.n_channels=n_channels
        self.kernel_size = kernel_size
        
        super().__init__(in_channels=n_channels,
                         out_channels=n_channels, 
                         kernel_size=kernel_size, 
                         stride=1, 
                         padding='same', 
                         padding_mode='replicate',
                         groups=n_channels,
                         bias=False)
        
        self.init_kernel()
        for param in self.parameters():
            param.requires_grad = False
        
    def init_kernel(self):
        self.weight.data[:] = 1/self.weight.data.numel()
        self.weight.data.requires_grad = False