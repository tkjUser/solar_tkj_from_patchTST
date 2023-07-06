import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.SelfAttention_Family import FullAttention


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):  # x(32,96,6)
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)  # (32,12,6)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)   # (32,11,6)
        x = torch.cat([front, x, end], dim=1)  # (32,119,6) 在原始数据的左右填充，使得平滑前后的维度不变
        x = self.avg(x.permute(0, 2, 1))  # (32,6,96)
        x = x.permute(0, 2, 1)            # (32,96,6)
        return x       # 在当前平均核下的趋势部分


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]  # kernel_size为列表，里面每个列表项对应一个卷积核，步长固定为1
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:   # 这里只使用了一次循环
            moving_avg = func(x)    # kernel大小不同的趋势季节分解结构(32,96,6)
            moving_mean.append(moving_avg.unsqueeze(-1))  # moving_mean包含多个平滑核对应的趋势，组成一个列表
        moving_mean = torch.cat(moving_mean, dim=-1)  # (32,96,6,n)，其中n是平滑核的个数，即moving_avg参数的值
        # 对x(32,96,6)先伸展(32,96,6,1)然后线性映射为(32,96,6,n),然后对x的最后一个维度进行Softmax，再与moving_mean相乘(维度不变)，然后再在最后一个维度累加得到(32,96,6)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)  # TODO： 为什么要将x和提取趋势相加？
        res = x - moving_mean   # 周期部分
        return res, moving_mean 


class FourierDecomp(nn.Module):
    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):  # x(32,96,512)
        new_x, attn = self.attention(  # AutoCorrelation.py的AutoCorrelationLayer的forward方法
            x, x, x,
            attn_mask=attn_mask
        )  # 里面对x分为Q，K,V，并对它们进行了映射，对Q使用了傅里叶块处理（Q，K，V相同，所以只需要处理一个）
        x = x + self.dropout(new_x)  # 残差连接 (32,96,512)
        x, _ = self.decomp1(x)       # FiXME: MOE Decomp  这里丢弃了分解的趋势部分！
        y = x    # (32,96,512)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 使用一维卷积，激活，dropout (32,2048,96)    Conv1d(512, 2048, kernel_size=(1,), stride=(1,), bias=False)
        y = self.dropout(self.conv2(y).transpose(-1, 1))                   # (32,96,512)    Conv1d(2048, 512, kernel_size=(1,), stride=(1,), bias=False)
        res, _ = self.decomp2(x + y)  # MOE Decomp  残差连接并分解, 丢弃了分解的趋势部分！
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:  # 循环使用编码器层（两层）
                x, attn = attn_layer(x, attn_mask=attn_mask)   # EncoderLayer的forward方法
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)   # TODO: 层归一化的位置不对啊，Transformer原文不是在这里，而是 “add+norm”  这里好像是在编码器层之间加了层归一化

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(  # 对应解码器的Frequency Enhanced Block并残差连接 x(32,144,512)
            x, x, x,
            attn_mask=x_mask
        )[0])  # AutoCorrelation.py的AutoCorrelationLayer类的forward方法

        x, trend1 = self.decomp1(x)
        # x = x + self.dropout(self.cross_attention(  # 解码器的cross attention  并残差连接
        #     x, cross, cross,
        #     attn_mask=cross_mask
        # )[0])  # TODO： 这里使用[0]仅返回了第一个值，第二个注意力的值没有返回
        a, atten_de = self.cross_attention(  # 解码器的cross attention  并残差连接
            x, cross, cross,
            attn_mask=cross_mask
        )  # TODO： 这里修改了原本的代码
        x = x + self.dropout(a)

        x, trend2 = self.decomp2(x)  # MOE Decomp
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 一维卷积，激活，dropout
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)  # x,trend3(32,144,512)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)  # (32,144,512)=conv1d=>(32,144,6)
        return x, residual_trend, atten_de


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:  # 解码器仅有一层解码器层
            x, residual_trend, atten_de = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)  # DecoderLayer的forward
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)  # 层归一化

        if self.projection is not None:
            x = self.projection(x)    # (32,144,512) ==> (32,144,6)
        return x, trend, atten_de
