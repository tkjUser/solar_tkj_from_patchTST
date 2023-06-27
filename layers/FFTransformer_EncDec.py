# 来自FFTransformer
import torch
import torch.nn as nn
from layers.Functionality import MLPLayer


class EncoderLayer(nn.Module):
    def __init__(self, attention_fft, attention_trend, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention_fft = attention_fft        # 傅里叶注意力
        self.attention_trend = attention_trend    # 趋势注意力
        self.mlp_freq = MLPLayer(d_model=d_model, d_ff=d_ff, kernel_size=1, dropout=dropout, activation=activation)
        self.mlp_trend = MLPLayer(d_model=d_model, d_ff=d_ff, kernel_size=1, dropout=dropout, activation=activation)
        self.norm_freq = nn.LayerNorm(d_model)
        self.norm_trend = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, **kwargs):
        # FFT Attention
        x_freq, x_trend = x

        # Freq Stream  TODO: 对周期部分使用傅里叶自注意力
        freq_out, attn_freq = self.attention_fft(
            x_freq, x_freq, x_freq,
        )  # freq_out(32, 64, 512)
        x_freq = self.norm_freq(x_freq + self.dropout(freq_out))  # 残差连接并层归一化

        # Trend Stream  TODO: 对趋势部分使用Logsparse自注意力
        trend_out, attn_trend = self.attention_trend(
            x_trend, x_trend, x_trend,
            attn_mask=attn_mask,
        )
        x_trend = self.norm_trend(x_trend + self.dropout(trend_out))  # 残差连接并层归一化

        # MLP
        x_freq = self.mlp_freq(x_freq)     # torch.Size([32, 64, 512])
        x_trend = self.mlp_trend(x_trend)  # torch.Size([32, 64, 512])

        return [x_freq, x_trend], [attn_freq, attn_trend]


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_freq=None, norm_trend=None, conv_layers_freq=None, conv_layers_trend=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers_freq = nn.ModuleList(conv_layers_freq) if conv_layers_freq is not None else None
        self.conv_layers_trend = nn.ModuleList(conv_layers_trend) if conv_layers_trend is not None else None
        self.norm_freq = norm_freq
        self.norm_trend = norm_trend

    def forward(self, x, attn_mask=None, **kwargs):
        attns = []
        x_freq, x_trend = x
        if self.conv_layers_freq is not None:
            for i, (attn_layer, conv_layer_freq, conv_layer_trend) in enumerate(zip(self.attn_layers, self.conv_layers_freq, self.conv_layers_trend)):
                x, attns_i = attn_layer([x_freq, x_trend], attn_mask=attn_mask, **kwargs)
                x_freq, x_trend = x
                x_freq = conv_layer_freq(x_freq)
                x_trend = conv_layer_trend(x_trend)
                attns.append(attns_i)
            x_freq, x_trend, attns_i = self.attn_layers[-1](x_freq, x_trend, attn_mask=attn_mask, **kwargs)
            attns.append(attns_i)
        else:
            for i, attn_layer in enumerate(self.attn_layers):  # 编码器有两层
                x, attns_i = attn_layer([x_freq, x_trend], attn_mask=attn_mask, **kwargs)  # TODO: 编码器层的forward
                x_freq, x_trend = x  #
                attns.append(attns_i)

        if self.norm_freq is not None:
            x_freq = self.norm_freq(x_freq)  # 层归一化
        if self.norm_trend is not None:
            x_trend = self.norm_trend(x_trend)  # 层归一化

        return x_freq, x_trend, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_fft_attention, self_trend_attention, cross_fft_attention, cross_trend_attention, d_model,
                 d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()

        self.self_fft_attention = self_fft_attention
        self.self_trend_attention = self_trend_attention
        self.cross_fft_attention = cross_fft_attention
        self.cross_trend_attention = cross_trend_attention

        self.mlp_freq = MLPLayer(d_model=d_model, d_ff=d_ff, kernel_size=1, dropout=dropout, activation=activation)
        self.mlp_trend = MLPLayer(d_model=d_model, d_ff=d_ff, kernel_size=1, dropout=dropout, activation=activation)
        self.norm_freq1 = nn.LayerNorm(d_model)
        self.norm_freq2 = nn.LayerNorm(d_model)
        self.norm_trend1 = nn.LayerNorm(d_model)
        self.norm_trend2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross_freq, cross_trend, x_mask=None, cross_mask=None, **kwargs):
        x_freq, x_trend = x  # (32,54,512)

        # First Attn Freq Stream  周期注意力的计算
        freq_out, attn_sa_freq = self.self_fft_attention(
            x_freq, x_freq, x_freq
        )
        x_freq = self.norm_freq1(x_freq + self.dropout(freq_out))

        # First Attn Trend Stream  趋势注意力的计算
        trend_out, attn_sa_trend = self.self_trend_attention(
            x_trend, x_trend, x_trend,
            attn_mask=x_mask,
        )
        x_trend = self.norm_trend1(x_trend + self.dropout(trend_out))

        # Cross Attn Freq Stream  结合编码器输出的周期注意力，其中的cross_freq是编码器的输出（注意这里的K和V是cross_freq）
        freq_out, attn_cr_freq = self.cross_fft_attention(  # 这里使用的也是ProbAttention
            x_freq, cross_freq, cross_freq
        )
        x_freq = self.norm_freq2(x_freq + self.dropout(freq_out))

        # Cross Attn Trend Stream  结合编码器输出的趋势注意力，其中的cross_freq是编码器的输出（注意这里的K和V是cross_freq）
        trend_out, attn_cr_trend = self.cross_trend_attention(
            x_trend, cross_trend, cross_trend,
            attn_mask=cross_mask)
        x_trend = self.norm_trend2(x_trend + self.dropout(trend_out))

        # MLP:
        x_freq = self.mlp_freq(x_freq)
        x_trend = self.mlp_trend(x_trend)

        return [x_freq, x_trend], [attn_sa_freq, attn_sa_trend, attn_cr_freq, attn_cr_trend]


class Decoder(nn.Module):
    def __init__(self, attn_layers, norm_freq=None, norm_trend=None, mlp_out=None, out_projection=None):
        super(Decoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm_freq = norm_freq
        self.norm_trend = norm_trend
        self.mlp_out = mlp_out
        self.out_projection = out_projection

    def forward(self, x, cross_freq, cross_trend, x_mask=None, cross_mask=None, **kwargs):
        attn = []
        x_freq,  x_trend = x  # 带掩码的输入 x_freq(32,54,512) x_trend(32,54,512)
        for i, attn_layer in enumerate(self.attn_layers):  # 解码器仅有一层
            x, attns_i = attn_layer([x_freq, x_trend], cross_freq, cross_trend,
                                    x_mask=x_mask, cross_mask=cross_mask, **kwargs)  # TODO: 调用解码器层的forward方法
            x_freq, x_trend = x  # (32,54,512)
            attn.append(attns_i)

        if self.norm_freq is not None:
            x_freq = self.norm_freq(x_freq)
        if self.norm_trend is not None:
            x_trend = self.norm_trend(x_trend)

        if self.mlp_out is not None:
            x = self.mlp_out(x_freq + x_trend)
        else:
            x = x_freq + x_trend

        if self.out_projection is not None:
            x = self.out_projection(x)

        return x, attn
