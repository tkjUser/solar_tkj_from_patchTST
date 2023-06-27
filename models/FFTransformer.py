import torch
import torch.nn as nn
from layers import FFTransformer_EncDec as FFT
from layers.FFT_SelfAttention import FFTAttention
from layers.SelfAttention_Family_FFTTransformer import LogSparseAttentionLayer, ProbAttention
from layers.Embed import DataEmbedding
from layers.WaveletTransform import get_wt
from layers.Functionality import MLPLayer


class Model(nn.Module):
    """
    FFTransformer Encoder-Decoder with Convolutional ProbSparse Attn for Trend and ProbSparse for Freq Strean
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.output_attention = configs.output_attention  # False
        self.num_decomp = configs.num_decomp              # 4

        self.seq_len = configs.seq_len       # 96
        self.label_len = configs.label_len   # 48
        self.pred_len = configs.pred_len     # 96
        # self.kernel_size = 3    # TODO: 这里把 kernel_size 设置注释掉，因为它和PatchTST的参数重名了，其值直接使用默认值3

        # Frequency Embeddings:
        self.enc_embeddingF = DataEmbedding(self.num_decomp * configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, pos_embed=False)  # Embed.py的DataEmbedding类的初始化，仅使用值嵌入（一维卷积）
        self.dec_embeddingF = DataEmbedding(self.num_decomp * configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, pos_embed=False)

        # Trend Embeddings:
        self.enc_embeddingT = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)  # Embed.py的DataEmbedding类的初始化，使用值嵌入（一维卷积）和时间嵌入（线性层）
        self.dec_embeddingT = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)

        self.encoder = FFT.Encoder(
            [
                FFT.EncoderLayer(
                    FFTAttention(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=False, context_zero=True),
                        configs.d_model, configs.n_heads
                    ),
                    LogSparseAttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=configs.top_keys),
                        d_model=configs.d_model, n_heads=configs.n_heads,
                        qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_freq=torch.nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            norm_trend=torch.nn.LayerNorm(configs.d_model) if configs.norm_out else None,
        )
        # Decoder
        self.decoder = FFT.Decoder(
            [
                FFT.DecoderLayer(
                    FFTAttention(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=False, context_zero=True),
                        configs.d_model, configs.n_heads
                    ),
                    LogSparseAttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=configs.top_keys),  # 这里掩码设为了True
                        d_model=configs.d_model, n_heads=configs.n_heads, qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    ),
                    FFTAttention(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=False, context_zero=True),
                        configs.d_model, configs.n_heads
                    ),
                    LogSparseAttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=configs.top_keys),
                        d_model=configs.d_model, n_heads=configs.n_heads, qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_freq=nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            norm_trend=nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            mlp_out=MLPLayer(d_model=configs.d_model, d_ff=configs.d_ff, kernel_size=1,
                             dropout=configs.dropout, activation=configs.activation) if configs.mlp_out else None,
            out_projection=nn.Linear(configs.d_model, configs.c_out),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, **_):
        # Wavelet Decomposition，两部分，分别是对编码器和解码器的输入的分解    x_enc(32,64,8)  x_mark_enc(32,64,5)  x_dec(32,54,8)  x_mark_dec(32,54,5)
        x_enc_freq, x_enc_trend = get_wt(x_enc, num_decomp=self.num_decomp)  # x_enc(32,64,8) ==> 周期(32,64,32)  趋势(32,64,8)
        x_dec_freq, x_dec_trend = get_wt(x_dec[:, :-self.pred_len, :], num_decomp=self.num_decomp)  # Remove PHs first（应该是指移除全0的填充值，分解用不上全0）  x_dec(32,48,8) ==> (32,48,32)  (32,48,8)

        # Add placeholders after decomposition: 添加占位符  解码器输入（分两部分，分别是小波分解的趋势部分和周期部分）
        dec_trend_place = torch.mean(x_dec_trend, 1).unsqueeze(1).repeat(1, self.pred_len, 1)  # (32,48,8) ==> (32,8) ==> (32,1,8) ==> (32,6,8)
        x_dec_trend = torch.cat([x_dec_trend, dec_trend_place], dim=1)     # (32,48,8)concat(32,6,8) ==> (32,54,8)
        # 周期部分
        dec_freq_place = torch.zeros([x_dec_freq.shape[0], self.pred_len, x_dec_freq.shape[2]], device=x_dec_freq.device)  # (32,6,32)
        x_dec_freq = torch.cat([x_dec_freq, dec_freq_place], dim=1)        # (32,48,32)concat(32,6,32) ==> (32,54,32)  前面是label_len=48的已知序列，后面是pred_len=6的全0数据

        # Embed the inputs:(对于趋势嵌入，应该是把趋势当作时间序列，嵌入了时间和位置和值；对于周期嵌入，当作频率处理有值嵌入和时间嵌入，没有硬位置嵌入)
        x_enc_freq = self.enc_embeddingF(x_enc_freq, x_mark_enc)  # x_enc_freq(32,64,32)，x_mark_enc(32,64,5) =conv1d=> (32,64,512)
        x_dec_freq = self.dec_embeddingF(x_dec_freq, x_mark_dec)  # x_dec_freq(32,54,32)，x_mark_dec(32,54,5) =conv1d=> (32,64,512) 这里仅对x_dec_freq进行一维卷积，而没有对x_mark_dec处理（作者没有使用位置编码）
        # 上面是对周期部分的编解码输入嵌入，下面是对趋势的编解码输入嵌入
        x_enc_trend = self.enc_embeddingT(x_enc_trend, x_mark_enc)  # (32,64,512) 这里使用了值嵌入(32,64,512)，位置嵌入(1,64,512)，时间嵌入(32,64,512)
        x_dec_trend = self.dec_embeddingT(x_dec_trend, x_mark_dec)  # (32,54,512) 同上，也是三种嵌入

        attns = []
        # TODO: 编码器的forward方法，同时输入嵌入后的周期部分和趋势部分
        enc_freq, enc_trend, a = self.encoder([x_enc_freq, x_enc_trend], attn_mask=enc_self_mask)
        attns.append(a)
        # TODO: 解码器的forward方法，解码器的输入包含编码器的输出，以及嵌入后的label_len+pred_len
        dec_out, a = self.decoder([x_dec_freq, x_dec_trend], enc_freq, enc_trend, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        attns.append(a)   # dec_out(32,54,1)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]  裁剪到预测长度 torch.Size([32, 6, 1])
