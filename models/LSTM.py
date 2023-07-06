import torch
import torch.nn as nn
from layers.LSTM_EncDec import Encoder, Decoder
from layers.Embed import DataEmbedding
import random


class Model(nn.Module):
    """
    LSTM in Encoder-Decoder
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_model = configs.d_model
        self.enc_layers = configs.e_layers
        self.dec_layers = configs.d_layers

        self.train_strat_lstm = configs.train_strat_lstm

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_size = configs.c_out
        assert configs.label_len >= 1

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, pos_embed=False)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, pos_embed=False)

        self.encoder = Encoder(d_model=self.d_model, num_layers=self.enc_layers, dropout=configs.dropout)
        self.decoder = Decoder(output_size=configs.c_out, d_model=self.d_model,
                               dropout=configs.dropout, num_layers=self.dec_layers)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, teacher_forcing_ratio=None, **_):
        if self.train_strat_lstm == 'mixed_teacher_forcing' and self.training:
            assert teacher_forcing_ratio is not None
        # TODO: 目前看，至少recursive的LSTM是可以使用的！
        target = x_dec[:, -self.pred_len:, -self.output_size:]  # (32,144,6) =选取后面的96步（全0掩码?）=> (32,96,6)  TODO: 这里的target可能是未来的值
        enc_out = self.enc_embedding(x_enc, x_mark_enc)         # (32,96,6) =嵌入？把嵌入后的x和x_mark相加再次嵌入？？=> (32,96,512)

        enc_out, enc_hid = self.encoder(enc_out)    # 输入到一个LSTM编码器层，输出运行结果和隐层状态

        if self.enc_layers != self.dec_layers:      # 编码器层比解码器层多的时候
            assert self.dec_layers <= self.enc_layers
            enc_hid = [hid[-self.dec_layers:, ...] for hid in enc_hid]

        dec_inp = x_dec[:, -(self.pred_len + 1), -self.output_size:]  # (32,144,6) =选取过去时间步的最后一行作为解码器的输入=> (32,6)
        dec_hid = enc_hid                                             # 编码器的隐层

        outputs = torch.zeros_like(target)                            # 全零的(32,96,6)

        if not self.training or self.train_strat_lstm == 'recursive':  # 自回归输出
            for t in range(self.pred_len):
                dec_out, dec_hid = self.decoder(dec_inp, dec_hid)     # 输入前一个步长，得到后一个步长的预测值
                outputs[:, t, :] = dec_out     # dec_out(32,6)  给outputs的第t个时间步赋予值，通过循环给outputs的每个值赋值
                dec_inp = dec_out              # 预测值作为输入迭代获取新的值
        else:
            if self.train_strat_lstm == 'mixed_teacher_forcing':      # 混合学习
                for t in range(self.pred_len):
                    dec_out, dec_hid = self.decoder(dec_inp, dec_hid)
                    outputs[:, t, :] = dec_out
                    if random.random() < teacher_forcing_ratio:  # 一定几率的监督学习  TODO: 这里监督学习是把真实数据输入，但没有真实数据啊
                        dec_inp = target[:, t, :]  # 使用未来真实值作为输入
                    else:                                        # 自回归
                        dec_inp = dec_out

        return outputs  # [B, L, D]
