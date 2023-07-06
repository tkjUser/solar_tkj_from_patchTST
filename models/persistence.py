# 来自FFTransformer
import torch
import torch.nn as nn


class Model(nn.Module):  # TODO: 对于这里的persistence，需要修改为 24-h persistence，即对于10min分辨率pred-144，对于15min分辨率pred-96
    """
    Persistence model
    persistence模型用t-1时刻的数据预测t时刻的数据(t-1时刻的数据是真实值，这种预测其实就是真实值的滞后一步)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.output_size = configs.c_out

    def forward(self, x_enc, *_, **__):    # (32,96,6)
        # outputs = x_enc[:, -1:, -self.output_size:]    # 选取最后一个步长的数据作为输出
        # outputs = outputs.repeat(1, self.pred_len, 1)

        # 这里的144是相对的，对于分辨率为15min是96，对于分辨率为10min是144
        interval = 144         # TODO: 这里需要确保 seq_len>=144
        assert self.pred_len >= interval, self.pred_len >= interval   # 日前预测确保窗口长度和预测长度不低于96
        y = torch.zeros((x_enc.shape[0], self.pred_len, x_enc.shape[2]))

        if self.pred_len > interval:    # 对于日前预测必须大于96
            y[:, 0:interval, :] = x_enc[:, -interval:, :]
            y[:, interval:self.pred_len, :] = x_enc[:, -interval:-interval+self.pred_len-interval, :]  # 这里需要确保pred_len不大于192
        else:                     # 最基础的也是等96
            y[:, 0:interval, :] = x_enc[:, -interval:, :]

        return y
