import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size  # 25
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)  # 步长为1窗口大小为25的一维池化

    def forward(self, x):  # x(32,336,321)
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # (32,1,321) =repeat=> (32,12,321)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # 同上，只是上面的是重复第一行数据，这里是重复最后一行数据
        x = torch.cat([front, x, end], dim=1)  # (32,12,321) + (32,336,321) + (32,12,321) = (32,360,321)
        x = self.avg(x.permute(0, 2, 1))  # 对x进行一维平均池化，池化范围为25，步长为1  (32,360,321)=>(32,321,360)=>(32,321,336)
        x = x.permute(0, 2, 1)  # (32,321,336) ==> (32,336,321)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)  # moving_avg类的__init__

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # 调用moving_avg类的forward方法,运用一维平均池化提取数据的趋势特征trend
        res = x - moving_mean  # 剩余数据Remainder
        return res, moving_mean


class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # 336
        self.pred_len = configs.pred_len  # 96

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)  # series_decomp类的__init__方法：步长为1窗口大小为25的一维平均池化
        self.individual = configs.individual  # 0
        self.channels = configs.enc_in  # 321

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]   (32,336,321)
        seasonal_init, trend_init = self.decompsition(x)  # 调用series_decomp类的forward方法，返回剩余项和趋势项
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # (32,321,336)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:  # (32,321,336) ==> (32,321,96)  两个线性层分别预测两个值seasonal_init、trend_init
            seasonal_output = self.Linear_Seasonal(seasonal_init)  # Linear(in_features=336, out_features=96, bias=True)
            trend_output = self.Linear_Trend(trend_init)           # 同上

        x = seasonal_output + trend_output  # (32,321,96)
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]  (32,96,321)
