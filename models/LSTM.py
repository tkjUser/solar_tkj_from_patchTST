# 自己写的，普通的的LSTM，不使用编解码结构
import torch
import torch.nn as nn


class Model(nn.Module):
    """
        思想：
        利用lstm后接的线性层实现使用过去多步预测未来多步（非编解码器，非自回归预测）
        - 对于单变量输入，设置线性层的输出为多个实现多步预测；
        - 对于多变量输入，设置线性层的输入为特征数时相当于多变量的单步预测（有待验证）；
        - 对于多变量输入，设置线性层的输入为 特征数*预测步长 时相当于多变量的多步预测；
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len          # 假如96
        self.pred_len = configs.pred_len        # 假如48
        self.input_size = configs.input_size    # 假如输入特征个数=6
        self.hidden_size = configs.hidden_size  # LSTM隐藏层的单元个数128
        self.num_layers = configs.num_layers            # 假如2
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.pred_len * self.input_size)  # 这里就是把输出长度拉长为预测步长乘以特征数
        # self.linear = nn.Linear(self.hidden_size, self.input_size)  # TODO:这里设置的两个参数分别是隐层数和输出个数
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel] ==> [Input length, Batch, Channel] ==> (seq_length,batch_size,hidden_size)
        lstm_out, hidden = self.lstm(x.permute(1, 0, 2))  # 输入(32,96,48)  LSTM的输出lstm_out(96,32,128)
        output = lstm_out[-1, :, :]   # 只取最后时刻的隐藏状态作为全连接的输入
        # 全连接层输出预测值，并reshape为(batch_size, pred_len, input_size)的张量
        output = self.linear(output).view(-1, self.pred_len, self.input_size)  # 全连接层输出预测值

        return output  # [Batch, Output length, Channel]
