import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    这个类干什么用的？？？再次归一化？？？为什么要再次归一化？？？
    应该是对输入序列的归一化，在划分数据集时进行了一次归一化，那时的归一化针对的是整个训练集；
    在训练时把训练集划分为多个输入序列，这里是对当前的序列进行归一化，使得当前的序列符合正态分布
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):  # x(32,336,321)
        if mode == 'norm':      # 对单个序列进行归一化，
            # Instance Normalization：sklearn.preprocessing.StandardScaler()对输入的每一列单独进行归一化
            self._get_statistics(x)
            x = self._normalize(x)    # 把x减去均值然后再除以标准差（0均值，单位标准差）
        elif mode == 'denorm':  # 反归一化
            x = self._denormalize(x)  # (32,96,321) ==>
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev           # 乘以标准差
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean        # 加上均值
        return x