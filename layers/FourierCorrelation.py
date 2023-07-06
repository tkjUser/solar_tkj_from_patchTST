# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)  # 如果是 seq_len 固定为96的话，那么这里就固定是 modes=48 < 64
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))  # 48个，即[0,...,47]
        np.random.shuffle(index)  # 对index洗牌(打乱顺序)
        index = index[:modes]     # 选取modes个数
    else:
        index = list(range(0, modes))  # 顺序选取前modes个，FFT分解的结果貌似是从小到大排序的
    index.sort()  # 排列为升序
    return index  # 返回排为升序的随机取数


# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))  # TODO: 这个scale干啥的？
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)  # (32,8,64)*(8,64,64) ==> (32,8,64)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]  [batch,seq_len,head_num,each_head]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)        # (32,8,64,96)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)  # 在seq_len维度上做傅里叶变换(32,8,64,49)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)  # (32,8,64,49)，每个元素实部和虚部都是0
        for wi, i in enumerate(self.index):  # self.index是一个列表，其值从0递增到47，共48个值
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])  # out_ft(32,8,64,49) 这里还包含了padding操作
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # 反傅里叶 x(32,8,64,96)  # TODO： 这里应该把x还原为输入的形状(32,8,64,96) 作者说疏忽了：https://github.com/MAZiqing/FEDformer/issues/16
        return (x, None)  # TODO: 这里的None应该可以返回那个可学习的权重


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)    # 64
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)  # 48

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))  # 传统信号傅立叶逆变换会有一个scale的1/n的系数，n是sample数，这里我们就用了这个1 / (in_channels * out_channels) 来近似这个scale
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape  # B=32,L=144,H=8,E=64
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]  (32,8,64,144) Q来自解码器，K和V来自编码器
        xk = k.permute(0, 2, 3, 1)  # (32,8,64,96)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)  # 全零的(32,8,64,64)
        xq_ft = torch.fft.rfft(xq, dim=-1)  # x的Q的最后一个维度傅里叶变换(32,8,64,73)
        for i, j in enumerate(self.index_q):  # self.index_q是一个长度为64的列表，包含0~71的随机64个数
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]   # 这里i和j可能不同！！！
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)  # 全零的(32,8,64,48)
        xk_ft = torch.fft.rfft(xk, dim=-1)  # x的K的最后一个维度傅里叶变换(32,8,64,49)
        for i, j in enumerate(self.index_kv):  # self.index_q是一个长度为48的列表，包含0-47的48个整数
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]  # i和j相同，赋值运算

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))  # Q和K相乘 (32,8,64,64)*(32,8,64,48)=(32,8,64,48)
        if self.activation == 'tanh':  # TODO: 激活后的xqk_ft就是需要可视化的 attention map
            xqk_ft = xqk_ft.tanh()  # 激活，作者附录E4部分说了这里的xqk_ft可以作为频域cross attention，通过可视化来检查一些东西
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        # TODO: 这里 xk_ft_（作者使用的这个）可以替换为 xv_ft_ ，尝试一下。作者原话： https://github.com/MAZiqing/FEDformer/issues/34
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)        # QK和V相乘（这里的V使用xk_ft_代替，可能是因为K和V相等，而xk_ft_是K的傅里叶后的版本）
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)  # (32,8,64,64) TODO: 为什么要和weights1相乘？
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)  # 全零的(32,8,64,73)
        for i, j in enumerate(self.index_q):  # index长度为64，包含0~71的随机64个数
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]  # 这里似乎不是赋值操作，而是padding (32,8,64,64)==>(32,8,64,73)
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))  # 反傅里叶(32,8,64,144)
        xqk_ft_atten = torch.fft.irfft(xqk_ft / self.in_channels / self.out_channels, n=xq.size(-1))  # 对Attention Map进行傅里叶逆变换（自己写的）
        # return (out, None)
        return (out, xqk_ft_atten)




