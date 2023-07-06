import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple
import math
from functools import partial
from einops import rearrange, reduce, repeat
from torch import nn, einsum, diagonal
from math import log2, ceil
import pdb
from utils.masking import LocalMask
from layers.utils_FED import get_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiWaveletTransform(nn.Module):
    """
    1D multiwavelet block.
    """

    def __init__(self, ich=1, k=8, alpha=16, c=128,
                 nCZ=1, L=0, base='legendre', attention_dropout=0.1):
        super(MultiWaveletTransform, self).__init__()
        print('base', base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # B=32, L=96, H=8, E=64, D=64, S=96
        _, S, _, D = values.shape
        if L > S:  # 对应Q为解码器传入，K和V从编码器传入（解码器的cross attention）
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:  # 编码器
            values = values[:, :L, :, :]  # (32,96,8,64)不变
            keys = keys[:, :L, :, :]      # 同上
        values = values.view(B, L, -1)    # (32,96,8,64) ==> (32,96,512)
        # TODO： 为什么要把 values 变成这个形状？
        V = self.Lk0(values).view(B, L, self.c, -1)  # (32,96,512) => (32,96,1024) => (32,96,128,8)
        for i in range(self.nCZ):  # i=0，只循环一次
            V = self.MWT_CZ[i](V)  # FEB-W  V(32,96,128,8)
            if i < self.nCZ - 1:
                V = F.relu(V)

        V = self.Lk1(V.view(B, L, -1))   # V(32,96,128,8) ==> (32,96,1024) ==> (32,96,512)  前面将value改变形状，这里还原
        V = V.view(B, L, -1, D)          # (32,96,8,64)
        return (V.contiguous(), None)


class MultiWaveletCross(nn.Module):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64,
                 k=8, ich=512,
                 L=0,
                 base='legendre',
                 mode_select_method='random',
                 initializer=None, activation='tanh',
                 **kwargs):
        super(MultiWaveletCross, self).__init__()
        print('base', base)

        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.attn1 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn2 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn3 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn4 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape  # (B, N, H, E)  q(32,144,8,64)
        _, S, _, _ = k.shape  # (B, S, H, E)  k(32,96,8,64)

        q = q.view(q.shape[0], q.shape[1], -1)  # (32,144,512)
        k = k.view(k.shape[0], k.shape[1], -1)  # (32,96,512)
        v = v.view(v.shape[0], v.shape[1], -1)  # (32,96,512)
        q = self.Lq(q)                                          # 线性映射，形状不变
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)      # (32,144,64,8)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)      # (32,96,64,8)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)      # (32,96,64,8)

        if N > S:  # 解码器执行
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()    # (32,48,64,8)
            v = torch.cat([v, zeros], dim=1)       # (32,96,64,8)cat(32,48,64,8) ==> (32,144,64,8)
            k = torch.cat([k, zeros], dim=1)       # 同上
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]
        # 上面的代码对K，V进行0填充，此时Q，K，V的形状大小一样，为(32,144,64,8)
        ns = math.floor(np.log2(N))          # 7
        nl = pow(2, math.ceil(np.log2(N)))   # 256
        extra_q = q[:, 0:nl - N, :, :]       # (32,144,64,8) ==> (32,112,64,8)
        extra_k = k[:, 0:nl - N, :, :]       # (32,144,64,8) ==>(32,112,64,8)
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)       # (32,144,64,8)cat(32,112,64,8) ==> (32,256,64,8)
        k = torch.cat([k, extra_k], 1)       # (32,144,64,8)cat(32,112,64,8) ==> (32,256,64,8)
        v = torch.cat([v, extra_v], 1)

        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])  # 声明三个列表并指明列表项的类型
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose  对于周期L(例如L=96)，通过小波处理得到：Ud是高频分量，Us是低频分量
        for i in range(ns - self.L):  # i的范围[0,..,6]
            # print('q shape',q.shape)
            d, q = self.wavelet_transform(q)  # d,q(32,128,64,8) => (32,128,64,8) ... (32,2,64,8)
            Ud_q += [tuple([d, q])]   # 列表的列表项是一个元组，元组包含两个Tensor  每次循环给列表添加一个列表项
            Us_q += [d]               # 列表的列表项是一个Tensor
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)  # d,k(32,128,64,8) => (32,64,64,8) => (32,32,64,8) ... (32,2,64,8)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):  # i的范围[0,..,6]   计算傅里叶的自注意力    attn1,..,4 都是论文图中的FEA-f
            dk, sk = Ud_k[i], Us_k[i]  # 提取列表信息
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [self.attn1(dq[0], dk[0], dv[0], mask)[0] + self.attn2(dq[1], dk[1], dv[1], mask)[0]]  # 将分解后的高频和低频分量（Q，K，V）输入到FourierCrossAttentionW类，计算其forward方法
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]   # 相当于论文图中的 X'(L+1)

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):  # i范围 [6,5,...,0]
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)  # 连接 v(32,2,64,16)
            v = self.evenOdd(v)            # 给v赋值  v(32,256,64,8)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))  # v(32,144,512)
        return (v.contiguous(), None)

    def wavelet_transform(self, x):         # (32,256,64,8)
        xa = torch.cat([x[:, ::2, :, :],    # 在第二维度（时间步维度）每隔一个元素（步长为2）取一次数，得到(32,128,64,8)
                        x[:, 1::2, :, :],   # 在第二维度（时间步维度）从第2个元素起每隔一个元素（步长为2）取一次数（上面的相当于奇数，这个相当于偶数）
                        ], -1)              # 在最后一个维度连接两个矩阵 xa(32,128,64,16)
        d = torch.matmul(xa, self.ec_d)    # (32,128,64,16)*(16,8)  ==> d(32,128,64,8)
        s = torch.matmul(xa, self.ec_s)    # 同上
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)  (32,2,64,16)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)  # (32,2,64,16)*(16,8) ==> (32,2,64,8)
        x_o = torch.matmul(x, self.rc_o)  # 同上

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)  # 全零的(32,4,64,8)
        x[..., ::2, :, :] = x_e   # x的偶数索引（第1，3，...,对应索引0,2,4,...）位置赋值为x_e
        x[..., 1::2, :, :] = x_o  # x的奇数索引位置赋值为x_o
        return x


class FourierCrossAttentionW(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh',
                 mode_select_method='random'):
        super(FourierCrossAttentionW, self).__init__()
        print('corss fourier correlation used!')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def forward(self, q, k, v, mask):
        B, L, E, H = q.shape  # q(32,128,64,8)

        xq = q.permute(0, 3, 2, 1)    # size = [B, H, E, L]   xq(32,8,64,128)
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))              # 列表 [0,..,63] 64个列表项
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))  # 同上

        # Compute Fourier coefficients  计算傅里叶系数
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)  # (32,8,64,64)
        xq_ft = torch.fft.rfft(xq, dim=-1)   # (32,8,64,65)
        for i, j in enumerate(self.index_q):       # 选取其中的一部分
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)  # (32,8,64,64)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))  # 分别对Q和K进行傅里叶变换并选取部分频率，然后计算Q和K的乘积
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()     # 激活
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)   # 计算傅里叶变换后的Q 与 选取部分傅里叶频率的K 的乘积

        xqkvw = xqkv_ft  # TODO: 这里是qkv？v在哪？
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)  # (32,8,64,65)
        for i, j in enumerate(self.index_q):      # 给 out_ft 赋值
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # 逆傅里叶 out(32, 128, 64, 8)  TODO: 这里为什么要除以 in_channels 和 out_channels ?
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        # size = [B, L, H, E]
        return (out, None)


class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)  x(32,64,128,8)

        x = x.view(B, N, -1)    # (32,64,128,8) ==> (32,64,1024)
        x = x.permute(0, 2, 1)  # (32,1024,64)
        x_fft = torch.fft.rfft(x)    # (32,1024,33)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)  # 16
        # l = N//2+1
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)  # 全零的(32,1024,33)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])  # (32,1024,16)*(1024,1024,16)==>(32,1024,16) 这里相当于padding
        x = torch.fft.irfft(out_ft, n=N)  # 反傅里叶 (32,1024,33)==> (32,1024,64)
        x = x.permute(0, 2, 1).view(B, N, c, k)  # (32,64,128,8)
        return x


# ##
class MWT_CZ1d(nn.Module):
    def __init__(self,
                 k=3, alpha=64,
                 L=0, c=1,
                 base='legendre',
                 initializer=None,
                 **kwargs):
        super(MWT_CZ1d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)

        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):    #
        B, N, c, k = x.shape  # (B, N, k)  x(32,96,128,8)
        ns = math.floor(np.log2(N))                    # ns=6=log_2 96
        nl = pow(2, math.ceil(np.log2(N)))             # nl=128=2^7
        extra_x = x[:, 0:nl - N, :, :]                 # (32,32,128,8)  在第二维度取前32个值
        x = torch.cat([x, extra_x], 1)                 # (32,96,128,8)cat(32,32,128,8)=在时间步维度连接=>(32,128,128,8)
        Ud = torch.jit.annotate(List[Tensor], [])    # 传递函数，返回值为第二参数，用于提示TorchScript编译器 第二参数的类型（类型由第一参数定义）
        Us = torch.jit.annotate(List[Tensor], [])    # 其实就是规定了Ud和Us这两个列表的类型为张量列表（列表里面放张量）
        # decompose  使用三个FEB-f模块（A，B，C）分别对小波分解得到的高频部分、低频部分和剩余部分
        for i in range(ns - self.L):  # i取值[0,1,2]
            # print('x shape',x.shape)
            d, x = self.wavelet_transform(x)  # d(32,64,128,8)   x:(32,128,128,8)==>(32,64,128,8)
            Ud += [self.A(d) + self.B(x)]  # A是sparseKernelFT1d类的forward方法。 Ud列表，里面包含一个Tensor(32,64,128,8)，每次循环新增一个列表项
            Us += [self.C(d)]              # Us列表，里面包含一个Tensor(32,64,128,8)
        x = self.T0(x)  # coarsest scale transform  线性层 (32,16,128,8) =Linear=> (32,16,128,8)

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):  # range(2,-1,-1)  即[2,1,0]
            x = x + Us[i]                   # 相加，(32,16,128,8)
            x = torch.cat((x, Ud[i]), -1)   # i=2,(32,16,128,8)  i=1,(32,16,128,16)  i=0,(32,16,128,16)
            x = self.evenOdd(x)             # (32,128,128,8)
        x = x[:, :N, :, :]  # (32,128,128,8)==>(32,96,128,8)

        return x

    def wavelet_transform(self, x):  # x(32,128,128,8)
        xa = torch.cat([x[:, ::2, :, :],      # 在第二维度（时间步维度）每隔一个元素（步长为2）取一次数，得到(32,64,128,8)
                        x[:, 1::2, :, :],     # 在第二维度（时间步维度）从第2个元素起每隔一个元素（步长为2）取一次数（上面的相当于奇数，这个相当于偶数）
                        ], -1)  # 在最后一个维度连接两个矩阵 xa(32,64,128,16)
        d = torch.matmul(xa, self.ec_d)    # self.ec_d(16,8)  ==> d(32,64,128,8)
        s = torch.matmul(xa, self.ec_s)    # self.ec_s(16,8)  ==> s(32,64,128,8)
        return d, s

    def evenOdd(self, x):

        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x
