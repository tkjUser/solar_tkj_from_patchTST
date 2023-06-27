# 来自FFTransformer
import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse


def get_wt(x, num_decomp=4, sep_out=True):
    dwt_for = DWT1DForward(J=num_decomp, mode='symmetric', wave='db4').to(x.device)  # 一维离散小波变换，分解等级J为4，填充方式model为对称填充，db4是小波函数
    dwt_inv = DWT1DInverse(mode='symmetric', wave='db4').to(x.device)                # 一维离散小波逆重建
    approx, detail = dwt_for(x.permute(0, 2, 1))     # [A3, [D0, D1, D2, D3]]    执行小波分解 (32,8,64) =DWT=> approx(32,8,10)  detail一个列表，包含四个Tensor，分别是(32,8,35-21-14-10)
    coefs = [approx, *detail[::-1]]  # 列表，包含5个tensor，也就是approx, detail    这里的*应该是解包，把列表分为列表项的总和
    coefs_res = []
    sizes = []
    additional_zeros = []
    for coef in coefs[::-1][:-1]:  # a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序。  coefs[::-1][:-1]就是detail的四个Tensor
        params = [torch.zeros_like(coef).to(x.device), coef]  # 一个Tensor，形状和coef一样，但其值为全0
        if len(sizes) != 0:  # 每次反小波变换需要把之前的值放入（这里就是使用形状相同的0值的Tensor代替）
            additional_zeros = [torch.zeros(s).to(x.device) for s in sizes[::-1]]
            params += additional_zeros
        cr = dwt_inv((params[0], params[1:][::-1]))  # 反小波变换  (32,8,64)
        sizes.append(coef.shape)   # 记录当前的形状
        coefs_res.append(cr)
    params = [coefs[0], torch.zeros_like(coefs[0]).to(x.device)] + additional_zeros  # 列表，包含5个列表项，approx一个（非零值），detail四个（全0值）
    cr = dwt_inv((params[0], params[1:][::-1]))  # 反小波变换
    coefs_res.append(cr)
    coefs_res = coefs_res[::-1]  # 逆序
    x_freq = torch.stack(coefs_res[1:], -1)[:, :, :x.shape[1], :]  # (32,8,64,4)
    x_trend = coefs_res[0][..., None][:, :, :x.shape[1], :]        # (32,8,64,1)

    x_freq = x_freq.permute(0, 2, 1, 3).reshape(*x.shape[:2], -1)    # (32,64,32)      # Concatenate all the series
    x_trend = x_trend.permute(0, 2, 1, 3).reshape(*x.shape[:2], -1)  # (32,64,8)

    if sep_out:
        return x_freq, x_trend
    else:
        return torch.cat([x_freq, x_trend], -1)
