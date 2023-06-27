# 来自FFTransformer
import torch
import torch.fft
import torch.nn as nn


#  Here we use ProbSparse attention, but this could be changed if desirable
class FFTAttention(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, scale=None, **_):
        super(FFTAttention, self).__init__()

        self.scale = scale     # None
        # 自注意力Key和Value的维度
        d_keys = d_keys or (d_model // n_heads)         # 64
        d_values = d_values or (d_model // n_heads)     # 64

        self.query_projection = nn.Conv1d(d_model, d_keys * n_heads, kernel_size=3, stride=3)  # TODO： 这里为什么要使用一维卷积映射，且步长为3？后续可以尝试不使用这些卷积或者替换为线性层或者修改参数
        self.key_projection = nn.Conv1d(d_model, d_keys * n_heads, kernel_size=3, stride=3)
        self.value_projection = nn.Conv1d(d_model, d_values * n_heads, kernel_size=3, stride=3)
        self.out_projection = nn.Linear(d_values * n_heads, d_model * 2)
        self.n_heads = n_heads
        self.attn = attention

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape  # B=32 L=64
        _, S, _ = keys.shape     # S=64
        H = self.n_heads
        # fft：快速离散傅里叶变换, rfft：因为中心共轭对称，所以将共轭的那一部分去除，减少存储量,其意义都是去除那些共轭对称的值
        queries_fft = torch.fft.rfft(queries.permute(0, 2, 1))  # (32,64,512) =时间维度=> (32,512,64) ==> (32,512,33)
        keys_fft = torch.fft.rfft(keys.permute(0, 2, 1))        # (32,512,33)
        values_fft = torch.fft.rfft(values.permute(0, 2, 1))    # (32,512,33)

        L_fft = queries_fft.shape[-1]  # 33
        S_fft = keys_fft.shape[-1]     # 33
        # torch.fft.rfftfreq 计算实值信号在傅里叶变换之后的频率，仅返回正频率，传入参数L和S表示实值信号的长度（这里是64）
        freqs_L = torch.fft.rfftfreq(L).unsqueeze(0).to(queries_fft.device)  # (1,33)
        freqs_S = torch.fft.rfftfreq(S).unsqueeze(0).to(queries_fft.device)  # (1,33)

        # (BS, L, D) --> (BS, 3L, D)    i.e. perform 1dConv with kernel=stride=3 to obtain (BS, L, D)
        # 3L corresponds to the real and imaginary components + the frequency values (as some sort of positional enc). 实部和虚部+频率值
        queries_fft = torch.stack([queries_fft.real, queries_fft.imag, freqs_L.unsqueeze(0).expand(queries_fft.shape)], -1)  # (32,512,33,3)
        queries_fft = queries_fft.reshape(B, queries_fft.shape[1], -1)    # (32,512,33,3) ==> (32,512,99)
        queries_fft = self.query_projection(queries_fft).permute(0, 2, 1).view(B, L_fft, H, -1)  # (32,512,99) =conv1d=> (32,512,33) ==> (32,33,512) ==> (32,33,8,64)

        keys_fft = torch.stack([keys_fft.real, keys_fft.imag, freqs_S.unsqueeze(0).expand(keys_fft.shape)], -1)
        keys_fft = keys_fft.reshape(B, keys_fft.shape[1], -1)
        keys_fft = self.key_projection(keys_fft).permute(0, 2, 1).view(B, S_fft, H, -1)  # (32,33,8,64)

        values_fft = torch.stack([values_fft.real, values_fft.imag, freqs_S.unsqueeze(0).expand(values_fft.shape)], -1)
        values_fft = values_fft.reshape(B, values_fft.shape[1], -1)
        values_fft = self.value_projection(values_fft).permute(0, 2, 1).view(B, S_fft, H, -1)  # (32,33,8,64)

        V, attn = self.attn(        # TODO: 调用ProbAttention的forward函数
            queries_fft, keys_fft, values_fft,
            attn_mask=None
        )  # V[32, 8, 33, 64]  attn为None（因为没有设置返回注意力）
        V = V.transpose(2, 1)  # (32, 8, 33, 64) ==> (32, 33, 8, 64)
        V = V.contiguous().view(B, L_fft, -1)  # torch.Size([32, 33, 512])

        V = self.out_projection(V)   # 线性层映射 [32, 33, 512] ==> torch.Size([32, 33, 1024])
        V = V.view(B, L_fft, -1, 2)  # [32, 33, 1024] ==> [32, 33, 512, 2]

        V = torch.complex(V[..., 0], V[..., 1]).permute(0, 2, 1)  # 创建复数张量torch.Size([32, 33, 512]) ==> (32,512,33)

        V = torch.fft.irfft(V, n=L).permute(0, 2, 1)  # 逆傅里叶变换，(32,512,33) ==> (32, 512, 64) ==> (32,64,512)

        return V.contiguous(), attn

