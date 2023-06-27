__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len  # 16
        self.stride = stride  # 8
        self.padding_patch = padding_patch  # 'end'
        patch_num = int((context_window - patch_len) / stride + 1)  # 41
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d(
                (0, stride))  # 通过复制输入边界来填充输入张量,如果是2元组，则使用(padding_left,padding_right)表示左右两边填充的个数
            patch_num += 1

        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num  # 5376 = 128*42
        self.n_vars = c_in  # 321
        self.pretrain_head = pretrain_head  # False
        self.head_type = head_type  # 'flatten'
        self.individual = individual  # 0

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in,
                                                  fc_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)  # (32,321,336) ==> (32,336,321)
            # Instance Normalization
            z = self.revin_layer(z, 'norm')  # Z-Score归一化，零均值单位标准差（使得当前序列符合正态分布）
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)  # 调用torch的函数进行尾部填充 (32,336,321)==>(32，321，344)
        # 先把当前序列长度336填充到合适划分的长度344，然后使用下面的代码进行划分patch  (32，321，344) ==> (32, 321, 42, 16)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # (32,321,16,42)                        # z: [bs x nvars x patch_len x patch_num]

        # model
        # 对应论文图1(b)的橙黄色的”Transformer Encoder“部分
        z = self.backbone(z)  # 执行 TSTiEncoder 的forward函数               # z: [bs x nvars x d_model x patch_num]
        # 对应论文图1(b)的绿色的”Flatten + Linear Head“      调用Flatten_Head类的forward方法
        z = self.head(z)      # z: [bs x nvars x target_window]  # (32,321,128,42) ==> (32,321,96)

        # denorm
        if self.revin:  # 解除对序列的归一化
            z = z.permute(0, 2, 1)               # (32,321,96) ==> (32,96,321)
            z = self.revin_layer(z, 'denorm')    # 调用RevIN类的forward方法，反归一化
            z = z.permute(0, 2, 1)               # (32,96,321) ==> (32,321,96)
        return z  # (32,321,96)

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)  # 合并最后两个维度的数据 (32,321,128,42) ==> [32,321,5376]
            x = self.linear(x)   # Linear(in_features=5376, out_features=96, bias=True)   [32,321,96]
            x = self.dropout(x)
        return x  # [32,321,96]


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        """ 编码器的参数设置，包括基本的变量和编码器层的初始化 """

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding  位置编码编的是当前序列的所有patch（42个）的相对位置，每个patch被映射为128的向量
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)  # 生成大小为(42,128)，服从均匀分布的二维矩阵

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]
        #
        n_vars = x.shape[1]  # 321
        # Input encoding
        x = x.permute(0, 1, 3, 2)    # x: [bs x nvars x patch_num x patch_len]  (32, 321, 42, 16)
        x = self.W_P(x)              # 线性映射 x: [bs x nvars x patch_num x d_model] 将patch_len=16映射到128的向量
        # u： torch.Size([10272, 42, 128]) 将批次数与变量数相乘合并
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
        # 实现了论文图1(b)里面的 Projection + Position Embedding 也就是把位置信息放到输入里面，得到的结果作为编码器输入
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]  torch.Size([10272, 42, 128])

        # Encoder  z(10272,42,128) ==> (32,321,42,128) ==> (32,321,128,42)
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]   调用TSTEncoder的forward方法
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z  # (32,321,128,42)

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src  # (10272,42,128)
        scores = None
        if self.res_attention:  # 是否返回注意力
            for mod in self.layers:
                # mod是编码器的层数，这里设置了三个编码器，因此会依次把数据输入进行运行，这里会执行TSTEncoderLayer的forward代码
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        ## Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        # Multi-Head attention
        if self.res_attention:  # 调用 MultiheadAttention 的forward方法
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:  # 是否保存注意力，不保存
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # 残差连接 Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)        # 进行批归一化

        # Feed-forward sublayer
        if self.pre_norm:  # 不执行
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward  前馈网络层  (10272,42,128)
        src2 = self.ff(src)  # 两个线性层，Linear1(in=128,out=256),Gelu激活，Dropout，Linear2(in=256,out=128)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # 残差连接  Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)        # 批归一化

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        # Q(10272,42,128)
        bs = Q.size(0)  # 10272
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)     W_Q,W_K,W_V分别是三个线性层，输入维度和输出维度都是128
        # (10272,42,128)=view=>(10272,42,16,8)=trans=>(10272,16,42,8)        q_s: [bs x n_heads x max_q_len x d_k]
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        # (10272,42,128)=view=>(10272,42,16,8)=trans=>(10272,16,8,42)   这里K的形状与Q和V不同,其实就是K的转置矩阵K^T
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)  # k_s: [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        # (10272,42,128)=view=>(10272,42,16,8)=trans=>(10272,16,42,8)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s: [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:  # 执行 _ScaledDotProductAttention 的forward函数
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v]    (10272,16,42,8)
        # attn: [bs x n_heads x q_len x q_len]        (10272,16,42,42)
        # scores: [bs x n_heads x max_q_len x q_len]  (10272,16,42,42)

        # back to the original inputs dimensions   还原 output(10272,16,42,8) ==> (10272,42,128)
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)  # 一个线性层（128）加上一个Dropout

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        ''' 只执行了自注意力的计算，残差连接和批归一化不在这里实现，在外面的代码
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]    (10272,16,42,8)
            k               : [bs x n_heads x d_k x seq_len]      (10272,16,8,42)
            v               : [bs x n_heads x seq_len x d_v]      (10272,16,42,8)
            prev            : [bs x n_heads x q_len x seq_len]    None
            key_padding_mask: [bs x seq_len]                      None
            attn_mask       : [1 x seq_len x seq_len]             None
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''
        # TODO: 这里只把Q和K的转置相乘，并没有乘以V啊，怎么得出的自注意力？？？ 后面把这个自注意力和V相乘了，得到输出
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence  (10272,16,42,42)
        attn_scores = torch.matmul(q, k) * self.scale  # 这里相当于除以根号d_k  attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev  # 不执行

        # Attention mask (optional)  不执行
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)  不执行
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights            softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights 使用给定的注意力权重与V相乘得到新的注意力权值
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:  # 是否返回注意力
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
