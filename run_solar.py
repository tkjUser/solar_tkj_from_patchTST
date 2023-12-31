# 用于运行光伏数据集的启动文件，里面的参数修改为合适光伏数据的大小
import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Transformer family for Solar Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2023, help='random seed')

# basic config
# the following arguments are required: --is_training, --model_id, --model, --data
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test_solar_5', help='task id')
# 设置使用的模型
parser.add_argument('--model', type=str, default='FEDformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')  # PatchTST  FFTransformer  LSTM
# 设置读取数据的位置 data loader
parser.add_argument('--data', type=str,  default='custom', help='dataset type')  # solar 或 custom
parser.add_argument('--root_path', type=str, default='dataset/solar_Australasian', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='data_36_resample2_51984.csv', help='data file')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='ac_power', help='target feature in S or MS task')  # TODO: 默认值为OT，改为ac_power
parser.add_argument('--freq', type=str, default='t',                                                  # TODO: 时间编码可进一步精确
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
# for pred_len in 96 192 336 720
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# ======================================================================================================================

# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.2, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')  # 这个应该是控制是否使用Instance Normalization
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')  # PatchTST还有分解，后续可以看一下
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# FFTransformer
parser.add_argument('--qk_ker', type=int, default=4,
                    help='Key/Query convolution kernel length for LogSparse Transformer')
parser.add_argument('--v_conv', type=int, default=0,
                    help='Weather to apply ConvAttn for values (in addition to K/Q for LogSparseAttn')
parser.add_argument('--top_keys', type=int, default=0, help='Weather to find top keys instead of queries in Informer')
parser.add_argument('--norm_out', type=int, default=1,
                    help='Whether to apply laynorm to outputs of Enc or Dec in FFTransformer')
parser.add_argument('--num_decomp', type=int, default=4, help='Number of wavelet decompositions for FFTransformer')
parser.add_argument('--mlp_out', type=int, default=0, help='Whether to apply MLP to GNN outputs.')

# LSTM
parser.add_argument('--train_strat_lstm', type=str, default='recursive',
                    help='The training strategy to use for the LSTM model. recursive or mixed_teacher_forcing')

# FEDformer  supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')  # 多重小波的小波基
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

# ======================================================================================================================
# Formers   TODO:这里选择位置编码的方式  这个选择在实现上是怎么样的？ Embed.py里面实现  是否是模型通用的？ 是
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
parser.add_argument('--c_out', type=int, default=6, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')  # 默认是128而不是512？
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')         # 默认是16而不是8
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')       # 这里默认是256而不是2048
parser.add_argument('--moving_avg', type=int, default=[12, 24], help='window size of moving average')   # TODO: 注意，这里的moving_avg是Autoformer(对其为25)和FEDformer(对其为[12,24,48])共用的参数，对FEDformer来说其可以是一个列表，即多个分解
parser.add_argument('--factor', type=int, default=3, help='attn factor')            # 原本为1，修改为3(概率稀疏自注意力的采样因子)
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')  # TODO：0.2是不是有些大？设为0.1吧
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder', default=False)  # TODO: 需要可视化attention map时可用
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
# parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')   # pycharm调试界面只支持主进程
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')             # TODO：原本值为100？不是10？
# TODO: batch_size可以修改，提升运行速度
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')  # 原本为TST，修为type1
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--reverse', action='store_true', help='weather use denormalization', default=True)  # 自己添加的参数，控制是否反归一化

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=1, help='gpu')        # TODO: 修改以改变gpu的使用（防止gpu被占用的问题）
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)        # Python函数，初始化随机数生成器，指定起始值为fix_seed，保证每次运行代码时得到相同的随机数序列
torch.manual_seed(fix_seed)  # PyTorch函数，用于为CPU或GPU设置生成的随机数的种子值，方便复现实验（如果使用多个GPU进行训练，可能不足以保证结果的确定性）
np.random.seed(fix_seed)     # Numpy函数，初始化随机数生成器


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
# 不执行的代码，仅使用单个GPU
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main  # 仅仅是给这个引入的对象换个名字而已，啥也没干

if args.is_training:
    for ii in range(args.itr):  # itr=1，仅运行一次
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii)

        exp = Exp(args)  # 初始化对象，执行了父类的__init__方法，里面确定了使用的模型并调用了模型初始化方法
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)  # 训练模型时并未计算模型的指标

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)   # 评估模型并计算指标

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:  # 加载已有模型进行评估或预测
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des,
                                                                                                  ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
