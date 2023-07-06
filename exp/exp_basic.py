import os
import torch
import numpy as np
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, FFTransformer, LSTM, \
    persistence, FEDformer

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'FFTransformer': FFTransformer,
            'LSTM': LSTM,
            'persistence': persistence,
            'FEDformer': FEDformer,
        }
        self.device = self._acquire_device()  # 指定使用的GPU
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:  # 参数设为了False，可修改
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
