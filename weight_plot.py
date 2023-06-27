# LSTF-Linear 里面的代码，用于可视化模型的注意力分布
import torch
import numpy as np
import os 
import matplotlib.pyplot as plt

# model_name默认是./checkpoint/里面的文件名（也就是保存的模型）
model_name = ''  # 可以为空，表示绘制所有./checkpoint/里面的文件的权重
for root, dirs, files in os.walk("checkpoints"):
    for name in files:
        model_path = os.path.join(root, name)
        if model_name not in model_path:
            continue
        weights = torch.load(model_path, map_location=torch.device('cpu'))
        weights_list = {}  # TODO: 这里针对不同的模型需要设置不同的提取名称，下面的代码仅提取Dlinear权重
        weights_list['seasonal'] = weights['Linear_Seasonal.weight'].numpy()
        weights_list['trend'] = weights['Linear_Trend.weight'].numpy()

        save_root = 'weights_plot/%s'%root.split('/')[1]
        if not os.path.exists('weights_plot'):
            os.mkdir('weights_plot')
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        for w_name, weight in weights_list.items():
            fig, ax = plt.subplots()
            im = ax.imshow(weight, cmap='plasma_r')
            fig.colorbar(im, pad=0.03)
            plt.savefig(os.path.join(save_root, w_name + '.pdf'), dpi=500)
            plt.close()
