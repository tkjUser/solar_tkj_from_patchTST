import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# model_name默认是./checkpoint/里面的文件名（也就是保存的模型）
root = 'test_solar_2_Transformer_custom_ftMS_sl96_ll48_pl96_dm512_nh8_el3_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'
name = 'tensor_list.pth'
model_path = os.path.join(root, name)
weights = torch.load(model_path, map_location=torch.device('cpu'))
weights_list = []
weights_list.append(weights[0].numpy().squeeze())
weights_list.append(weights[1].numpy().squeeze())
weights_list.append(weights[2].numpy().squeeze())

save_root = 'weights_plot_result'
if not os.path.exists(save_root):
    os.mkdir(save_root)

# 绘制单个head的图
for weight in range(len(weights_list)):
    fig, ax = plt.subplots()
    im = ax.imshow(weights_list[weight][0, :, :], cmap='plasma_r')
    fig.colorbar(im, pad=0.03)
    plt.savefig(os.path.join(save_root, 'pic_single_' + str(weight) + '.pdf'), dpi=500)
    plt.close()

# 绘制多个head的平均的图
for weight in range(len(weights_list)):
    fig, ax = plt.subplots()
    im = ax.imshow(np.mean(weights_list[weight], axis=0), cmap='plasma_r')
    fig.colorbar(im, pad=0.03)
    plt.savefig(os.path.join(save_root, 'pic_mean_' + str(weight) + '.pdf'), dpi=500)
    plt.close()
