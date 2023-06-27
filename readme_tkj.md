# 1 说明
这个项目用于对光伏数据进行探索，预测，计划使用下列模型作为对比：
- Transformer
- Informer
- Autoformer
- FEDformer
- Pyraformer
- PatchTST
- FFTransformer

后续还可进一步完善！
可以考虑加入的模型：
- MICN


## 数据部分
使用两个数据集
- 一个是澳大利亚的数据集，分辨率为10min的一年数据，共51984条
- 第二个是中国河北的数据集，不到一年，分辨率为15min




# 2 如何添加新的模型
分以下步骤进行：
- 首先，把自己写的模型文件（例如`example_models.py`）放到 **models** 文件夹下，如果该模型还有一些子层的实现代码，将其放到 **layers** 文件夹并在模型文件中导入这些子层；
- 其次，修改 **exp** 文件夹下的 `exp_main.py` ，该文件是模型的训练和评估的地方，也就是使用模型的地方。
  - 需要添加一句 `from models import example_models.py` 来导入模型。
  - 需要在 `_build_model()` 里面的 model_dict 字典中添加当前新增模型的键值对。
- 最后，修改 `run_longExp.py` 里面的参数传递部分，可以添加一些和当前模型相关的参数
  - 这些参数会传给 exp_main.py ，然后传给要调用方法的init()方法。
  - 因此在编写模型文件时需要先想好需要使用到的参数，写在init()里面。对于已有的模型进行整合时如何确定其使用的参数？
    - 1 首先，在模型文件（例如 `FFTransformer.py`） 里面找到Model()类的 `__init__()` 方法，查看其使用的超参数，将其记录下来。
    - 2 然后，对照 `run.py` 里面的超参数，将两者不相同的超参数作为当前模型的超参数，在 `run.py` 里面隔开一个位置专门放置该模型对应的超参数。
    - 3 注意保持一些公共超参数的名称一致，例如 `n_model,n_head` 之类的共用参数。


FFTransformer.py 使用到的参数(使用的自注意力是ProbAttention)：
```
# data loader 
configs.freq                  # freq for time features encoding, options

# forecasting task
configs.seq_len         
configs.label_len     
configs.pred_len 
configs.enc_in
configs.dec_inc
configs.c_out

# model define
configs.d_model
configs.n_heads
configs.e_layers            # number of encoder layers
configs.d_layers            # number of decoder layers
configs.d_ff                # dimension of fcn
configs.factor              # attn factor（ProbAttention的一个参数）
configs.dropout
configs.embed               # time features encoding, options:[timeF, fixed, learned]
configs.activation          
configs.output_attention    # whether to output attention in ecoder
# 下面的参数都是FFTransformer独有的（其中的kernel_size和PatchTST重合了，建议将其设为默认值3）
configs.qk_ker              # Key/Query convolution kernel length for LogSparse Transformer
configs.v_conv              # Weather to apply ConvAttn for values (in addition to K/Q for LogSparseAttn
configs.top_keys            # eather to find top keys instead of queries in Informer
configs.kernel_size         # Kernel size for the 1DConv value embedding
configs.norm_out            # Whether to apply laynorm to outputs of Enc or Dec in FFTransformer
configs.num_decomp          # Number of wavelet decompositions for FFTransformer
configs.mlp_out             # Whether to apply MLP to GNN outputs

```




# 3 TODO
1. 修改 data_loader.py 的数据，添加对光伏数据的数据处理类；删除其他的数据集处理代码；
2. 尝试添加新的模型FFTransformer，并记录过程（注意处理自注意力和PatchTST的区别）；
3. 运行模型获取结果；
4. 搭建自己的模型的框架；
5. 添加训练时的损失函数变化图的绘制
6. 添加模型评价的一些指标，如技能评分（同时添加naive model）
7. 添加attention map可视化代码


# 运行脚本
定位到当前项目，然后运行类似于下面的命令：
```shell
sh ./scripts/Transformer.sh
```


