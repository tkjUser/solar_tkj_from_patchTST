from .data_loader import Dataset_solar_15min, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'solar': Dataset_solar_15min,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]  # 引入Dataset_Custom对象（仅仅给它换个名字，没有初始化对象），这是一个自定义的数据加载器
    timeenc = 0 if args.embed != 'timeF' else 1  # 1

    if flag == 'test':           # 训练和测试使用的都是同一个加载类： Dataset_Custom
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':         # 预测使用的加载类： Dataset_Pred
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:  # 训练集
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq   # h
    # 这里初始化 Dataset_Custom 对象，会执行该对象的__init__方法，在该方法里面有读取数据的代码，可以debug进去看看
    data_set = Data(  # 读取csv文件数据，划分数据集，并对数据集进行归一化
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,  # 1
        freq=freq
    )
    print(flag, len(data_set))  # test集：5156
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,  # 10
        drop_last=drop_last)
    return data_set, data_loader
