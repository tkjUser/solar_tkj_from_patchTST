from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, FFTransformer, LSTM, \
    persistence, FEDformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, plot_loss
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'      # 添加这个定位问题  参考： https://blog.csdn.net/qq_38308388/article/details/131046609

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)  # 这里执行了父类（Exp_Basic）的__init__方法(执行多条语句，最后一条是调用_build_model)，
        # 而其父类初始化时调用了_build_model()方法，该方法被当前类重写了，因此执行的还是当前类的 _build_model() 方法

    def _build_model(self):
        # model_dict = {
        #     'Autoformer': Autoformer,
        #     'Transformer': Transformer,
        #     'Informer': Informer,
        #     'DLinear': DLinear,
        #     'NLinear': NLinear,
        #     'Linear': Linear,
        #     'PatchTST': PatchTST,
        #     'FFTransformer': FFTransformer,
        #     'LSTM': LSTM,
        #     'persistence': persistence,
        #     'FEDformer': FEDformer,
        # }
        # model = model_dict[self.args.model].Model(self.args).float()  # TODO:这里执行了当前model的初始化方法
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # 设为验证模式，在评估模式下，批归一化和Dropout的行为会发生变化。例如，批归一化层将使用整个训练数据集的均值和方差，而不是每个小批量数据的均值和方差。Dropout 层不再随机丢弃神经元。
        self.model.eval()
        with torch.no_grad():  # 临时将所有requires_grad标志设置为False，告诉PyTorch在前向传递期间不要跟踪或更新梯度，节省内存并加快计算速度
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)  # (32,336,321)
                batch_y = batch_y.float()                  # (32,144,321)

                batch_x_mark = batch_x_mark.float().to(self.device)  # (32,336,4)
                batch_y_mark = batch_y_mark.float().to(self.device)  # (32,144,4)

                # decoder input  (32,96,321)==>(32,144,321)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # (32,96,321)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)  # 训练模型
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()  # 将模型设置为训练模式，在训练模式下，启用批量归一化（Batch Normalization）和 Dropout
        return total_loss

    def train(self, setting):
        # 划分数据集
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if 'persistence' in self.args.model:       # Check if the model is the persistence model
            criterion = self._select_criterion()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.test(setting, test=0)  # 对于persistence模型不需要训练，也不需要加载模型，直接调用test()评估其性能即可

            print('vali_loss: ', vali_loss)
            print('test_loss: ', test_loss)
            assert False              # 这个好像是用来结束程序的

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)   # 1131
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # 根据不同的策略来动态地改变学习率
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        total_train_loss = []    # 把所有epoch的平均损失放到一个列表中，方便绘图
        total_val_loss = []      # 把所有epoch的验证集损失放到一个列表中
        for epoch in range(self.args.train_epochs):  # 100
            iter_count = 0
            teacher_forcing_ratio = 0.8    # For LSTM Enc-Dec training (not used for others).
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # 这里执行代码时会自动调用Dataloader，而其中定义了n_works大于0，也就是使用多线程运行，这会导致无法调试，必须把n_works=0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                # 输入变量
                batch_x = batch_x.float().to(self.device)  # batch_x(32,336,6)
                batch_y = batch_y.float().to(self.device)  # batch_y(32,144,6)
                batch_x_mark = batch_x_mark.float().to(self.device)  # batch_x_mark(32,336,5)  这里的4是时间戳分开形成的四列
                batch_y_mark = batch_y_mark.float().to(self.device)  # batch_y_mark(32,144,5)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # (32,96,6)
                # 解码器输入(带掩码)：拼接batch_y(32,0:48,6)和全0的(32,96,6)得到(32,144,6)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:  # use automatic mixed precision training
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:  # 执行线形模型或者PatchTST（仅输入batch_x，没有其他输入！！！为什么？？？）
                    if 'Linear' in self.args.model or 'TST' in self.args.model:  # (32,336,321) ==> (32,96,321)
                        # TODO:训练模型 仅有batch_x
                        outputs = self.model(batch_x)  # 这里把当前长度的数据输入到PatchTST，调用PatchTST.py的forward训练模型
                    else:  # 编解码器的预测器输入有四个
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 teacher_forcing_ratio=teacher_forcing_ratio)[0]
                        else:  # TODO: 其他模型的训练入口
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,             #  batch_y,
                                                 teacher_forcing_ratio=teacher_forcing_ratio)  # TODO: 这里的 batch_y 不能乱输入，会与mask弄混
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0     # 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]  # （32,96,321），对于多变量输入不变，单变量仅取最后一列
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # (32,144,321)==>(32,96,321)
                    loss = criterion(outputs, batch_y)  # MSE
                    train_loss.append(loss.item())      # 保存损失

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:  # 不执行
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()     # 反向传播
                    model_optim.step()  # 根据反向传播期间计算的梯度来更新模型的参数

                if self.args.lradj == 'TST':  # 调整学习率？？？
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # Reduce the teacher forcing ration every epoch
            if self.args.model == 'LSTM':    # 仅针对LSTM模型的mixed_teacher_forcing
                teacher_forcing_ratio -= 0.08
                teacher_forcing_ratio = max(0., teacher_forcing_ratio)
                print('teacher_forcing_ratio: ', teacher_forcing_ratio)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)  # TODO: 验证模型
            test_loss = self.vali(test_data, test_loader, criterion)  # 测试模型

            total_train_loss.append(train_loss)  # 添加损失到列表
            total_val_loss.append(vali_loss)     # 添加验证集损失

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':  # 调整学习率
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # 绘制训练损失变化图
        loss_path = './loss_plot/' + setting + '_loss.pdf'
        plot_loss(total_train_loss, total_val_loss, loss_path)  # 在util里面创建个函数！

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')  # 读取测试集数据
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()  # 设置验证模式
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)  # (32,96,321)
                batch_y = batch_y.float().to(self.device)  # (32,144,321)

                batch_x_mark = batch_x_mark.float().to(self.device)  # (32,96,4)
                batch_y_mark = batch_y_mark.float().to(self.device)  # (32,144,4)

                # decoder input     全零的(32,96,321) + batch_y的部分(32,48,321) ==> (32,144,321)  把batch_y的第二维度的
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)  # 使用训练好的模型输入测试集数据得到模型输出 (32,96,6)
                    else:
                        if self.args.output_attention:
                            outputs, atten_list = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            # if not os.path.exists('./weights_plot/'+setting):           # 这里进行注意力图的可视化
                            #     os.makedirs('./weights_plot/'+setting)
                            # torch.save(atten_list, './weights_plot/'+setting+'/tensor_list.pth')
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.reverse:
                    # 进行反归一化，在归一化时使用的是二维（时间步，特征个数），所以反归一化时也需要输入二维数据
                    outputs = outputs.detach().cpu().numpy()  # 把结果从GPU转到CPU
                    batch_y = batch_y.detach().cpu().numpy()
                    # (1,96,6) ==> (1*96, 6)
                    outputs = outputs.reshape(-1, outputs.shape[-1])  # (96,6)
                    batch_y = batch_y.reshape(-1, batch_y.shape[-1])  # (144,6)
                    # 进行反归一化  (96,6) =反归一化=> (96,6) =保持维度不变=> (1,96,6)      batch_y(1,144,6)
                    outputs = test_data.inverse_transform(outputs).reshape(-1, self.args.pred_len, outputs.shape[-1])
                    batch_y = test_data.inverse_transform(batch_y).reshape(-1, self.args.pred_len + self.args.label_len,
                                                                           outputs.shape[-1])
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]  # (32,96,321)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]  # (32,96,321)
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    # print(outputs.shape,batch_y.shape)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]  # (32,96,321)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # (32,96,321)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)  # (432,)  选取第一批数据的最后一列的指定行
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)  # (432,)  输入数据336行加上预测数据96行
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))      # 绘图(输入数据与预测数据)

        if self.args.test_flop:  # 测试模型浮点计算性能（默认不使用）
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)    # (321,32,96,1)
        trues = np.array(trues)    # (321,32,96,1)
        inputx = np.array(inputx)  # (321,32,96,6)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])       # (10272,96,1)
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])       # (10272,96,1)
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])   # (10272,96,6)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds[:, :, -1], trues[:, :, -1])  # TODO：这里仅取了一列值，适用于多预测单，可能不适用于多预测多
        print('mse:{}, mae:{}, rse:{}, rmse:{}, R2:{}'.format(mse, mae, rse, rmse, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, rmse:{}, R2:{}'.format(mse, mae, rse, rmse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
