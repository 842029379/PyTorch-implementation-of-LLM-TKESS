import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class My_data(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='IndPensim.csv',
                 target='OT', scale=True,
                 timeenc=0, freq='t', percent=100, max_len=-1, train_all=False):

        self.seq_len = size[0]  # 96

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features  # MS
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))

        self.tot_len = len(self.data_x) - self.seq_len + 1

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                            self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        df_raw = df_raw[cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]



        border1 = border1s[self.set_type]  # 0
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 不取时间
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:  # 标准化


            data = self.scaler.fit_transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]  # 提取时间
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)  # 对每一行应用函数
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:  # 时间编码为1
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values),
                                       freq=self.freq)  # 指定时间序列的频率 转换为datatimeIndex格式
            data_stamp = data_stamp.transpose(1, 0)  # 8640，6

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]  # 如果输出为一维，需要改动
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # feat_id = index // self.tot_len  # 得到特征的id
        # s_begin = index % self.tot_len  # 余数在0-8209以内

        s_begin = index
        s_end = self.seq_len + s_begin  # s_begin 到s_begin+336

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[s_begin:s_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]  # 取第一个时间戳的时间信息当作token的编码信息
        seq_y_mark = self.data_stamp[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len  + 1)  # 总共的样本数量

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
