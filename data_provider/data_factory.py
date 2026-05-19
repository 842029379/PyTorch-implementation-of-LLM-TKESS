from data_provider.data_loader import My_data
from torch.utils.data import DataLoader
def data_provider(args,flag,drop_last_test=True,train_all=False):
    Data = My_data
    timeenc = 0 if args.embed != 'timeF' else 1  # timeenc 1、
    percent = args.percent  # 100
    max_len = args.max_len  # -1

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # 256
        freq = args.freq  # h

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,  # train
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,  # M
        target=args.target,  # OT
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all
    )
    print(flag, len(data_set))  # 15669


    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader