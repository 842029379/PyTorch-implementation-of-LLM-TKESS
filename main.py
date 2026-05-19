from data_provider.data_factory import data_provider
import argparse
import os.path
import warnings
import random
import torch
from tqdm import tqdm
import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali
import torch.nn as nn
import os
from tensorboardX import SummaryWriter
import time
from models.PatchTST import PatchTST
from models.LLM_TKESS import LLM_TKESS
from utils.metrics import metric
from models.DLinear import DLinear
from thop import profile
warnings.filterwarnings('ignore')
fix_seed = 111
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def test(model, test_data, test_loader, args, device, itr):
    preds = []
    trues = []
    pred_all = []
    trues_all = []
    # mases = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, _) in tqdm(enumerate(test_loader)):
            # outputs_np = batch_x.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_input_itr{}_{}.npy".format(itr, i), outputs_np)
            # outputs_np = batch_y.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_true_itr{}_{}.npy".format(itr, i), outputs_np)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_x = batch_x.float().to(device)
            torch.cuda.reset_peak_memory_stats()
            # batch_y = batch_y.float().to(device).unsqueeze(-1)
            shift_outputs, shift_labels = model(batch_x, batch_x_mark)
            # outputs = model(batch_x[:, -args.seq_len:, :], itr)
            flops, params = profile(model, inputs=(batch_x, batch_x_mark))
            print("FLOPs:", flops)
            print("GFLOPs:", flops / 1e9)

            # encoder - decoder
            # outputs = outputs[:, -args.pred_len:, :]
            # batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = shift_outputs.detach().cpu().numpy()
            true = shift_labels.detach().cpu().numpy()

            pred_all.append(pred)
            trues_all.append(true)
            preds.append(pred)
            trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd, r2 = metric(preds, trues)
    # print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, mases:{:.4f}'.format(mae, mse, rmse, smape, mases))
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f},r2:{:.4f}'.format(mae, mse, rmse, smape, r2))

    return mse, mae, rmse, r2

parser = argparse.ArgumentParser(description="LLMTKESS")
parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--features', type=str, default='MS')
parser.add_argument('--model_id', type=str, required=True, default='train')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--tmax', type=int, default=10)
parser.add_argument('--itr', type=int, default=1, help='experi ments times')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--freq', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--root_path', type=str, default='./dataset/')
parser.add_argument('--data_path', type=str, default='IndPensim.csv')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--tensorboard_dir', type=str, default='tensorboard')

parser.add_argument('--lora_layer', type=int, default=2, help='lora attn dimension')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')
parser.add_argument('--lora_dropout', default=0.0, type=float,
                    help='dropout probability for lora layers')
args = parser.parse_args()
start_time = time.time()

SEASONALITY_MAP = {
    "1_minute_5_seconds": 1328,
    "minutely": 1440,  # 每分钟
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
}

mses = []
maes = []
for ii in range(args.itr):
    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len,
                                                                             args.pred_len,
                                                                             args.d_model, args.n_heads, args.e_layers,
                                                                             args.gpt_layers,
                                                                             args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints,setting)
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, args.model_id))
    if not os.path.exists(path):
        os.makedirs(path)
    # if args.freq == 0:
    args.freq = 's'
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    args.features_num = train_data.data_x.shape[1]





    device = torch.device('cuda:0')

    time_now = time.time()
    train_steps = len(train_loader)
    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    else:
        model = LLM_TKESS(args, device)
    for i, (name, param) in enumerate(model.named_parameters()):
        print(name + '   ' + str(param.requires_grad))


    params = model.parameters()   # 去除要优化的参数 即输入输出线性层以及GPT2中的归一化层和位置编码层
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':  # 损失为MSE
        criterion = nn.MSELoss()
    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        criterion = SMAPE()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
    k = 0

    for epoch in range(args.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, _) in tqdm(enumerate(train_loader)):

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)


            batch_x_mark = batch_x_mark.float().to(device)



            shift_outputs,shift_labels = model(batch_x, batch_x_mark)



            loss = criterion(shift_outputs, shift_labels)
            train_loss.append(loss.item())

            if (i + 1) % 20 == 0:

                board.add_scalar('loss_train_epoch{}'.format(epoch), loss, (k + 1)*20)
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
                k+=1
            loss.backward()
            model_optim.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        board.add_scalar('Val_train', vali_loss, i)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        early_stopping(vali_loss, model, path)
        if epoch % 5 ==0:
            torch.save(model.state_dict(), path + '/' + 'checkpoint'+str(epoch)+'.pth')
        if early_stopping.early_stop:

            print("Early stopping")
            print("All time: {}".format(time.time() -start_time))
            break

#     best_model_path = path + '/' + 'checkpoint.pth'
#     model.load_state_dict(torch.load(best_model_path))
#     print("------------------------------------")
#     mse, mae, rmse, r2 = test(model, test_data, test_loader, args, device, ii)
#     mses.append(mse)
#     maes.append(mae)
#
# mses = np.array(mses)
# maes = np.array(maes)
# print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
# print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))