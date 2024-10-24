import argparse
import os
import torch
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='UEA', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # aligned dims
    parser.add_argument('--d_qformer', type=int, default=768, help='aligned dims for qformer')
    
    # ConvTimeNet
    parser.add_argument('--patch_size', type=int, default=8, help='patch size of Deformable Patch Embedding')
    parser.add_argument('--dw_ks', type=str, default='7,7,13,13,19,19', help="kernel size for each deep-wise convolution")
    
    # InceptionTime
    parser.add_argument('--use_patch', type=int, default=8, help='use patch in Inception when true')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--is_Revin',type=int,default=0,help='whether to use Revin')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    args.dw_ks = [int(ks) for ks in args.dw_ks.split(',')]

    torch.cuda.set_device(0)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Classification

    accs, f1s, precs, recalls = [], [], [], []
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_bs{}_lr{}_ps{}_dk{}_{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.batch_size,
                args.learning_rate,
                args.patch_size,
                args.dw_ks,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            accuracy, f1, precision, recall = exp.test(setting)
            accs.append(accuracy); f1s.append(f1)
            precs.append(precision), recalls.append(recall)
            torch.cuda.empty_cache()
            
        print('average acc:{0:.4f}±{1:.4f}, f1:{2:.4f}±{3:.4f}'.format(np.mean(
            accs), np.std(accs), np.mean(f1s), np.std(f1s)))
        
        print('average precision:{0:.4f}±{1:.4f}, recall:{2:.4f}±{3:.4f}'.format(np.mean(
            precs), np.std(precs), np.mean(recalls), np.std(recalls)))
    else:
        ii = 0
        setting = '{}_{}_{}_bs{}_lr{}_ps{}_dk{}_{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.batch_size,
            args.learning_rate,
            args.patch_size,
            args.dw_ks,
            args.distil,
            args.des, ii)

        # setting = ""
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, test=1)
        checkpoint_path = None
        # checkpoint_path = f'../datas/Encoder-Model/{args.model}/{args.model}_{args.model_id}.pth'
        exp.test(setting, test=1, checkpoint_path=checkpoint_path)
        torch.cuda.empty_cache()
