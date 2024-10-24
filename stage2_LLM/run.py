import argparse
import os
import torch

from pprint import pprint
from exp.exp_finetune import Exp_finetune
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
    parser.add_argument('--model', type=str, required=False, default='QTime')
    parser.add_argument('--cfg_path', type=str, default="../configs/stage_2_finetune.yaml")

    # data loader
    parser.add_argument('--data', type=str, default='UEA', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # ConvTimeNet
    parser.add_argument('--patch_size', type=int, default=8, help='patch size of Deformable Patch Embedding')
    parser.add_argument('--dw_ks', type=str, default='7,7,13,13,19,19', help="kernel size for each deep-wise convolution")
    
    # CrossTimeNet
    parser.add_argument('--pretrained_data', type=str, default='All')
    parser.add_argument('--local_model_path', type=str, default="/root/autodl-tmp/LM-Base-Model/Bert-Base-Uncased")

    # model define
    parser.add_argument('--proj_layer', type=str, default="Qformer", help='projection layer, options"[mlp, linear, Qformer]')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    
    # llm
    parser.add_argument('--reason_txt_len', type=int, required=False, default=0,  help='txt len of llm reasoning')
    
    # lora
    parser.add_argument('--lora_rank', type=int, default=8, help='rank of lora')
    parser.add_argument('--lora_alpha', type=int, default=8, help='alpha of lora')
    parser.add_argument('--lora_dropout', type=float, default=0.01, help='dropout of lora')
    parser.add_argument('--lora_target_modules', type=str, default='q,v', help='target_modules of lora')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of test input data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # scheduler
    parser.add_argument('--lr_decay_rate', type=float, default=0.95, help='optimizer learning decay rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='scheduler warmup lr')
    parser.add_argument('--warmup_steps',type=int,default=5000, help='warmup steps')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='minist lr when training')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='initial lr after warmup')

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
    
    if args.pretrained_data == 'All':
        dataset_list = ['CT','FD', 'PD','SAD', 'HB','EP', 'NATOPS','SRS1','SRS2','PEMS_SF']
    elif args.pretrained_data == 'Large':
        dataset_list = ['CT','FD', 'PD','SAD']
    elif args.pretrained_data == 'Small':
        dataset_list = ['HB','EP', 'NATOPS','SRS1','SRS2','PEMS_SF']
    else:
        dataset_list = [args.pretrained_data]
        
    print('Used dataset list: ', dataset_list)
    
    args.pretrain_dataset_list = dataset_list
    # args.pretrain_dataset_list = [
    #     'FD'
    # ]

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    # print_args(args)
    pprint(args)

    Exp = Exp_finetune

    accs, f1s, precs, recalls = [], [], [], []
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            # setting = '{}_{}_{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            #     args.model_id,
            #     args.model,
            #     args.data,
            #     args.d_model,
            #     args.n_heads,
            #     args.e_layers,
            #     args.d_layers,
            #     args.d_ff,
            #     args.factor,
            #     args.embed,
            #     args.distil,
            #     args.des, ii)

            setting = '{}_{}_rl{}'.format(args.model_id, args.model, args.lora_rank)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>get metrix : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.get_metrix(setting)
            
            # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting)
        #     accs.append(accuracy); f1s.append(f1)
        #     precs.append(precision), recalls.append(recall)
        #     torch.cuda.empty_cache()
            
        # print('average acc:{0:.4f}±{1:.4f}, f1:{2:.4f}±{3:.4f}'.format(np.mean(
        #     accs), np.std(accs), np.mean(f1s), np.std(f1s)))
        
        # print('average precision:{0:.4f}±{1:.4f}, recall:{2:.4f}±{3:.4f}'.format(np.mean(
        #     precs), np.std(precs), np.mean(recalls), np.std(recalls)))
    else:
        ii = 0
        # setting = '{}_{}_{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        #     args.model_id,
        #     args.model,
        #     args.data,
        #     args.d_model,
        #     args.n_heads,
        #     args.e_layers,
        #     args.d_layers,
        #     args.d_ff,
        #     args.factor,
        #     args.embed,
        #     args.distil,
        #     args.des, ii)

        setting = '{}_{}_rl{}'.format(args.model_id, args.model, args.lora_rank)
        
        exp = Exp(args)  # set experiments
        print('>>>>>>>get metrix : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        
        exp.get_metrix(setting, test=1)
        
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, test=1)
        
        torch.cuda.empty_cache()
