from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_metrics
from exp.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler

import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

from types import MethodType
from pprint import pprint
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from tqdm import tqdm
from lavis.QTime import QTime

warnings.filterwarnings('ignore')


class Exp_finetune(Exp_Basic):
    def __init__(self, args):
        super(Exp_finetune, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        
        self.args.enc_in = train_data.dims 
        
        self.args.num_class = len(train_data.class_names)
        
        model = QTime(self.args)
        return model
    
    def _get_data(self, flag, **kwargs):
        data_set, data_loader = data_provider(self.args, flag, **kwargs)
        return data_set, data_loader

    def _select_optimizer(self):
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        # logging.info("number of trainable parameters: %d" % num_parameters)
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(0.05),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        
        model_optim = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.learning_rate),
            weight_decay=float(0.05),
            betas=(0.9, 0.999),
        )

        return model_optim
    
    def _select_scheduler(self, optimizer):
        
        lr_sched_cls = LinearWarmupCosineLRScheduler
        # lr_sched_cls = LinearWarmupStepLRScheduler

        # max_epoch = self.config.run_cfg.max_epoch
        max_epoch = self.args.train_epochs
        # min_lr = self.config.run_cfg.min_lr
        min_lr = self.args.min_lr 
        # init_lr = self.config.run_cfg.init_lr
        init_lr = self.args.init_lr 

        # optional parameters
        decay_rate = self.args.lr_decay_rate
        warmup_start_lr = self.args.warmup_lr
        warmup_steps = self.args.warmup_steps

        lr_sched = lr_sched_cls(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=decay_rate,
            warmup_start_lr=warmup_start_lr,
            warmup_steps=warmup_steps,
        )
        
        return lr_sched

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader):
        total_loss, loss_itm, loss_itc, loss_lm = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_ts, label, texts) in enumerate(vali_loader):
                
                batch_ts = batch_ts.to(self.device)
                label = label.to(self.device)
                samples = {"ts":batch_ts, "label":label, "text_input":texts}

                output = self.model(samples)
                total_loss.append(output['loss'].item())
                loss_itm.append(output['loss_itm'].item())
                loss_itc.append(output['loss_itc'].item())
                loss_lm.append(output['loss_lm'].item())

        total_loss = np.average(total_loss)
        loss_itm = np.average(loss_itm)
        loss_itc = np.average(loss_itc)
        loss_lm = np.average(loss_lm)

        self.model.train()
        return total_loss, loss_itm, loss_itc, loss_lm

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        # test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_sched = self._select_scheduler(model_optim)
        criterion = self._select_criterion()

        report_iter = min(len(train_loader) // 2, 100)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            print("Epoch: {}".format(epoch + 1))

            for i, (batch_ts, label, texts) in enumerate(train_loader):
                
                iter_count += 1
                
                batch_ts = batch_ts.to(self.device)
                label = label.to(self.device)
                samples = {"ts":batch_ts, "label":label, "text_input":texts}

                model_optim.zero_grad()
                output = self.model(samples)
                loss = output["loss"]
                
                loss_dict = {}
                for k,v in output.items():
                    if "loss" in k:
                        loss_dict[k] = v
                        
                train_loss.append(loss.item())
                
                # if np.isnan(loss.item()):
                #     exit(0)

                if (i + 1) % report_iter == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | loss_itc: {3:.7f} | loss_itm: {4:.7f} | loss_lm: {5:.7f}" \
                        .format(i + 1, epoch + 1, loss.item(), loss_dict['loss_itc'].item(), \
                            loss_dict['loss_itm'].item(), loss_dict['loss_lm'].item()))
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    now_lr = model_optim.state_dict()['param_groups'][0]['lr']
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s; learning rate: {:.7f}'.format(speed, left_time, now_lr))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                model_sched.step(cur_epoch=epoch, cur_step=i)
                
                # if i == 2:
                #     break
                

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            vali_loss, vali_loss_itm, vali_loss_itc, vali_loss_lm = self.vali(vali_data, vali_loader)
            # vali_loss, val_accuracy, val_f1 = self.vali(vali_data, vali_loader, criterion)
            # test_loss, test_accuracy, test_f1 = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Vali itc: {3:.7f} Vali itm: {4:.7f} Vali lm: {5:.7f}"
                .format(epoch + 1, train_loss, vali_loss, vali_loss_itm, vali_loss_itc, vali_loss_lm))

            # print(
            #     "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali F1: {5:.3f} Test Loss: {6:.3f} Test Acc: {7:.3f} Test F1: {8:.3f}"
            #     .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, val_f1, test_loss, test_accuracy, test_f1))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # if (epoch + 1) % 5 == 0:
            #     adjust_learning_rate(model_optim, epoch + 1, self.args)
            
            print("")

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')), strict=False)
            
        # if isinstance(self.model.pretrained_llm, ...): 
        #   self.model.pretrained_llm = self.model.pretrained_llm.merge_and_unload()

        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        total_loss, loss_itm, loss_itc, loss_lm = [], [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_ts, label, texts) in enumerate(test_loader):
                
                batch_ts = batch_ts.to(self.device)
                label = label.to(self.device)
                samples = {"ts":batch_ts, "label":label, "text_input":texts}

                output = self.model(samples)
                total_loss.append(output['loss'].item())
                loss_itm.append(output['loss_itm'].item())
                loss_itc.append(output['loss_itc'].item())
                loss_lm.append(output['loss_lm'].item())

        total_loss = np.average(total_loss)
        loss_itm = np.average(loss_itm)
        loss_itc = np.average(loss_itc)
        loss_lm = np.average(loss_lm)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        print("Test loss: {0:.7f} | loss_itc: {1:.7f} | loss_itm: {2:.7f} | loss_lm: {3:.7f}" \
            .format(total_loss, loss_itm, loss_itc, loss_lm))
        
        file_name='result_qformer_pretrain.txt'
        
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('Test loss:{}'.format(total_loss) + '\n')
        f.write('loss_itc:{}'.format(loss_itc) + '\n')
        f.write('loss_itm:{}'.format(loss_itm) + '\n')
        f.write('loss_lm:{}'.format(loss_lm) + '\n')
        f.write('\n')
        f.write('\n')
        f.close()
        return 

    def get_metrix(self, setting, test=0):
        from utils.tools import plot_confusion_matrix
        
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')), strict=False)
        
        self.model.pretrained_llm = self.model.pretrained_llm.merge_and_unload()
        
        # Tesing metrix
        self.model.eval()
        
        # save finetuned Qformer
        # torch.save(self.model.pretrained_Qformer.state_dict(), os.path.join('./checkpoints/' + setting, 'Qformer_fineturned_v0.pth'))
        # 
        # exit(0)
        
        with torch.no_grad():
            gt_list, pred_list = [], []
            for i, (batch_ts, label, texts) in tqdm(enumerate(test_loader)):
                batch_ts = batch_ts.to(self.device)
                label = label.tolist()
                samples = {"ts":batch_ts, 'label':label, 'text':texts}

                pred = self.model.generate(samples)
                
                gt_list.extend(label)
                pred_list.extend(pred)
                
                # if i == 10:
                #     break
                
                
        gt, pred = np.array(gt_list), np.array(pred_list)
        metrics = cal_metrics(pred, gt)
        accuracy, f1_score = metrics['acc'], metrics['macro avg']['f1-score']
        precision, recall = metrics['macro avg']['precision'], metrics['macro avg']['recall']

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # n_class = np.unique(gt)
        # plot_confusion_matrix(pred, gt, self.args.model_id, n_class.shape[0], folder_path)
            
        file_name='result_finetune.txt'
        
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy) + '\n')
        f.write('f1-score:{}'.format(f1_score) + '\n')
        f.write('precision:{}'.format(precision) + '\n')
        f.write('recall:{}'.format(recall) + '\n')
        f.write('\n')
        f.write('\n')
        f.close()

        print('accuracy:{}'.format(accuracy), 'f1:{}'.format(f1_score))
        print('precision:{}'.format(precision), 'recall:{}'.format(recall))
        
        # exit(0)
        return accuracy, f1_score, precision, recall
        
        
        
            
        