import os
import torch
# from models import FormerTime, Informer
from models import MLP, DLinear
from models import TCN, MCDCNN, MiniRocket, TimesNet, InceptionTime

from models.TST import TST
from models.MCNN import MCNN
from models.ConvTimeNet import ConvTimeNet

# from models.MCNN import MCNN

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MLP': MLP,
            'DLinear': DLinear,
            
            'TCN': TCN,
            'MCNN': MCNN,
            'MCDCNN': MCDCNN,
            'TimesNet': TimesNet,
            'MiniRocket': MiniRocket,
            'ConvTimeNet': ConvTimeNet,
            'InceptionTime': InceptionTime,
            
            'TST': TST,
            # 'Informer': Informer,
            # 'FormerTime': FormerTime,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
