import os
import torch

from lavis.models.blip2_models.blip2_qformer import Blip2Qformer

# from models.MCNN import MCNN

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_cls = Blip2Qformer
        
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        # save_path = os.path.join('checkpoints', self.args.model_id + '_Qformer', 'Qformer_v1.pth')
        # torch.save(self.model.state_dict(), save_path) # without pretrain on Qformer
        # exit(0)

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
