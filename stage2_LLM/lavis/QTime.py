"""
QTime based on llama 2
"""

import os
import torch
import copy

import logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from omegaconf import OmegaConf

from types import MethodType
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from utils.tools import extract_useful_message
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer

from lavis.models.ts_MLP import create_ts_MLP
from lavis.models.InceptionTime import create_InceptionTime
from lavis.models.ConvTimeNet.ConvTimeNet import create_ConvTimeNet

from data_provider.gen_text import get_prompt, TextGenerator

def lora_train(self, mode=True):
    for name, param in self.named_parameters():
        if ('lora' in name) and (mode is True):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    return self

class QTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.configs = OmegaConf.load(self.args.cfg_path)
        
        self.prompt = get_prompt(args.model_id)
        self.text_processor = TextGenerator(args.model_id)
        
        if self.args.proj_layer == 'Qformer': # Pretrained Qformer
            self.pretrained_Qformer = self._load_pretrained_Qformer()
            self.llm_proj = nn.Linear(768, 4096)
        
        elif self.args.proj_layer == 'mlp': # MLP
            self.ts_encoder = self._load_pretrained_Encoder()
            self.freeze_ts(freeze_ts=False)
            
            self.llm_proj = nn.Sequential(
                nn.Linear(768, 1024),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(1024, 2048),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(2048, 4096),
                nn.Dropout(0.1),
            )
            
        elif self.args.proj_layer == 'linear':
            self.ts_encoder = self._load_pretrained_Encoder()
            self.freeze_ts(freeze_ts=False)
            
            self.llm_proj = nn.Linear(768, 4096)
        
        else:
            raise ValueError(f"No such projection layer@ {self.args.proj_layer}!")
            
        self.pretrained_llm, self.tokenizer = self._load_pretrained_LLM()
        # print(self.pretrained_llm.generate.__doc__)
        # exit(0)
        
        self.pretrained_llm.resize_token_embeddings(len(self.tokenizer))
        
        # def disabled_train(self, mode=True):
        #     """Overwrite model.train with this function to make sure train/eval mode
        #     does not change anymore."""
        #     return self
        
        # self.pretrained_Qformer = self.pretrained_Qformer.eval()
        # self.pretrained_Qformer.train = disabled_train
        
        # print(self.pretrained_Qformer)
        
        # print("===========================")
        
        # print(self.pretrained_llm)
        
        # exit(0)

    def freeze_ts(self, freeze_ts):
        def disabled_train(self, mode=True):
            """Overwrite model.train with this function to make sure train/eval mode
            does not change anymore."""
            return self
        
        if freeze_ts:
            for name, param in self.ts_encoder.named_parameters():
                param.requires_grad = False
            self.ts_encoder = self.ts_encoder.eval()
            self.ts_encoder.train = disabled_train
            logging.info("freeze ts encoder")

    def _load_pretrained_Encoder(self):
        Qformer_config = self.configs['model']['Qformer']
        Qformer_config.update(self.configs['model'][self.args.model_id])
        
        model_name = Qformer_config.get("encoder_model", "ts_MLP")
        precision = Qformer_config.get("ts_precision", "fp16")
                
        Qformer_config.update(vars(self.args))
        if model_name == "ts_MLP":
            ts_encoder = create_ts_MLP(Qformer_config, precision)
        elif model_name == "InceptionTime":
            ts_encoder = create_InceptionTime(Qformer_config, precision)
        elif model_name == "ConvTimeNet":
            ts_encoder = create_ConvTimeNet(Qformer_config, precision)
                    
        print(f"Load ts encoder@ {model_name}")
        self.encoder_name = model_name
        
        return ts_encoder
    

    def _load_pretrained_Qformer(self):
        Qformer_config = self.configs['model']['Qformer']
        Qformer_config.update(vars(self.args))
        
        Qformer_config['attn_implementation'] = "eager"
        pretrained_Qformer = Blip2Qformer.from_config(Qformer_config)
            
        print('loading pretrained Qformer...')
        qformer_pretrained_pth = os.path.join(Qformer_config['qformer_pretrained_folder'], f"Qformer_{self.args.model_id}.pth")
        pretrained_Qformer.load_state_dict(torch.load(qformer_pretrained_pth))
        
        # if 'qformer_transfer_pretrained_pth' in Qformer_config.keys():
        #     import copy
        #     transfer_config = copy.deepcopy(self.configs['model']['Qformer'])
        #     transfer_config.update(self.configs['model'][Qformer_config['transfer_set']])
            
        #     # print(transfer_config)
        #     # exit(0)
            
        #     transfer_pretrained_Qformer = Blip2Qformer.from_config(transfer_config)
        #     transfer_pretrained_Qformer.load_state_dict(torch.load(Qformer_config['qformer_transfer_pretrained_pth']))
            
        #     pretrained_Qformer.Qformer.load_state_dict(transfer_pretrained_Qformer.Qformer.state_dict())
        
        print(f"Successfully Load pretrained Qformer!")
        # print('Using random bert...')
        return pretrained_Qformer
    
    def _load_pretrained_LLM(self):
        llm_config = self.configs['model']['llm']
        
        print('loading pretrained llm...')
        
        # load tokenizer
        print(llm_config['model_root'])
        tokenizer = AutoTokenizer.from_pretrained(llm_config['model_root'], local_files_only=True)
        
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model = AutoModelForCausalLM.from_pretrained(
            llm_config['model_root'],
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        
        target_modules=[]
        if self.args.lora_target_modules == 'q,k,v,o':
            target_modules=['q_proj','k_proj','v_proj','o_proj']    
        elif self.args.lora_target_modules == 'q':
            target_modules=['q_proj']
        else:
            target_modules=['q_proj', 'v_proj'] 
        
        peft_config = LoraConfig(
            r=self.args.lora_rank, 
            lora_alpha=self.args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.args.lora_dropout,
            bias='none',
            task_type='CALSAL_LM'
        )
        
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = True
        
        model.train = MethodType(lora_train, model)
        
        return model, tokenizer

    def _get_trainable_stat(self):
        self.train()
        stat_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad is True:
                c_name, c_param = copy.deepcopy(name), copy.deepcopy(param)
                stat_dict.update({c_name: c_param})
                
        self.eval()
        return stat_dict
        

    @staticmethod
    def init_weights_kaiming(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, samples):
        ts = samples["ts"]
        label = samples["label"]
        text = samples["text_input"]

        device = label.device
        
        # print('ts: ', ts.shape)
        
        # Qformer inference
        # qformer_output: (B, 32, 768)    ts_embed: (B, 128, 768)
        
        if self.args.proj_layer == 'Qformer':
            qformer_output, ts_embed = self.pretrained_Qformer.forward_ts(ts)
            ts_query_embeds = self.llm_proj(qformer_output)
        else:
            ts_embed = self.ts_encoder.forward_feature(ts)
            ts_query_embeds = self.llm_proj(ts_embed)

        ts_query_embeds = ts_query_embeds.to(torch.bfloat16)
        
        text_query_tokens = self.tokenizer(self.prompt, return_tensors="pt",)
        text_query_tokens_ids = text_query_tokens.input_ids[0].clone()
        text_query_tokens_ids = text_query_tokens_ids.to(ts.device)
        text_query_embeds = self.pretrained_llm.model.model.embed_tokens(text_query_tokens_ids)
        text_query_embeds = text_query_embeds.to(torch.bfloat16)
        
        text_query_embeds = text_query_embeds.unsqueeze(0).repeat(ts_query_embeds.shape[0], 1, 1)
        
        query_embeds = torch.cat([ts_query_embeds, text_query_embeds], dim=1)
        
        # query_embeds = ts_query_embeds
        query_output = self.pretrained_llm.model(
            inputs_embeds=query_embeds,
            use_cache=True,
            return_dict=True,
        )
        
        # new_text = []
        
        # for t in text:
        #     new_t = self.prompt + '\n' + t
        #     new_text.append(new_t)
        # text = new_text
        
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=15, # without <bos>
            return_tensors="pt",
        ).to(ts.device)
        
        # print(text_tokens.input_ids.shape)
        # text_tokens = text_tokens.input_ids.tolist()
        # print(self.tokenizer.batch_decode(text_tokens))
        # exit(0)
        
        # decoder_input_ids = text_tokens.input_ids.clone()
        # decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )
        
        query_atts = torch.ones(query_embeds.size()[:-1], dtype=torch.long).to(
            ts.device
        )
        
        # bos_attns = torch.ones((ts.size(0), ), dtype=torch.long).to(ts.device)
        # text_tokens.attention_mask = torch.roll(text_tokens.attention_mask, 1, 0) 
        # text_tokens.attention_mask[:, 0] = bos_attns
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        # attention_mask = torch.cat([query_atts, bos_attns, text_tokens.attention_mask], dim=1)
        
        lm_output = self.pretrained_llm(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            use_cache=True,
            return_dict=True,
            labels=labels,
        )
        
        # print(lm_output.loss)
        
        loss_lm = lm_output.loss
        
        return {
            "loss":loss_lm,
            "loss_itc":torch.Tensor([0]).to(loss_lm.device),
            "loss_itm":torch.Tensor([0]).to(loss_lm.device),
            "loss_lm":loss_lm,
        }
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - ts (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each ts.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        ts = samples["ts"]
        
        device = ts.device
    
        # Qformer inference
        # qformer_output: (B, 32, 768)    ts_embed: (B, 128, 768)
        
        qformer_output, ts_embeds = self.pretrained_Qformer.forward_ts(ts)
        ts_query_embeds = self.llm_proj(qformer_output).to(torch.bfloat16)
        
        text_query_tokens = self.tokenizer(self.prompt, return_tensors="pt",)
        # print(self.prompt)
        text_query_tokens_ids = text_query_tokens.input_ids[0].clone()
        # print(text_query_tokens_ids)
        # exit(0)
        
        # text_query_tokens_ids = torch.cat([torch.tensor([self.tokenizer.bos_token_id]), text_query_tokens_ids])
        text_query_tokens_ids = text_query_tokens_ids.to(ts.device)
        text_query_embeds = self.pretrained_llm.model.embed_tokens(text_query_tokens_ids)
        text_query_embeds = text_query_embeds.to(torch.bfloat16)
        
        text_query_embeds = text_query_embeds.unsqueeze(0).repeat(ts_query_embeds.shape[0], 1, 1)
        query_embeds = torch.cat([ts_query_embeds, text_query_embeds], dim=1)
        
        query_embeds = query_embeds.to(torch.bfloat16)
        query_output = self.pretrained_llm.model(
            inputs_embeds=query_embeds,
            use_cache=True,
            return_dict=True,
        )
        
        if use_nucleus_sampling:
            ts_embeds = ts_embeds.repeat_interleave(num_beams, dim=1)
        else:
            num_beams = 1

        tokens_tensor = (
            torch.LongTensor(ts.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(ts.device)
        )
        
        max_length = 15 + self.args.reason_txt_len
        
        past = query_output.past_key_values
        
        for i in range(max_length):
            with torch.no_grad(): # output, past
                output = self.pretrained_llm(tokens_tensor, past_key_values=past, \
                    use_cache=True, return_dict=False)[0] # past

            new_tokens_tensor = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)
            tokens_tensor = torch.cat([tokens_tensor, new_tokens_tensor], dim=-1)
            
            # print(type(indexed_tokens))
            # print(token.flatten()); exit(0)

            # indexed_tokens[:, i:i+1] = tokens_tensor
            
        indexed_tokens = tokens_tensor.tolist()
        
        # batch_decode
        nums = len(indexed_tokens)
        seqs = self.tokenizer.batch_decode(indexed_tokens)
        gts = [self.text_processor.get_str_from_label(int(samples['label'][idx])) for idx in range(nums)]
        label_list = [
                self.text_processor.extract_key_labels(seqs[k])
                for k in range(nums)
            ]
        
        for idx in range(len(seqs)):
            print(gts[idx], seqs[idx], label_list[idx])
            
        # exit(0)
                
        return label_list
