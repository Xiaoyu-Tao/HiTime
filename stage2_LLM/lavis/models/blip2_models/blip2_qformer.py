"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import copy
import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    model = model.eval()
    model.train = disabled_train
    logging.info("freeze ts encoder")

class Blip2Qformer(Blip2Base):
    """
    BLIP2 model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with ts_MLP
        - finetuned: finetuned model with ts_MLP
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_ts", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "finetune": "configs/models/blip2/blip2_finetune.yaml"
    }

    def __init__(
        self,
        config, 
        encoder_model="InceptionTime",
        ts_precision="fp16",
        freeze_ts=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        model_root="/data/tingyue/tingyue/TS2LLM-new/data/LLM-Base-Model/Bert-Base-Uncased"
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer(model_root=model_root)
        self.ts_encoder_ft, self.ts_encoder_pt, self.ts_ln = self.init_ts_encoder(
            config, encoder_model, ts_precision
        )
        
        print("freeze ts: ", freeze_ts)
        if freeze_ts:
            freeze_model(self.ts_encoder_ft)
            freeze_model(self.ts_encoder_pt)

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.ts_encoder_ft.num_features, cross_attention_freq, model_root
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.ts_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def forward(self, samples):
        ts = samples["ts"]
        label = samples["label"]
        text = samples["text_input"]
        
        device = label.device
        
        # get target matrix
        B = label.shape[0]
        label_x = label.unsqueeze(0).repeat(B, 1)
        label_y = label.unsqueeze(1).repeat(1, B)

        zeros = torch.zeros((B, B)).to(device)
        ones = torch.ones((B, B)).to(device)
        
        tgt_matrix = torch.where((label_x == label_y), ones, zeros)
        NOR_tgt_matrix = torch.where((label_x == label_y), zeros, ones)
        
        all_the_same_flag = (tgt_matrix == 1.).all()
        # print(all_the_same_flag, torch.unique(tgt_matrix))
        # exit(0)
        
        ft_features = self.ts_encoder_ft.forward_feature(ts)
        # ts_features = ft_features
        pt_features = self.ts_encoder_pt.forward_feature(ts)
        
        ts_features = torch.cat([pt_features, ft_features], dim=1)
        
        ts_embeds = self.ts_ln(ts_features)
        # print(ts_embeds.shape); exit(0)
        
        ts_atts = torch.ones(ts_embeds.size()[:-1], dtype=torch.long).to(
            ts.device
        )

        query_tokens = self.query_tokens.expand(ts_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=ts_embeds,
            encoder_attention_mask=ts_atts,
            use_cache=True,
            return_dict=True,
        )

        ts_feats = F.normalize(
            self.ts_proj(query_output.last_hidden_state), dim=-1
        )
 
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(ts.device)
        
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        # print("ts_feats: ", ts_feats.shape)
        
        # print("text_feat: ", text_feat.shape)

        ###============== ts-text Contrastive ===================###
        ts_feats_all = concat_all_gather(
            ts_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            ts_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # ts-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = F.sigmoid(sim_i2t / self.temp)

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), ts_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-ts similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = F.sigmoid(sim_t2i / self.temp)  # [batch_size, batch_size*num_gpu]
        
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # print('sim t2i: \n',sim_t2i)
        
        # print('sim i2t: \n',sim_i2t)
        
        # print('label: \n',label)
        
        # print('tgt_matrix: \n',tgt_matrix)

        # exit(0)

        rank = 0 # dist.get_rank()
        bs = ts.size(0)
        targets = tgt_matrix.to(ts.device)

        loss_itc = (
            F.binary_cross_entropy(sim_i2t, targets, reduction='mean')
            + F.binary_cross_entropy(sim_t2i, targets, reduction='mean')
        ) / 2
        
        # return BlipOutput(
        #     loss=loss_itc, #  + loss_itm + loss_lm,
        #     loss_itc=loss_itc,
        #     loss_itm=torch.Tensor([0]).to(loss_itc.device),
        #     loss_lm=torch.Tensor([0]).to(loss_itc.device),
        # )

        ###============== ts-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        ts_embeds_world = all_gather_with_grad(ts_embeds)
        # print("ts_embeds_world", ts_embeds_world.shape)
        # exit(0)
        # print(ts_embeds_world)
        
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i *= NOR_tgt_matrix
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t *= NOR_tgt_matrix
            
        # print(weights_t2i)
        # exit(0)

        if not all_the_same_flag:
            # select a negative ts for each text
            ts_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                ts_embeds_neg.append(ts_embeds_world[neg_idx])
            ts_embeds_neg = torch.stack(ts_embeds_neg, dim=0)

            # select a negative text for each ts
            text_ids_neg = []
            text_atts_neg = []
            
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(text_input_ids_world[neg_idx])
                text_atts_neg.append(text_attention_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
                dim=0,
            )
            
            ts_embeds_all = torch.cat(
                [ts_embeds, ts_embeds_neg, ts_embeds], dim=0
            )  # pos, neg, pos
            ts_atts_all = torch.ones(ts_embeds_all.size()[:-1], dtype=torch.long).to(
                ts.device
            )
        else:
            text_ids_all = text_tokens.input_ids
            text_atts_all = text_tokens.attention_mask
            
            ts_embeds_all = ts_embeds
            ts_atts_all = torch.ones(ts_embeds_all.size()[:-1]).to(ts.device)

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1) 
        # print("query_tokens_itm", query_tokens_itm.shape)
        # print("text_atts_all", text_atts_all.shape)
        # exit(0)
        
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            ts.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=ts_embeds_all,
            encoder_attention_mask=ts_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        if not all_the_same_flag:
            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                dim=0,
            ).to(ts.device)
        else:
            itm_labels = torch.ones(bs, dtype=torch.long).to(ts.device)
        
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= ts Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        # decoder_input_ids = torch.roll(decoder_input_ids, 1, 0) 
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        
        # bos_attns = torch.ones((bs, ), dtype=torch.long).to(ts.device)
        # text_tokens.attention_mask = torch.roll(text_tokens.attention_mask, 1, 0) 
        # text_tokens.attention_mask[:, 0] = bos_attns
        
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            ts.device
        )
        
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_itc + loss_itm, #  + loss_lm,  
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )
        
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
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
        
        ft_features = self.ts_encoder_ft.forward_feature(ts)
        pt_features = self.ts_encoder_pt.forward_feature(ts, dataid=0)
        
        ts_features = torch.cat([pt_features, ft_features], dim=1)
        
        ts_embeds = self.ts_ln(ts_features)

        if not use_nucleus_sampling:
            ts_embeds = ts_embeds.repeat_interleave(num_beams, dim=1)
        else:
            num_beams = 1
        ts_atts = torch.ones(ts_embeds.size()[:-1], dtype=torch.long).to(
            ts.device
        )

        model_kwargs = {
            "encoder_hidden_states": ts_embeds,
            "encoder_attention_mask": ts_atts,
        }

        input_ids = (
            torch.LongTensor(ts.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(ts.device)
        )
        query_tokens = self.query_tokens.expand(ts_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_ts(self, ts):
        ft_features = self.ts_encoder_ft.forward_feature(ts)
        pt_features = self.ts_encoder_pt.forward_feature(ts)
        
        ts_features = torch.cat([pt_features, ft_features], dim=1)
        
        ts_embeds = self.ts_ln(ts_features)
            
        ts_atts = torch.ones(ts_embeds.size()[:-1], dtype=torch.long).to(
            ts.device
        )

        query_tokens = self.query_tokens.expand(ts_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=ts_embeds,
            encoder_attention_mask=ts_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, ts_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, ts_inputs, text_ids, text_atts):
        ts_atts = torch.ones(ts_inputs.size()[:-1], dtype=torch.long).to(
            ts_inputs.device
        )
        query_tokens = self.query_tokens.expand(ts_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            ts_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=ts_inputs,
            encoder_attention_mask=ts_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - ts (torch.Tensor): A tensor of shape (B, C, H, W) containing the ts.
                    Raw tss should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "ts".
                If "multimodal", return ts features and multimodal features;
                if "text", return text features;
                if "ts", return ts features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        ts = samples.get("ts")
        caption = samples.get("text_input")

        # assert mode is one of "ts", "text", "multimodal"
        assert mode in [
            "ts",
            "text",
            "multimodal",
        ], "mode must be one of 'ts', 'text', 'multimodal'"

        # initalize output
        ts_embeds, text_embeds, multimodal_embeds = None, None, None
        ts_features, text_features = None, None

        if mode == "ts":
            assert (
                ts is not None
            ), "ts is not provided for mode 'ts' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                ts_embeds_frozen = self.ts_ln(self.ts_encoder(ts))
            ts_embeds_frozen = ts_embeds_frozen.float()
            ts_atts = torch.ones(
                ts_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                ts_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=ts_embeds_frozen,
                encoder_attention_mask=ts_atts,
                return_dict=True,
            )
            ts_embeds = query_output.last_hidden_state
            ts_features = F.normalize(self.ts_proj(ts_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                ts_embeds_frozen = self.ts_ln(self.ts_encoder(ts))
            ts_embeds_frozen = ts_embeds_frozen.float()
            ts_atts = torch.ones(
                ts_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                ts_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=ts_embeds_frozen,
                encoder_attention_mask=ts_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            ts_embeds=ts_embeds,
            ts_embeds_proj=ts_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        # print(cfg); exit(0)
        
        encoder_model = cfg.get("encoder_model", "ts_MLP")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        # drop_path_rate = cfg.get("drop_path_rate", 0)
        # use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        ts_precision = cfg.get("ts_precision", "fp16")
        freeze_ts = cfg.get("freeze_ts", False)

        max_txt_len = cfg.get("max_txt_len", 32)
        model_root = cfg.get("model_root", "/data/tingyue/tingyue/TS2LLM-new/data/LLM-Base-Model/Bert-Base-Uncased")
        
        model = cls(
            cfg, 
            encoder_model=encoder_model,
            ts_precision=ts_precision,
            freeze_ts=freeze_ts,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=256,
            max_txt_len=max_txt_len,
            model_root=model_root
        )

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
    
# if __name__ == "__main__":
#     Qformer = 