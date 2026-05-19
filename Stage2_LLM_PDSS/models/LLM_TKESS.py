import numpy as np
import torch
import torch.nn as nn
from torch import optim
from models.Model import GPT2ConfigMy, GPT2Model
# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time,TimeFeatureEmbedding
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
# from layers.Embed import DataEmbedding
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print("x.shape = {}".format(x.shape))
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
class Adapter(nn.Module):
    def __init__(self, in_feat, hid_dim, skip=True):
        super().__init__()
        self.D_fc1 = nn.Linear(in_feat, hid_dim)
        self.D_fc2 = nn.Linear(hid_dim, in_feat)
        self.act = nn.GELU()
        self.skip = skip
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        if self.skip:
            return x + self.drop(self.D_fc2(self.act(self.D_fc1(x))))
        else:
            return self.drop(self.D_fc2(self.act(self.D_fc1(x))))

class LLM_TKESS(nn.Module):
    
    def __init__(self, configs, device):
        super(LLM_TKESS, self).__init__()
        self.is_gpt = configs.is_gpt  # 1
        self.pretrain = configs.pretrain  # 1

        self.value_embedding = TokenEmbedding(c_in=configs.enc_in, d_model=configs.d_model)
        if configs.is_gpt:

            self.config = GPT2ConfigMy(
                n_embd=768, n_layer=12, n_head=12,
                lora_attn_dim=configs.lora_dim,
                lora_attn_alpha=configs.lora_alpha,
                lora_dropout=configs.lora_dropout,
                lora_layer=configs.lora_layer,
                gpt_layers=configs.gpt_layers
            )
            self.gpt2 = GPT2Model(self.config)


        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]  # 只取6层
        for i in range(configs.gpt_layers):
            self.gpt2.h[i].Imputation_adapter_attn = Adapter(configs.d_model,configs.adapter_dim,skip=True)
            self.gpt2.h[i].Imputation_adapter_FF = Adapter(configs.d_model,configs.adapter_dim,skip=True)

        self.in_layer = nn.Linear(configs.enc_in, configs.d_model)  # 16，768

        self.out_layer = nn.Linear(configs.d_model*configs.seq_len , 1)
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'wpe' in name or 'adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer,self.value_embedding):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x,x_mak_enc ,itr):
        B, L, M = x.shape

        outputs1 = self.value_embedding(x)
        outputs2 = self.in_layer(x)

        outputs = outputs1+outputs2

        if self.is_gpt:
            outputs,_ = self.gpt2(inputs_embeds=outputs)

        outputs = outputs.reshape(B,1,-1)
        outputs = self.out_layer(outputs)  # 64，96


        return outputs
