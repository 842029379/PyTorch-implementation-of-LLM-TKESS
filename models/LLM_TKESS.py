import numpy as np
import torch
import torch.nn as nn
from torch import optim
from layers.Embed import  DataEmbedding_wo_time, TemporalEmbedding, TimeFeatureEmbedding
# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from models.Model import GPT2Model, GPT2ConfigMy
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

class LLM_TKESS(nn.Module):

    def __init__(self, configs, device):
        super(LLM_TKESS, self).__init__()
        self.is_gpt = configs.is_gpt  # 1
        self.pretrain = configs.pretrain  # 1
        self.seq_len = configs.seq_len

        if configs.is_gpt:
            if configs.pretrain:

                self.config = GPT2ConfigMy(
                    n_embd=768, n_layer=12, n_head=12,
                    lora_attn_dim=configs.lora_dim,
                    lora_attn_alpha=configs.lora_alpha,
                    lora_dropout=configs.lora_dropout,
                    lora_layer = configs.lora_layer,
                    gpt_layers = configs.gpt_layers
                )
                self.gpt2 = GPT2Model(self.config)
                if configs.init_checkpoint is not None:
                    print('loading model pretrained weight.')
                    state_dict = torch.load(configs.init_checkpoint)
                    for n, p in self.gpt2.named_parameters():
                        if n not in state_dict:
                            state_dict[n] = p
                    self.gpt2.load_state_dict(state_dict, strict=False)

            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())

            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))

        self.out_layer_auto = nn.Linear(configs.d_model, configs.enc_in)
        self.in_layer = nn.Linear(configs.enc_in, configs.d_model)
        self.value_embedding = TokenEmbedding(c_in=configs.enc_in, d_model=configs.d_model)
        if configs.freeze and configs.pretrain:

            # layer = 0
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                # if 'ln' in name or 'wpe' in name or 'lora_' in name:  # 只微调归一化层以及位置嵌入层
                if 'wpe' in name or 'lora_' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for i, (name,param) in enumerate(self.gpt2.named_parameters()):
                print("{:30s} : {}".format(name,param.requires_grad))


        for layer in (self.gpt2, self.out_layer_auto, self.in_layer,self.value_embedding):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    def forward(self, x, x_mark):
        B, L, M = x.shape

        x_input = x

        outputs1 = self.value_embedding(x)
        outputs2 = self.in_layer(x)

        outputs = outputs1 + outputs2


        x_label = x_input


        if self.is_gpt:
            outputs, _ = self.gpt2(inputs_embeds=outputs)

        outputs = self.out_layer_auto(outputs)



        shift_outputs = outputs[:, :-1, ].contiguous()
        shift_labels = x_label[:, 1:].contiguous()

        return shift_outputs, shift_labels
