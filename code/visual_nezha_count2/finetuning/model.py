import torch
import random
import os
import sys
sys.path.append('./visual_nezha_count2/finetuning')
from torch import nn, optim
import torch.nn.functional as F
from transformers.activations import get_activation
from NEZHA.modeling_nezha import NeZhaEncoder
from NEZHA.configuration_nezha import NeZhaConfig
from Config import *
abs_path = os.path.abspath(os.path.dirname(__file__))

from visualbert_model import VisualNezhaModel

class VisualBertLastCls(nn.Module):
    def __init__(self, config):
        super(VisualBertLastCls, self).__init__()
        self.n_classes = config.num_class
        self.device = config.device
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True)
        self.bert_config.visual_embedding_dim=2048
        self.bert_model = VisualNezhaModel(config=self.bert_config)
        # 定义nezha编码器
        self.nezha_config = NeZhaConfig.from_pretrained(os.path.join(abs_path,'../../../data/pretrain_model/pretrained_nezha/'),
                                                        output_hidden_states=True,
                                                        output_attentions=True)
        self.bert_model.encoder = NeZhaEncoder(self.nezha_config)        
        
        self.drop_out = config.dropout

        self.classifier1 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier2 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier3 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier4 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier5 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier6 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier7 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier8 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier9 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier10 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier11 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier12 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )
        self.classifier13 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size, 512),
                            nn.LayerNorm(512),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(512, 256),
                            nn.LayerNorm(256),
                            nn.LeakyReLU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(256, self.n_classes)
                            )

    def forward(self, input_ids, input_masks, segment_ids, features):
        visual_embeds = features.unsqueeze(1)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(self.device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(self.device)
        outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    visual_embeds=visual_embeds,
                    visual_attention_mask=visual_attention_mask,
                    visual_token_type_ids=visual_token_type_ids
        )
        pooler_output = outputs.pooler_output
        logit1 = self.classifier1(pooler_output)
        logit2 = self.classifier2(pooler_output)
        logit3 = self.classifier3(pooler_output)
        logit4 = self.classifier4(pooler_output)
        logit5 = self.classifier5(pooler_output)
        logit6 = self.classifier6(pooler_output)
        logit7 = self.classifier7(pooler_output)
        logit8 = self.classifier8(pooler_output)
        logit9 = self.classifier9(pooler_output)
        logit10 = self.classifier10(pooler_output)
        logit11 = self.classifier11(pooler_output)
        logit12 = self.classifier12(pooler_output)
        logit13 = self.classifier13(pooler_output)

        return (logit1,logit2,logit3,logit4,logit5,logit6,logit7,logit8,logit9,logit10,logit11,logit12,logit13)


