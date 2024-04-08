import torch
import random
import os
from torch import nn, optim
import torch.nn.functional as F
from transformers.activations import get_activation

from Config import *

class VisualBertLastCls(nn.Module):
    def __init__(self, config):
        super(VisualBertLastCls, self).__init__()
        self.n_classes = config.num_class
        self.device = config.device
        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True)
        self.bert_config.visual_embedding_dim=2048
        # self.bert_config.num_hidden_layers=12
        self.bert_model = MODELS[config.model](config=self.bert_config)
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


if __name__=='__main__':
    import os
    abs_path = os.path.abspath(os.path.dirname(__file__))

    class Config:
        def __init__(self):
            # 预训练模型路径
            self.modelId = 2
            self.model = "VisualBertLastCls"
            self.Stratification = False
            self.model_path = os.path.join(abs_path,'../../../data/pretrain_model/pretrained_visual_bert/')
            self.pretrained_bert_path = os.path.join(abs_path,'../../../data/pretrain_model/pretrained_bert/')
            self.imgs_path = os.path.join(abs_path,'../../../data/tmp_data/imgs/')

            self.num_class = 2
            self.dropout = 0.1
            self.MAX_LEN = 32
            self.epoch = 6
            self.learn_rate = 4e-5
            self.normal_lr = 2e-4
            self.batch_size = 256
            self.k_fold = 10
            self.seed = 2022
            
            self.device = torch.device('cuda')
            # self.device = torch.device('cpu')
            # 优化器选择
            self.use_swa = True
            self.swa_lr = 2e-5
            self.swa_start_step = 13500
            self.swa_steps = 300
            self.swa_model_valid = False

            # 损失函数选择
            self.focalloss = False
            # 对抗训练策略
            self.pgd = False
            self.fgm = True

    config = Config()
    model  = VisualBertLastCls(config).to(config.device) # visual_bert

    total = sum([param.nelement() for param in model.parameters()])
 
    print("Number of parameter: %.2fM" % (total/1e6))