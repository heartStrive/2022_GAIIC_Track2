import torch
import random
import os
from torch import nn, optim
import torch.nn.functional as F
from transformers.activations import get_activation

from Config import *

class BertLSTMMult(nn.Module):
    def __init__(self, config):
        super(BertLSTMMult, self).__init__()
        self.n_classes = config.num_class

        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True)
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        self.isDropout = True if 0 < config.dropout < 1 else False
        self.dropout = nn.Dropout(p=config.dropout)
        # self.classifier = nn.Sequential(
        #                     nn.Linear(self.bert_config.hidden_size * 2, self.bert_config.hidden_size * 4),
        #                     nn.ReLU(True),
        #                     nn.Dropout(config.dropout),
        #                     nn.Linear(self.bert_config.hidden_size * 4, self.bert_config.hidden_size),
        #                     nn.ReLU(True),
        #                     nn.Dropout(config.dropout),
        #                     nn.Linear(self.bert_config.hidden_size, self.n_classes)
        #                     )
        self.classifier = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size * 2, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes)
                            )
        self.bilstm = nn.LSTM(input_size=self.bert_config.hidden_size,
                              hidden_size=self.bert_config.hidden_size, batch_first=True, bidirectional=True)
        self.highway = nn.Linear(2048, self.bert_config.hidden_size)


    def forward(self, input_ids, input_masks, segment_ids, features):
        output = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        sequence_output = output[0]
        pooler_output = output[1]
        hidden_states = output[2]
        sequence_output = torch.cat([self.highway(features).unsqueeze(1),sequence_output], dim=1)
        output_hidden, _ = self.bilstm(sequence_output)
        concat_out = torch.mean(output_hidden, dim=1)
        
        if self.isDropout:
            concat_out = self.dropout(concat_out)
        logit = self.classifier(concat_out)
        return logit


if __name__=='__main__':
    import os
    abs_path = os.path.abspath(os.path.dirname(__file__))

    class Config:
        def __init__(self):
            # 预训练模型路径
            self.modelId = 2
            self.model = "BertConcat"
            self.Stratification = False
            self.model_path = os.path.join(abs_path,'../../../data/pretrain_model/pretrained_bert/')
            self.pretrained_bert_path = os.path.join(abs_path,'../../../data/pretrain_model/pretrained_bert/')
            self.imgs_path = os.path.join(abs_path,'../../../data/tmp_data/imgs/')

            self.num_class = 13
            self.dropout = 0.1
            self.MAX_LEN = 32
            self.epoch = 10
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
    model  = BertLSTMMult(config).to(config.device) # visual_bert

    total = sum([param.nelement() for param in model.parameters()])
 
    print("Number of parameter: %.2fM" % (total/1e6))#79.01M