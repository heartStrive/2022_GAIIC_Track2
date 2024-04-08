import torch
import random
import os
from torch import nn, optim
import torch.nn.functional as F
from transformers.activations import get_activation

from Config import *

class BertConcat(nn.Module):
    def __init__(self, config):
        super(BertConcat, self).__init__()
        self.n_classes = config.num_class

        config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
        self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
                                                                 output_hidden_states=True)
        self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        self.isDropout = True if 0 < config.dropout < 1 else False
        self.dropout = nn.Dropout(p=config.dropout)
        
        self.classifier1 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier2 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier3 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier4 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier5 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier6 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier7 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier8 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier9 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier10 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier11 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier12 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
        self.classifier13 = nn.Sequential(
                            nn.Linear(self.bert_config.hidden_size + 2048, self.bert_config.hidden_size),
                            nn.ReLU(True),
                            nn.Dropout(config.dropout),
                            nn.Linear(self.bert_config.hidden_size, self.n_classes))
                            
        self.feature_extractor1=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor2=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor3=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor4=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor5=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor6=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor7=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor8=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor9=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor10=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor11=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor12=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.feature_extractor13=nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
                                
    def forward(self, input_ids, input_masks, segment_ids, features):
        output = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                 attention_mask=input_masks)
        sequence_output = output[0]
        pooler_output = output[1]
        hidden_states = output[2]

        img_feature = features
        
        text_feature1 = self.feature_extractor1(pooler_output)
        fuse_feature1 = torch.cat((img_feature, text_feature1), dim=1)
        logit1 = self.classifier1(fuse_feature1)

        text_feature2 = self.feature_extractor2(pooler_output)
        fuse_feature2 = torch.cat((img_feature, text_feature2), dim=1)
        logit2 = self.classifier2(fuse_feature2)
        
        text_feature3 = self.feature_extractor3(pooler_output)
        fuse_feature3 = torch.cat((img_feature, text_feature3), dim=1)
        logit3 = self.classifier3(fuse_feature3)
        
        text_feature4 = self.feature_extractor4(pooler_output)
        fuse_feature4 = torch.cat((img_feature, text_feature4), dim=1)
        logit4 = self.classifier4(fuse_feature4)
        
        text_feature5 = self.feature_extractor5(pooler_output)
        fuse_feature5 = torch.cat((img_feature, text_feature5), dim=1)
        logit5 = self.classifier5(fuse_feature5)
        
        text_feature6 = self.feature_extractor6(pooler_output)
        fuse_feature6 = torch.cat((img_feature, text_feature6), dim=1)
        logit6 = self.classifier6(fuse_feature6)
        
        text_feature7 = self.feature_extractor7(pooler_output)
        fuse_feature7 = torch.cat((img_feature, text_feature7), dim=1)
        logit7 = self.classifier7(fuse_feature7)
        
        text_feature8 = self.feature_extractor8(pooler_output)
        fuse_feature8 = torch.cat((img_feature, text_feature8), dim=1)
        logit8 = self.classifier8(fuse_feature8)
        
        text_feature9 = self.feature_extractor9(pooler_output)
        fuse_feature9 = torch.cat((img_feature, text_feature9), dim=1)
        logit9 = self.classifier9(fuse_feature9)
        
        text_feature10 = self.feature_extractor10(pooler_output)
        fuse_feature10 = torch.cat((img_feature, text_feature10), dim=1)
        logit10 = self.classifier10(fuse_feature10)
        
        text_feature11 = self.feature_extractor11(pooler_output)
        fuse_feature11 = torch.cat((img_feature, text_feature11), dim=1)
        logit11 = self.classifier11(fuse_feature11)
        
        text_feature12 = self.feature_extractor12(pooler_output)
        fuse_feature12 = torch.cat((img_feature, text_feature12), dim=1)
        logit12 = self.classifier12(fuse_feature12)
        
        text_feature13 = self.feature_extractor13(pooler_output)
        fuse_feature13 = torch.cat((img_feature, text_feature13), dim=1)
        logit13 = self.classifier13(fuse_feature13)
        
        return (logit1,logit2,logit3,logit4,logit5,logit6,logit7,logit8,logit9,logit10,logit11,logit12,logit13)

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

            self.num_class = 2
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
    model  = BertConcat(config).to(config.device) # visual_bert

    total = sum([param.nelement() for param in model.parameters()])
 
    print("Number of parameter: %.2fM" % (total/1e6))