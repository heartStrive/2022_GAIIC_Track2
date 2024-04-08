import torch
import collections
import os
import sys

from bert_base_count1.finetuning.model import BertConcat as VisualBertLastCls
from bert_base_count1.finetuning.train_classifier import Config as VisualBertLastClsConfig

sys.path.append('./process_data')
model_path = '../data/model_data/bert_base_count2/'
models = []
for i in range(7,10):
    config = VisualBertLastClsConfig()
    model = VisualBertLastCls(config)
    weights = torch.load(model_path+'epoch_{}.pth'.format(str(i)))
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    print('*'*50)
    print('missing_keys:')
    print(missing_keys)
    print(unexpected_keys)
    print('*'*50)
    models.append(model)

worker_state_dict=[x.state_dict() for x in models]
weight_keys=list(worker_state_dict[0].keys())
fed_state_dict=collections.OrderedDict()
for key in weight_keys:
    key_sum=0
    for i in range(len(models)):
        key_sum+=worker_state_dict[i][key]
    fed_state_dict[key]=1.0*key_sum/len(models)
torch.save(fed_state_dict, '../best_model_2_fusion.pth')
