import os
import sys
from bert_base_count1.finetuning.model import BertConcat as BertConcat1
from visual_bert_count1.finetuning.model import VisualBertLastCls as VisualBertLastCls1
from visual_bert_count2.finetuning.model import VisualBertLastCls as VisualBertLastCls2
from visual_nezha_count2.finetuning.model import VisualBertLastCls as VisualNezhaLast2
from visual_roberta_count1.finetuning.model import VisualBertLastCls as VisualRobertaLastCls1
from bert_base_lstm_mult_count1.finetuning.model import BertLSTMMult as BertLSTMMult

from bert_base_count1.finetuning.train_classifier import Config as BertConcatConfig1
from visual_bert_count1.finetuning.train_classifier import Config as VisualBertLastClsConfig1
from visual_bert_count2.finetuning.train_classifier import Config as VisualBertLastClsConfig2
from visual_nezha_count2.finetuning.train_classifier import Config as VisualNezhaLastClsConfig2
from visual_roberta_count1.finetuning.train_classifier import Config as VisualRobertaLastClsConfig1
from bert_base_lstm_mult_count1.finetuning.train_classifier import Config as BertLSTMMultConfig

sys.path.append('./process_data')
import torch
from tqdm import tqdm, trange
import argparse
import logging
import json
from utils_test import data_generator, write_results, loadData
from format_title import format_title
import torch.nn.functional as F
abs_path = os.path.abspath(os.path.dirname(__file__))

MODEL_CLASSES = {
    'best_model_1.pth':BertConcat1,
    'best_model_2.pth':BertLSTMMult,
    'best_model_3.pth':VisualBertLastCls1,
    'best_model_4.pth':VisualBertLastCls2,
    'best_model_5.pth':VisualNezhaLast2,
    'best_model_6.pth':VisualRobertaLastCls1,
}

CONFIGS = {
    'best_model_1.pth':BertConcatConfig1,
    'best_model_2.pth':BertLSTMMultConfig,
    'best_model_3.pth':VisualBertLastClsConfig1,
    'best_model_4.pth':VisualBertLastClsConfig2,
    'best_model_5.pth':VisualNezhaLastClsConfig2,
    'best_model_6.pth':VisualRobertaLastClsConfig1
}

def get_key_attrs(json_path):
    '''
    :params json_path: 关键词表的路径
    :return 关键属性列表
    '''
    with open(json_path, 'r', encoding='utf-8') as f:
        attrvals = f.read()
    attrvals = json.loads(attrvals)
    label_list = list(attrvals.keys())
    return label_list

label_list = get_key_attrs('../data/contest_data/attr_to_attrvals.json') + ['图文']
label2id = {l: i for i, l in enumerate(label_list)}
print(label2id)
def score(gt, pred):
    gt = sorted(gt, key=lambda it: it['img_name'])
    pred = sorted(pred, key=lambda it: it['img_name'])

    key_correct, key_all = 0, 0
    it_correct, it_all = 0, len(gt)
    for gt_one, pred_one in zip(gt, pred):
        assert gt_one['img_name'] == pred_one['img_name']
        gt_match = gt_one['match']
        pred_match = pred_one['match']
        for key in gt_match.keys():
            assert key in pred_match.keys()
            if key in '图文':
                continue
            key_correct += (gt_match[key] == pred_match[key])
            key_all += 1
        it_correct += (gt_match['图文'] == pred_match['图文'])
    if key_all == 0:
        return (1. * it_correct / it_all)
    print('key_attr_score = ',0.5 * (1. * key_correct / key_all), 'image_text_score = ',0.5 * (1. * it_correct / it_all))
    return 0.5 * (1. * key_correct / key_all) + 0.5 * (1. * it_correct / it_all)

def eval_models_averaging(config, models, val_D, val_D2):

    results = []
    labels = []
    
    weights = [ torch.tensor(0.15,dtype=float).to(config.device),
                torch.tensor(0.15,dtype=float).to(config.device),
                torch.tensor(0.20,dtype=float).to(config.device),
                torch.tensor(0.15,dtype=float).to(config.device),
                torch.tensor(0.15,dtype=float).to(config.device),
                torch.tensor(0.20,dtype=float).to(config.device),
            ]
    
    
    with torch.no_grad():
        for data in tqdm(val_D,disable=False):
            input_ids, input_masks, segment_ids, targets = data
            label_t = torch.tensor(targets['match'], dtype=torch.float).to(config.device)
            features = torch.tensor(targets['feature'], dtype=torch.float).to(config.device)

            tmp_logits = []
            for idx in range(len(models)):
                y_pred = models[idx](input_ids, input_masks, segment_ids, features)
                tmp_logits.append(y_pred)
            
            
            # 先softmax
            logits = []
            for i in range(13):
                logit_one = weights[0]*F.softmax(tmp_logits[0][i],dim=1)
                for j in range(1,len(models)):
                    if j==1:
                        logit_half = F.sigmoid(tmp_logits[j][:,i])
                        logit_one += weights[j]*F.softmax(torch.cat([(1.0-logit_half).unsqueeze(1),logit_half.unsqueeze(1)],dim=1),dim=1)
                    else:
                        logit_one+=weights[j]*F.softmax(tmp_logits[j][i],dim=1)
                logits.append(logit_one)
            
            queries = targets['query']
            for idx, img_n, query, y_l in zip(range(len(targets['img_name'])),targets['img_name'], queries, targets['match']):
                pred_item = {'img_name': img_n, 'match': {}}
                gt_item = {'img_name': img_n, 'match': {}}
                for q in query:
                    pred_item['match'][q] = int(torch.argmax(logits[label2id[q]][idx]).item())
                    # pred_item['match'][q] = int(logits[label2id[q]][idx][1]>0.4) # 备用
                    gt_item['match'][q] = y_l[label2id[q]]
                results.append(pred_item)
                labels.append(gt_item)

    
        cur_score = score(labels, results)
    
    write_results(results,'../data/submission/results.txt')
    return cur_score

if __name__=='__main__':
    # 获取测试集路径
    parser = argparse.ArgumentParser(description='para transfer')
    parser.add_argument('--test_data_path', type=str, default='../data/contest_data/preliminary_testB.txt')
    args = parser.parse_args()

    # 删除停用词
    test_data_path = os.path.join(abs_path,'../data/contest_data', args.test_data_path.split('/')[-1])
    user_dict_path = os.path.join(abs_path,'../data/hand_data/user_dict.txt')
    stop_words_path = os.path.join(abs_path,'../data/hand_data/stop_words.txt')
    
    format_title(test_data_path)
    print('*'*50)
    print(test_data_path)
    test_data_path = test_data_path.replace('.txt', '.json')

    class TestConfig:
        def __init__(self):
            # 预训练模型路径
            self.num_class = 13 # 12个关键属性以及图文
            self.MAX_LEN = 32
            self.batch_size = 32
            self.seed = 2022
            self.device = torch.device('cuda')
            self.model_path = '../data/pretrain_model/pretrained_bert/' # 词表路径

    
    class ModelConfig:
        def __init__(self):
             # 预训练模型路径
            self.modelId = 1
            self.model = "BertConcat"
            self.Stratification = False
            self.model_path = '../data/pretrain_model/pretrained_bert/'

            self.num_class = 2
            self.dropout = 0.1
            self.MAX_LEN = 32
            self.epoch = 10
            self.learn_rate = 4e-5
            self.normal_lr = 1e-4
            self.batch_size = 32
            self.k_fold = 10
            self.seed = 2022

            self.device = torch.device('cuda')
        
        
    test_config = TestConfig()
    valid_data = loadData(test_data_path, 13)
    val_D = data_generator(valid_data, test_config)
    
    model_path_list =  ['../data/best_model_1.pth','../data/best_model_2.pth','../data/best_model_3.pth','../data/best_model_4.pth','../data/best_model_5.pth','../data/best_model_6.pth']
    models = []
    #根据模型参数加载模型
    for model_path  in model_path_list:
        model_config = CONFIGS[model_path.split('/')[-1]]()

        model = MODEL_CLASSES[model_path.split('/')[-1]](model_config).to(model_config.device)
        weights = torch.load(model_path)
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        print('*'*50)
        print('missing_keys:')
        print(missing_keys)
        print('unexcepted_keys:')
        print(unexpected_keys)
        print('*'*50)
        models.append(model.eval())
        
    eval_models_averaging(test_config,models,val_D,val_D)

    labels = []
    with open('./../data/submission/results_95.1.txt','r',encoding='utf-8') as f:
        texts = f.readlines()
        for data in texts:
            data=json.loads(data)
            labels.append(data)
    results = []
    with open('../data/submission/results.txt','r',encoding='utf-8') as f:
        texts = f.readlines()
        for data in texts:
            data=json.loads(data)
            results.append(data)

    cur_score = score(labels, results)
    print(cur_score)