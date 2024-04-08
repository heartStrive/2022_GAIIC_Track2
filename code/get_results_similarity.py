
import sys
sys.path.append('../pretrain')
import torch
from tqdm import tqdm, trange

import logging
import json
from utils_test import data_generator, write_results, loadData

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
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


def eval_model(config, model, val_D):
    results = []
    labels = []
    model.eval()
    with torch.no_grad():
        for input_ids, input_masks, segment_ids, targets in tqdm(val_D,disable=False):
            label_t = torch.tensor(targets['match'], dtype=torch.float).to(config.device)
            features = torch.tensor(targets['feature'], dtype=torch.float).to(config.device)

            logits = model(input_ids, input_masks, segment_ids, features)
            
            queries = targets['query']
            for idx, img_n, query, y_l in zip(range(len(targets['img_name'])),targets['img_name'], queries, targets['match']):
                pred_item = {'img_name': img_n, 'match': {}}
                gt_item = {'img_name': img_n, 'match': {}}
                for q in query:
                    pred_item['match'][q] = int(torch.argmax(logits[label2id[q]][idx]).item())
                    gt_item['match'][q] = y_l[label2id[q]]
                results.append(pred_item)
                labels.append(gt_item)
        # all_score(labels,results)
        cur_score = score(labels, results)
    
    write_results(results,'./results.txt')
    return cur_score

if __name__=='__main__':

    class Config:
        def __init__(self):
            # 预训练模型路径
            self.model = "BertLSTM"
            # self.imgs_path = './raw_data/imgs/'
            self.num_class = 13 # 12个关键属性以及图文
            self.MAX_LEN = 32
            self.batch_size = 32
            self.seed = 2022
            self.device = torch.device('cuda')



    # model_path_list =  ['../data/best_model_1.pth', '../data/best_model_2.pth','../data/best_model_3.pth','../data/best_model_4.pth','../data/best_model_5.pth','../data/best_model_6.pth','../data/best_model_7.pth']

    # result_path_list =  ['../data/submission/results_best_model_7.txt', '../data/submission/results_best_model_8.txt','../data/submission/results_best_model_9.txt','../data/submission/results_best_model_10.txt','../data/submission/results_best_model_11.txt','../data/submission/results_best_model_12.txt','../data/submission/fusion_results_95.317.txt']

    # result_path_list = ['../data/submission/results_0.5.txt', '../data/submission/results_0.4.txt', '../data/submission/results_0.6.txt']
    
    # result_path_list =  ['../data/submission/results_best_model_7.txt', '../data/submission/results_best_model_8.txt','../data/submission/results_best_model_9.txt','../data/submission/results_best_model_10.txt','../data/submission/results_best_model_11.txt','../data/submission/results_best_model_12.txt','../data/submission/results_92.60.txt']

    result_path_list =  ['../data/submission/results_best_model_1_fusion.txt','../data/submission/results_best_model_1.txt']
    
    result_arr = np.zeros([len( result_path_list),len( result_path_list)])
    for i in range(len(result_path_list)):
        for j in range(len(result_path_list)):
            labels = []
            with open(result_path_list[i],'r',encoding='utf-8') as f:
                texts = f.readlines()
                for data in texts:
                    data=json.loads(data)
                    labels.append(data)
            results = []
            with open(result_path_list[j],'r',encoding='utf-8') as f:
                texts = f.readlines()
                for data in texts:
                    data=json.loads(data)
                    results.append(data)

            cur_score = score(labels, results)
            result_arr[i][j] = cur_score
            # print(cur_score)
        print(result_arr[i])


    print(result_arr)
