
import sys

import torch

sys.path.append('../pretrain')
from tqdm import tqdm, trange
from model import *
from utils import *
import logging
import json
from NEZHA.modeling_nezha import *

def score(gt, pred):
    key_correct, key_all = 0, 0
    it_correct, it_all = 0, len(gt)
    for gt_one, pred_one in zip(gt, pred):

        for k1, k2 in zip(gt_one[:-1], pred_one[:-1]):
            key_correct += (k1 == k2)
            key_all += 1
        it_correct += (gt_one[-1] == pred_one[-1])
    if key_all == 0:
        return (1. * it_correct / it_all)
    return 0.5 * (1. * key_correct / key_all) + 0.5 * (1. * it_correct / it_all)

def eval_model(config, model, val_D):
    model.eval()
    with torch.no_grad():
        y_p = []
        y_l = []
        for input_ids, input_masks, segment_ids, targets in tqdm(val_D,disable=False):
            label_t = torch.tensor(targets['label'], dtype=torch.float).to(config.device)
            features = torch.tensor(targets['feature'], dtype=torch.float).to(config.device)

            logits = model(input_ids, input_masks, segment_ids, features)
            masks = (label_t != -1)

            for i,mask in enumerate(masks):
                pred_one = []
                for j in range(13):
                    if mask[j].item():
                        pred_one.append(int(torch.argmax(logits[j][i]).item()))
                y_p.append(pred_one)
            
            y_l.extend([i[mask].tolist() for i, mask in zip(label_t.cpu().numpy(), masks.cpu().numpy())])
        cur_score = score(y_l, y_p)
    return cur_score

if __name__=='__main__':
    MODEL_CLASSES = {
        'BertLSTM': BertLSTM,
        'BertForClass': BertForClass,
        'BertLastCls': BertLastCls,
        'BertLastTwoCls': BertLastTwoCls,
        'BertLastTwoClsPooler': BertLastTwoClsPooler,
        'BertLastTwoEmbeddings': BertLastTwoEmbeddings,
        'BertLastTwoEmbeddingsPooler': BertLastTwoEmbeddingsPooler,
        'BertLastFourCls': BertLastFourCls,
        'BertLastFourClsPooler': BertLastFourClsPooler,
        'BertLastFourEmbeddings': BertLastFourEmbeddings,
        'BertLastFourEmbeddingsPooler': BertLastFourEmbeddingsPooler,
        'BertDynCls': BertDynCls,
        'BertDynEmbeddings': BertDynEmbeddings,
        'BertRNN': BertRNN,
        'BertCNN': BertCNN,
        'BertRCNN': BertRCNN,
        'XLNet': XLNet,
        'Electra': Electra,
        'NEZHA': NEZHA,

    }

    class Config:
        def __init__(self):
            # 预训练模型路径
            self.modelId = 2
            self.model = "BertLSTM"
            self.Stratification = False
            # self.model_path = '../pretrain/bert_model/'
            self.model_path = '../../pretrained_bert/'
            self.imgs_path = '../../raw_data/imgs/'
            self.num_class = 13
            self.MAX_LEN = 32
            self.batch_size = 512
            self.k_fold = 10
            self.seed = 2022

            self.device = torch.device('cuda')
            # self.device = torch.device('cpu')

    config = Config()
    valid_data = loadData('../../raw_data/test_fine_sample.json')
    val_D = data_generator(valid_data, config, mode='test')
    model_path = './models/bert.pth'

    model = torch.load(model_path)

    cur_score = eval_model(config, model, val_D)
    print("score:{:.4f} \n".format(cur_score))

