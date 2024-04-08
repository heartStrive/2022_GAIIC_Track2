import torch
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
import numpy as np
import os
import json
import random
from Config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
jieba.load_userdict('../data/hand_data/user_dict.txt')
abs_path = os.path.abspath(os.path.dirname(__file__))
attr_to_attrvals_path = os.path.join(abs_path,'../data/contest_data/attr_to_attrvals.json')
# 将结果写入txt文件
def write_results(results, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for line_id, res in enumerate(results):
            text = json.dumps(res, ensure_ascii=False)
            f.write("%s\n" % text)

def load_attr_to_attrvals(attr_to_attrvals_path):
    attr_to_attrvals_dict = {}
    f = open(attr_to_attrvals_path, encoding='utf-8')
    json_obj = json.load(f)
    id_count = 0
    for attr,attrvals_list_1 in json_obj.items():
        attr_to_attrvals_dict[attr] = {}
        for attrvals in attrvals_list_1:
            attrvals_list_2 = attrvals.split('=')
            for attrval in attrvals_list_2:
                attr_to_attrvals_dict[attr][attrval] = id_count
            id_count += 1
    return attr_to_attrvals_dict

def loadData(path, num_class):
    '''
    params:
        path:数据集路径

    '''
    jieba.load_userdict('../data/hand_data/user_dict.txt')  #jieba分词设置自定义用户字典
    attr_to_attrvals_dict = load_attr_to_attrvals('../data/contest_data/attr_to_attrvals.json')
    allData = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            data = json.loads(line)
            """
            'img_name', 'title', 'query', 'feature'
            """
            title = data['title']
            key_attr = {}
            for attr in data['query']:
                if attr=='图文':
                    continue
                attrval_list = list(attr_to_attrvals_dict[attr].keys())
                attrval_list.sort(key=lambda x:len(x),reverse=True)
                for attrval in attrval_list:
                    if attrval in title:
                        key_attr[attr] = attrval
                        title=title.replace(attrval,'')
                        break
            data['key_attr'] = key_attr

            if 'match' not in data.keys():
                data['match'] = [-1]*num_class # 真正的测试数据是没match的，防止报错
            allData.append(data)
    return allData


############给关键属性包括图文一个id，方便制作标签####################
def get_key_attrs(json_path):
    '''params:
           json_path: 关键词表的路径

    return: 关键属性列表
    '''
    with open(json_path, 'r', encoding='utf-8') as f:
        attrvals = f.read()
    attrvals = json.loads(attrvals)
    label_list = list(attrvals.keys())
    return label_list

label_list = get_key_attrs(attr_to_attrvals_path) + ['图文']
label2id = {l: i for i, l in enumerate(label_list)}

############给关键值属性包括图文一个id####################
def get_attrvals(json_path):
    '''params:
           json_path: 关键词表的路径

    return: 关键属性列表
    '''
    with open(json_path, 'r', encoding='utf-8') as f:
        attrvals = f.read()
    json_obj = json.loads(attrvals)
    attrval_list = []
    for attr,attrvals_list_1 in json_obj.items():
        for attrvals in attrvals_list_1:
            attrvals_list_2 = attrvals.split('=')
            for attrval in attrvals_list_2:
                attrval_list.append(attrval)
    return attrval_list


attrval_list = get_attrvals(attr_to_attrvals_path)
# attrval_2id = get_attrval_2id_dict(attr_to_attrvals_path)

attrval_2id = {l: i+1 for i, l in enumerate(attrval_list)}


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


def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.tensor(ls,device='cuda') if returnTensor else ls

def paddingMask(ls:list,val,returnTensor=False):
    maxLen=max([len(i) for item in ls for i in item]) # batch * 13 * n
    for i in range(len(ls)):
        for j in range(len(ls[i])):
            ls[i][j]=ls[i][j]+[val]*(maxLen-len(ls[i][j]))+[1] # 1 for visual_embed.
    return torch.tensor(ls,device='cuda') if returnTensor else ls

def fastTokenizer(a,maxLen,tk):
    if isinstance(a,str):
        a=tk.tokenize(a)
    word_list = a
    a=tk.convert_tokens_to_ids(a)
    maxLen-=2 # 空留给cls sep
    assert maxLen>=0
    length = len(a)
    if length > maxLen:  # 需要截断
        a = a[:length]
        word_list = word_list[:length]

    input_ids=[tk.cls_token_id]+a+[tk.sep_token_id]
    token_type_ids=[0]*(len(a)+2)
    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'word_list': word_list}

class data_generator:
    def __init__(self, data, config, shuffle=False):
        self.data = data
        # self.imgs_dir = config.imgs_path

        self.batch_size = config.batch_size
        self.max_length = config.MAX_LEN
        self.shuffle = shuffle
        vocab = 'vocab.txt' if os.path.exists(config.model_path + 'vocab.txt') else 'spiece.model'

        # self.tokenizer = TOKENIZERS[config.model].from_pretrained(config.model_path + vocab)
        self.tokenizer = TOKENIZERS['VisualBertClassification'].from_pretrained(config.model_path + vocab)  # 只用BertTokenizer

        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data)))
        if self.shuffle:
            np.random.shuffle(idxs)

        input_ids, input_masks, segment_ids, targets = [], [], [], {'img_name':[],'query':[],'feature':[], 'match':[], 'attrval_ids':[]}
        for index, i in enumerate(idxs):
            text = self.data[i]['title']
            tkRes = fastTokenizer(text, self.max_length, self.tokenizer)
            input_ids.append(tkRes['input_ids'])
            segment_ids.append(tkRes['token_type_ids'])
            targets['img_name'].append(self.data[i]['img_name'])
            targets['query'].append(self.data[i]['query'])
            targets['match'].append(self.data[i]['match'])
            targets['feature'].append(self.data[i]['feature'])

            # 获取关键属性起始索引和结束索引
            word_list = tkRes['word_list']
            attrval_ids = [[0]*len(tkRes['input_ids']) for i in range(len(label2id))]
            for attr in self.data[i]['key_attr'].keys():
                attrval = self.data[i]['key_attr'][attr]
                attrval_list = self.tokenizer.tokenize(attrval)
                for idx in range(len(word_list)-len(attrval_list)+1):
                    if word_list[idx:idx+len(attrval_list)]==attrval_list:
                        attrval_ids[label2id[attr]][idx+1:idx+1+len(attrval_list)] = [1]*len(attrval_list)  # 0 for cls token.
                        break

            targets['attrval_ids'].append(attrval_ids)

            if len(input_ids) == self.batch_size or i == idxs[-1]:
                input_ids = paddingList(input_ids, 0, returnTensor=True)  # 动态padding
                segment_ids = paddingList(segment_ids, 0, returnTensor=True)
                targets['attrval_ids'] = paddingMask(targets['attrval_ids'], 0, returnTensor=False)
                input_masks = (input_ids != 0)
                yield input_ids, input_masks, segment_ids, targets
                input_ids, input_masks, segment_ids, targets = [], [], [], {'img_name':[], 'query':[], 'feature':[], 'match':[],'attrval_ids':[]}


class single_data_generator:
    def __init__(self, config, shuffle=False):
        self.batch_size = config.batch_size
        self.max_length = config.MAX_LEN
        self.shuffle = shuffle
        vocab = 'vocab.txt' if os.path.exists(config.model_path + 'vocab.txt') else 'spiece.model'
        # self.tokenizer = TOKENIZERS[config.model].from_pretrained(config.model_path + vocab)
        self.tokenizer = TOKENIZERS['VisualBertClassification'].from_pretrained(config.model_path + vocab)  # 只用BertTokenizer


    def generate(self, data):
        input_ids, input_masks, segment_ids, targets = [], [], [], {'img_name':[],'query':[],'feature':[]}

        text = data['title']
        text = fastTokenizer(text, self.max_length, self.tokenizer)
        input_ids.append(text['input_ids'])
        segment_ids.append(text['token_type_ids'])
        input_masks.append([1] * len(text['input_ids']))  # bs为1时无padding，全1

        targets['img_name'].append(data['img_name'])
        targets['query'].append(data['query'])
        targets['feature'].append(data['feature'])

        return input_ids, input_masks, segment_ids, targets

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]



class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss
    for well-classified examples (p>0.5) putting more
    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index,
    should be specific when alpha is float
    :param size_average: (bool, optional) By default,
    the losses are averaged over each loss element in the batch.
    """
    def __init__(self, num_class, alpha=None, gamma=2,
                smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        else:
            raise TypeError('Not support alpha type')
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def f1_match(y_true,y_pred):
    acc = sum(y_pred & y_true) / (sum(y_pred))
    rec = sum(y_pred & y_true) / (sum(y_true))

    return 2 * acc * rec /(acc + rec)