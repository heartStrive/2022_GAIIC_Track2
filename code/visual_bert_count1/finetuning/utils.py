import torch
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
import numpy as np
import os
import random
from Config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import jieba
from feature_augment import DiscontinuousRandomErasing, ContinuousRandomErasing, CenterFlip
import sys
import copy
abs_path = os.path.abspath(os.path.dirname(__file__))
jieba.load_userdict(os.path.join(abs_path,'../../../data/hand_data/user_dict.txt'))  #jieba分词设置自定义用户字典

# 统计数据集正负样本数量
def data_cnt(path):
    positive_cnt=0
    negtive_cnt=0
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            data = json.loads(line)
            """
            'img_name', 'title', 'key_attr', 'match', 'feature'
            """
           
            positive_cnt+=data['match'][-1]==1
            negtive_cnt+=data['match'][-1]==0
    print('***********************')
    print(path)
    print('positive_cnt = ',positive_cnt,'negtive_cnt=',negtive_cnt)

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

label_list = get_key_attrs('../../../data/contest_data/attr_to_attrvals.json') + ['图文']
label2id = {l: i for i, l in enumerate(label_list)}

def loadData(path):
    '''
    params:
        path:数据集路径
        mode:如果是coarse就只读取图文标签，如果是fine就读取所有标签。
    '''
    allData=[]
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            data = json.loads(line)
            """
            'img_name', 'title', 'key_attr', 'match', 'feature'
            """
            img_name = data['img_name']
            title = data['title']            
            # key_attr = data['key_attr'] if mode=='fine' else {}
            key_attr = data['key_attr']
            match = data['match']
            # feature = data['feature']

            # 判断下标签是不是已经从dict转换成list
            if isinstance(match, dict):
                label = [-1] * len(label2id)
                # if mode == 'fine':
                for key in key_attr:
                    label[label2id[key]] = match[key]
                label[label2id['图文']] = match['图文']
            else:
                label=match

            sample = {}
            sample['img_name'] = img_name
            sample['title'] = title
            sample['key_attr'] = key_attr
            sample['match'] = label
            # sample['feature'] = feature

            allData.append(sample)
    return allData

from generate_pseudo_samples import get_title_attr_to_attrvals_dict, load_attr_to_attrvals
attr_to_attrvals_path = '../../../data/contest_data/attr_to_attrvals.json'
attr_to_attrvals_dict = load_attr_to_attrvals(attr_to_attrvals_path)

def replace_same_type_attrval(json_data):
    
    title_attr_to_attrvals_dict = get_title_attr_to_attrvals_dict(json_data['title'],json_data['key_attr'],attr_to_attrvals_dict)

    temp_title_attr_to_attrvals_dict = {}
    for attr,attrval in title_attr_to_attrvals_dict.items():
        attrval_id = attr_to_attrvals_dict[attr][attrval]  # 获取此关键属性的id
        equal_attrvals_list = [attrval for attrval, id in attr_to_attrvals_dict[attr].items() if id == attrval_id]
        if len(equal_attrvals_list) >= 2:
            temp_title_attr_to_attrvals_dict[attr] = attrval
    title_attr_to_attrvals_dict = temp_title_attr_to_attrvals_dict

    # 根据modify_num随机筛选关键属性
    title_attr_num = len(list(title_attr_to_attrvals_dict.keys()))  # 得到标题的 存在等价属性的 关键属性数量
    modify_num = random.randint(1,max(1,title_attr_num))
    title_attr_list = []
    if title_attr_num >= 1:
        
        random_attr_index_list = random.sample(range(0, title_attr_num), modify_num)
        title_attr_list = [list(title_attr_to_attrvals_dict.keys())[i] for i in random_attr_index_list]
        
        # 得到筛选过后的title_attr_to_attrvals_dict
        title_attr_to_attrvals_dict = dict([(attr,title_attr_to_attrvals_dict[attr]) for attr in title_attr_list])
        
        # print(title_attr_to_attrvals_dict)

        for attr,attrval in title_attr_to_attrvals_dict.items():
            attrval_id = attr_to_attrvals_dict[attr][attrval]  # 获取此关键属性的id
            equal_attrvals_list = [attrval for attrval,id in attr_to_attrvals_dict[attr].items() if id == attrval_id]  # 找到等价属性值列表
            while 1:
                random_num =  random.randint(0,len(equal_attrvals_list)-1)  # 从等价关键属性列表中，随机取出与原attrval不同的等价关键属性
                equal_attrval = equal_attrvals_list[random_num]
                if equal_attrval != attrval:  # 等价属性与原属性不同时结束循环
                    break
            title_attr_to_attrvals_dict[attr] = equal_attrval
            json_data['title'] = json_data['title'].replace(attrval,equal_attrval)  # 修改title
            # 修改key_attr字段
            if attr in json_data['key_attr'].keys():
                json_data['key_attr'][attr] = equal_attrval
    return json_data

def change_title(title):
    '''
    0.5的概率翻转title
    0.5的概率随机打乱title
    '''
    word_list = list(jieba.cut(title))
    prob = random.random()
    if prob>0.5:
        word_list.reverse()
        title = ''.join(word_list)
    else:
        random_word_index_list = random.sample(range(0, len(word_list)), len(word_list))
        new_word_list = [ word_list[random_word_index] for random_word_index in random_word_index_list]
        title = ''.join(new_word_list)
    return title

def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.tensor(ls,device='cuda') if returnTensor else ls

def fastTokenizer(a,maxLen,tk):
    if isinstance(a,str):
        a=tk.tokenize(a)
    a=tk.convert_tokens_to_ids(a)
    maxLen-=2 # 空留给cls sep
    assert maxLen>=0
    length = len(a)
    if length > maxLen:  # 需要截断
        a = a[:length]

    input_ids=[tk.cls_token_id]+a+[tk.sep_token_id]
    token_type_ids=[0]*(len(a)+2)
    return {'input_ids': input_ids, 'token_type_ids': token_type_ids}

class data_generator:
    def __init__(self, data, config, mode='train', shuffle=False):
        self.data = data
        self.imgs_dir = config.imgs_path
        self.batch_size = config.batch_size
        self.max_length = config.MAX_LEN
        self.shuffle = shuffle
        self.mode = mode

        self.continuous_erasing = ContinuousRandomErasing()
        self.discontinuous_erasing = DiscontinuousRandomErasing()
        self.center_flip = CenterFlip()

        vocab = 'vocab.txt' if os.path.exists(config.model_path + 'vocab.txt') else 'spiece.model'

        self.tokenizer = TOKENIZERS[config.model].from_pretrained(config.model_path + vocab)

        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    # def feture_transform(self, feature):
    #     '''
    #     使用中心翻转，连续随机擦除以及不连续随机擦除增强特征。
    #     '''
    #     feature = self.center_flip(feature)
    #     prob = random.uniform(0, 1)
    #     if prob > 0.5:
    #         feature = self.continuous_erasing(feature)
    #     else:
    #         feature = self.discontinuous_erasing(feature)
    #     return feature

    def __iter__(self):
        idxs = list(range(len(self.data)))
        if self.shuffle:
            np.random.shuffle(idxs)
        # input_ids, input_masks, segment_ids, labels = [], [], [], []

        input_ids, input_masks, segment_ids = [], [], []
        targets = {'img_name':[], 'label':[], 'feature':[]}

        for index, i in enumerate(idxs):
            json_data = self.data[i]

            # # 更换等价属性
            # prob = random.random()
            # if prob>0.5:
            #     json_data = replace_same_type_attrval(json_data)

            text = json_data['title']
            # 打乱title
            prob = random.random()
            if prob>0.5:
                text = change_title(text)

            # tkRes = self.tokenizer(text, max_length=self.max_length, truncation='longest_first',
            #                        return_attention_mask=False)
            # input_id = tkRes['input_ids']
            # segment_id = tkRes['token_type_ids']
            # assert len(segment_id) == len(input_id)
            # input_ids.append(input_id)
            # segment_ids.append(segment_id)

            tkRes = fastTokenizer(text, self.max_length, self.tokenizer)
            input_ids.append(tkRes['input_ids'])
            segment_ids.append(tkRes['token_type_ids'])
           
            targets['img_name'].append(json_data['img_name'])
            targets['label'].append(json_data['match'])

            with open(self.imgs_dir+json_data['img_name']+'.txt', 'r', encoding='utf-8') as f:
                feature = json.loads(f.readlines()[0])
            
            # if self.mode=='train':
            #     feature = self.feture_transform(torch.tensor(feature)).tolist()

            targets['feature'].append(feature)

            if len(input_ids) == self.batch_size or i == idxs[-1]:
                input_ids = paddingList(input_ids, 0, returnTensor=True)  # 动态padding
                segment_ids = paddingList(segment_ids, 0, returnTensor=True)
                input_masks = (input_ids != 0)
                yield input_ids, input_masks, segment_ids, targets
                input_ids, input_masks, segment_ids, targets = [], [], [], {'img_name':[], 'label':[], 'feature':[]}

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=0.3, alpha=0.1, emb_name='word_embeddings', is_first_attack=False):
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

    def attack(self, epsilon=0.25, emb_name='word_embeddings'):
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


if __name__=='__main__':
    json_data = {"img_name": "train100012", "title": "中长款长袖标准型酒红色仿皮皮衣加厚娃娃领暗扣", "key_attr": {"版型": "标准型", "衣长": "中长款", "袖长": "长袖", "领型": "娃娃领"}, "match": {"图文": 1, "版型": 1, "衣长": 1, "袖长": 1, "领型": 1}}

    replace_same_type_attrval(json_data)
    print(json_data)