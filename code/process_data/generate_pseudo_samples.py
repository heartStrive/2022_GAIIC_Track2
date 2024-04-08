#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import os
import random
import jieba
import copy
from tqdm import tqdm
import gensim
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.linalg import norm
"""
mode说明
change: 适用于fine和coarse数据集
    √ 可执行：
        1.1正样本关键属性修改成等价属性值，还是正样本；1.2负样本关键属性修改成等价属性值，还是负样本；
        2.正样本关键属性修改成互斥属性值，变成负样本
    × 不可执行：
        负样本关键属性修改成互斥属性值，样本标签不确定
delete: 适用于fine和coarse数据集
    √ 可执行：
        3.正样本删除关键属性，还是正样本；
    × 不可执行：
        负样本删除关键属性，样本标签不确定
shuffle：适用于fine和coarse数据集
    √ 可执行：
        4.正样本的title被打乱，还是正样本；负样本的title被打乱，还是负样本
    
generate过程
    1. 执行四种mode，生成dataset_1、dataset_2、dataset_3和dataset_4
    2、将dataset_1、dataset_2和dataset_3再通过shuffle生成dataset_5、dataset_6和dataset_7
    3、final_dataset = dataset_1 + dataset_2 + dataset_3 + dataset_4 + dataset_5 + dataset_6 + dataset_7
    
"""
# 文件的绝对路径
abs_path = os.path.abspath(os.path.dirname(__file__))
contest_data_path = os.path.join(abs_path, '../../data/contest_data/')
tmp_data_path = os.path.join(abs_path, '../../data/tmp_data/')
hand_data_path = os.path.join(abs_path, '../../data/hand_data/')

def write_jieba_user_dict(attr_to_attrvals_path,jieba_user_dict_path):
    """
    生成jieba分词库的自定义用户字典，并写入文件
    :param attr_to_attrvals_path: 关键属性字典文件路径
    :param jieba_user_dict_path: jieba分词库自定义用户字典文件路径
    :return: None
    """
    attr_to_attrvals = load_attr_to_attrvals(attr_to_attrvals_path)
    hidden_attr_to_attrvals_path = attr_to_attrvals_path.replace('attr_to_attrvals.json','hidden_attr_to_attrvals.json')
    hidden_attr_to_attrvals_path = hidden_attr_to_attrvals_path.replace('contest_data','hand_data')
    hidden_attr_to_attrvals = load_attr_to_attrvals(hidden_attr_to_attrvals_path)
    jieba_user_dict = {}

    for attr,attrvals_list in attr_to_attrvals.items():
        for attrval in attrvals_list:
            jieba_user_dict[attrval] = '999999999999999999'
    
    for hidden_attr,hidden_attrvals_list in hidden_attr_to_attrvals.items():
        for hidden_attrval in hidden_attrvals_list:
            jieba_user_dict[hidden_attrval] = '999999999999999999'
    
    # f = open(os.path.join(abs_path,hand_data_path,'color.txt'),'r',encoding='utf-8')
    # colors_list = []
    # colors = f.readlines()
    # for color in colors:
    #     color_list = color.split(' ')
    #     if len(color_list)==1:
    #         continue
    #     colors_list.append(color_list[0])
    #     if '色' not in color_list[0]:
    #         colors_list.append(color_list[0]+'色')

    f = open(jieba_user_dict_path,'w',encoding='utf-8')
    for key,value in jieba_user_dict.items():
        f.write(key + ' ' + value +'\n')
    
    f.write('普厚'+' 999999999999999999\n')
    with open(os.path.join(abs_path,hand_data_path,'stop_words.txt')) as stop_words_f:
        stop_words = stop_words_f.readlines()
    for word in stop_words:
        word = word.strip()
        f.write(word+' 999999999999999999\n')

    # for color in colors_list:
    #     ok=True
    #     for c in hidden_attr_to_attrvals['颜色'].keys():
    #         if c in color:
    #             ok=False
    #             break
    #     if ok:
    #         f.write(color+' 999999999999\n')

    f.close()

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

# 生成user_dict.txt
attr_to_attrvals_path = os.path.join(abs_path, contest_data_path,'attr_to_attrvals.json')
jieba_user_dict_path = os.path.join(abs_path, hand_data_path,'user_dict.txt')
write_jieba_user_dict(attr_to_attrvals_path, jieba_user_dict_path)
jieba.load_userdict(os.path.join(abs_path, hand_data_path,'user_dict.txt'))  #jieba分词设置自定义用户字典

def get_title_attr_to_attrvals_dict(title, key_attr, attr_to_attrvals_dict):
    if key_attr == {}:
        word_list = list(jieba.cut(title))
        title_attr_to_attrvals_dict = {}
        for word in word_list:
            if word == '系带' or word == '拉链':  # 同时存在于“裤门襟”和“闭合方式”中
                if '鞋' in title:
                    title_attr_to_attrvals_dict['闭合方式'] = word
                if '靴' in title:
                    title_attr_to_attrvals_dict['闭合方式'] = word
                if '裤' in title:
                    title_attr_to_attrvals_dict['裤门襟'] = word
            else:
                for attr, attrvals_dict in attr_to_attrvals_dict.items():
                    if word in attrvals_dict.keys():
                        title_attr_to_attrvals_dict[attr] = word
                        break
    else:
        title_attr_to_attrvals_dict = key_attr
        
    return title_attr_to_attrvals_dict


def generate_pseudo_samples(dataset_path,attr_to_attrvals_path,mode):
    """
    根据执行模式，生成伪样本，并写入文件
    :param dataset_path: 数据集文件路径
    :param attr_to_attrvals_path: 关键属性字典文件路径
    :param mode: 生成伪样本方式
    :param generate_num: 生成伪样本数量，小于原样本总数
    :param modify_num: 修改关键属性的数量，如果值为-1，表示所有关键属性都修改
    :return: None
    """

    #  get dataset
    json_data_list = []
    for line in open(dataset_path, encoding='utf-8'):
        json_data_list.append(json.loads(line))
    attr_to_attrvals_dict = load_attr_to_attrvals(attr_to_attrvals_path)

    positive_pseudo_samples = []
    negative_pseudo_samples = []

    #  select json_data
    #1.正样本关键属性修改成等价属性值，还是正样本；负样本关键属性修改成等价属性值，还是负样本；
    if mode == 1 :
        for i in tqdm(range(len(json_data_list)),desc='1st process'):
            json_data = json_data_list[i]
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
                if modify_num == -1:
                    title_attr_list= list(title_attr_to_attrvals_dict.keys())
                elif modify_num <= title_attr_num:
                    random_attr_index_list = random.sample(range(0, title_attr_num), modify_num)
                    title_attr_list = [list(title_attr_to_attrvals_dict.keys())[i] for i in random_attr_index_list]
                else:
                    # print('修改个数大于总关键属性个数')
                    # print(json_data['title'])
                    continue
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
                # print(title_attr_to_attrvals_dict)
                if json_data['match']['图文'] == 1:
                    positive_pseudo_samples.append(json_data)
                else:
                    negative_pseudo_samples.append(json_data)

    # 2.正样本关键属性修改成互斥属性值，变成负样本
    elif mode == 2:
        for i in tqdm(range(len(json_data_list)),desc='2nd process'):
            json_data = json_data_list[i]
            if json_data['match']['图文'] == 1:  # 只对正样本操作
                # print('*' * 50)
                title_attr_to_attrvals_dict = get_title_attr_to_attrvals_dict(json_data['title'],json_data['key_attr'], attr_to_attrvals_dict)
            
                # 根据modify_num随机筛选关键属性
                title_attr_num = len(list(title_attr_to_attrvals_dict.keys()))  # 得到标题的关键属性数量
                modify_num = random.randint(1,max(1,title_attr_num))
                title_attr_list = []
                if title_attr_num >= 1:
                    if modify_num == -1:
                        title_attr_list = list(title_attr_to_attrvals_dict.keys())
                    elif modify_num <= title_attr_num:
                        random_attr_index_list = random.sample(range(0, title_attr_num), modify_num)
                        title_attr_list = [list(title_attr_to_attrvals_dict.keys())[i] for i in random_attr_index_list]
                    else:
                        # print('修改个数大于总关键属性个数')
                        continue
                    # 得到筛选过后的title_attr_to_attrvals_dict
                    title_attr_to_attrvals_dict = dict([(attr, title_attr_to_attrvals_dict[attr]) for attr in title_attr_list])

                    for attr, attrval in title_attr_to_attrvals_dict.items():
                        attrval_id = attr_to_attrvals_dict[attr][attrval]  # 获取此关键属性的id
                        mutex_attrvals_list = [] # 不同类型互斥
                        same_type_attrvals_list = [] # 同类型互斥
                        for attr_ in attr_to_attrvals_dict.keys():
                            if attr_==attr:
                                same_type_attrvals_list.extend([attrval_ for attrval_, id in attr_to_attrvals_dict[attr_].items() if id != attrval_id])
                            else:
                                mutex_attrvals_list.extend([attrval_ for attrval_, id in attr_to_attrvals_dict[attr_].items() if id != attrval_id])  # 找到互斥属性值列表
                        # print(same_type_attrvals_list)
                        # print(mutex_attrvals_list)
                        # print(len(same_type_attrvals_list)+len(mutex_attrvals_list))
                        prob = random.random()
                        if prob>1.0:
                            if len(mutex_attrvals_list) >= 1:  # 当存在互斥属性值，才进行替换
                                while 1:
                                    random_num = random.randint(0,len(mutex_attrvals_list) - 1)  # 从等价关键属性列表中，随机取出与原attrval不同的等价关键属性
                                    mutex_attrval = mutex_attrvals_list[random_num]
                                    if mutex_attrval != attrval:  # 互斥属性与原属性不同时结束循环
                                        break
                                title_attr_to_attrvals_dict[attr] = mutex_attrval
                                json_data['title'] = json_data['title'].replace(attrval, mutex_attrval)  # 修改title
                                # 如果是fine数据集，则修改key_attr字段为互斥属性，并且修改match字段中对应的关键属性匹配值
                                if attr in json_data['key_attr'].keys():
                                    json_data['key_attr'][attr] = mutex_attrval
                                    json_data['match'][attr] = 0
                                json_data['match']['图文'] = 0  # 只要替换过一次，就变成负样本
                        else:
                            if len(same_type_attrvals_list) >= 1:  # 当存在互斥属性值，才进行替换
                                while 1:
                                    random_num = random.randint(0,len(same_type_attrvals_list) - 1)  # 从等价关键属性列表中，随机取出与原attrval不同的等价关键属性
                                    same_type_attrval = same_type_attrvals_list[random_num]
                                    if same_type_attrval != attrval:  # 互斥属性与原属性不同时结束循环
                                        break
                                
                                title_attr_to_attrvals_dict[attr] = same_type_attrval
                                json_data['title'] = json_data['title'].replace(attrval, same_type_attrval)  # 修改title
                                # 如果是fine数据集，则修改key_attr字段为互斥属性，并且修改match字段中对应的关键属性匹配值
                                if attr in json_data['key_attr'].keys():
                                    json_data['key_attr'][attr] = same_type_attrval
                                    json_data['match'][attr] = 0
                                json_data['match']['图文'] = 0  # 只要替换过一次，就变成负样本
                        
                    # print(title_attr_to_attrvals_dict)
                    if json_data['match']['图文'] == 0:
                        negative_pseudo_samples.append(json_data)

    # 3.正样本删除关键属性，还是正样本；
    elif mode == 3:
        for i in tqdm(range(len(json_data_list)),desc='3rd process'):
            json_data = json_data_list[i]
            if json_data['match']['图文'] == 1:  # 只对正样本操作
                title_attr_to_attrvals_dict = get_title_attr_to_attrvals_dict(json_data['title'],json_data['key_attr'], attr_to_attrvals_dict)
            
                # 根据modify_num随机筛选关键属性
                title_attr_num = len(list(title_attr_to_attrvals_dict.keys()))  # 得到标题的关键属性数量
                modify_num = 1
                title_attr_list = []
                if title_attr_num >= 2:
                    if modify_num == -1:
                        title_attr_list = list(title_attr_to_attrvals_dict.keys())
                    elif modify_num <= title_attr_num - 1:
                        random_attr_index_list = random.sample(range(0, title_attr_num), modify_num)
                        title_attr_list = [list(title_attr_to_attrvals_dict.keys())[i] for i in random_attr_index_list]
                    else:
                        # print('删除个数大于总关键属性个数')
                        continue
                    # 得到筛选过后的title_attr_to_attrvals_dict
                    title_attr_to_attrvals_dict = dict([(attr, title_attr_to_attrvals_dict[attr]) for attr in title_attr_list])
                    
                    for attr, attrval in title_attr_to_attrvals_dict.items():
                        json_data['title'] = json_data['title'].replace(attrval, '')  # 删除title中的属性
                        # 如果是fine数据集，则修改删除key_attr和match中字段对应关键属性
                        if attr in json_data['key_attr'].keys():
                            json_data['key_attr'].pop(attr)
                            json_data['match'].pop(attr)

                    # print(title_attr_to_attrvals_dict)
                    if json_data['match']['图文'] == 1:
                        positive_pseudo_samples.append(json_data)
            # print('*' * 50)
    
    # 4.随机替换title
    elif mode == 4:
        used_pair_set = set()
        for i in tqdm(range(len(json_data_list)),desc='4th process'):
            json_data = json_data_list[i]
            if json_data['match']['图文'] == 1:  # 只对正样本操作
                # print('*' * 50)
                title_attr_to_attrvals_dict = get_title_attr_to_attrvals_dict(json_data['title'],json_data['key_attr'], attr_to_attrvals_dict)
                while 1:
                    random_num = random.randint(0, len(json_data_list) - 1)
                    if i != random_num and json_data['title']!=json_data_list[random_num]['title']: #and tf_similarity(json_data['title'],json_data_list[random_num]['title'])>0.3:
                        break

                # if random_num>i:
                #     pair = (i,random)
                # else:
                #     pair = (random_num,i)
                # if pair in used_pair_set:
                #     continue
                # else:
                #     used_pair_set.add(pair)

                # print('******************************************')
                # print(json_data['title'])
                # print(json_data_list[random_num]['title'])

                now_key_attr = get_title_attr_to_attrvals_dict(json_data_list[random_num]['title'],json_data_list[random_num]['key_attr'], attr_to_attrvals_dict)
                now_data = copy.deepcopy(json_data_list[random_num])
                now_data['img_name']=json_data['img_name'] # 原来数据的特征拿过来
                for now_key_attr in now_data['match'].keys(): # match先全改成0
                    now_data['match'][now_key_attr] = 0

                for now_key_attr in now_data['match'].keys():
                    if now_key_attr=='图文':
                        continue
                    # now_data 领型:高领，json_data：领型:半高领
                    if now_key_attr in json_data['match'].keys():
                        attrval_id = attr_to_attrvals_dict[now_key_attr][json_data['key_attr'][now_key_attr]]  # 获取此关键属性的id
                        equal_attrvals_list = [attrval for attrval,id in attr_to_attrvals_dict[now_key_attr].items() if id == attrval_id]  # 找到等价属性值列表
                        if now_data['key_attr'][now_key_attr] in equal_attrvals_list:
                            now_data['match'][now_key_attr] = 1
                    
                now_data['match']['图文']=0
                
                if now_data['match']['图文'] == 0:
                    negative_pseudo_samples.append(now_data)
        
    print('生成的伪样本总数量为：{}，其中正样本数量为：{}，负样本数量为：{}'.format(len(positive_pseudo_samples)+len(negative_pseudo_samples),len(positive_pseudo_samples),len(negative_pseudo_samples)))

    if positive_pseudo_samples != []:
        f = open(os.path.join(abs_path,tmp_data_path,'pseudo_sample_mode'+str(mode)+'_positive.txt'), 'a', encoding='utf-8')
        for positive_pseudo_sample in positive_pseudo_samples:
            json_obj = json.dumps(positive_pseudo_sample, ensure_ascii=False)
            f.write(json_obj)
            f.write('\n')
        f.close()

    if negative_pseudo_samples != []:
        f = open(os.path.join(abs_path,tmp_data_path,'pseudo_sample_mode' + str(mode) + '_negative.txt'), 'a', encoding='utf-8')
        for negative_pseudo_samples in negative_pseudo_samples:
            json_obj = json.dumps(negative_pseudo_samples, ensure_ascii=False)
            f.write(json_obj)
            f.write('\n')
        f.close()

def generate_data(generate_params, dataset_path=None):
    for param in generate_params:
        mode = param[0]
        dataset_type = param[1]
        if dataset_path==None:
            if dataset_type=='coarse':
                dataset_path = os.path.join(abs_path,tmp_data_path,'new_train_' + dataset_type + '.json')
            else:
                dataset_path = os.path.join(abs_path,tmp_data_path,'train_' + dataset_type + '.json')
        
        print(dataset_path)
        generate_pseudo_samples(dataset_path=dataset_path,
                            attr_to_attrvals_path=os.path.join(abs_path,contest_data_path ,'attr_to_attrvals.json'),
                            mode=mode)
        dataset_path = None
if __name__ == '__main__':
    pass
    # write_jieba_user_dict(os.path.join(contest_data_path,'attr_to_attrvals.json'),os.path.join(hand_data_path,'user_dict.txt'))
    # generate_data()
