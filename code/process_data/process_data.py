# coding:utf-8
import jieba
import json
import random
from sklearn.utils import shuffle
from tqdm import tqdm
import os
import copy
abs_path = os.path.abspath(os.path.dirname(__file__))

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

label_list = get_key_attrs(os.path.join(abs_path,'../../data/contest_data/attr_to_attrvals.json')) + ['图文']

label2id = {l: i for i, l in enumerate(label_list)}
'''
{'裤门襟': 0, '类别': 1, '领型': 2, '裤长': 3, '裤型': 4, 
 '裙长': 5, '穿着方式': 6, '闭合方式': 7, '衣长': 8, 
 '袖长': 9, '版型': 10, '鞋帮高度': 11, '图文': 12}
'''

label_list_save_path = os.path.join(abs_path,'../../data/tmp_data/label_list.txt')
with open(label_list_save_path, 'w', encoding='utf-8') as f:
    f.write(' '.join(label_list))

############将txt数据转换成json文件#####################
def process_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = []
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            data = json.loads(line)
            """
            'img_name', 'title', 'key_attr', 'match', 'feature'
            """
            img_name = data['img_name']
            title = data['title']
            key_attr = data['key_attr']
            match = data['match']
            if isinstance(match, dict):
                label = [-1] * len(label2id)
                for key in match.keys():
                    label[label2id[key]] = match[key]
                label[label2id['图文']] = match['图文']
            else:
                label = match
            sample = {}
            sample['img_name'] = img_name
            sample['title'] = title
            sample['key_attr'] = key_attr
            sample['match'] = label            
            texts.append(sample)
    return texts

####################根据行号将数据划分成训练集和测试集###############
def split_data(texts, split_id, out_train_path, out_test_path):
    with open(out_train_path, 'w', encoding='utf-8') as f:
        for line_id, text in enumerate(texts):
            if line_id < split_id:
                text = json.dumps(text, ensure_ascii=False)
                f.write("%s\n" % text)

    with open(out_test_path, 'w', encoding='utf-8') as f:
        for line_id, text in enumerate(texts):
            if line_id >= split_id:
                text = json.dumps(text, ensure_ascii=False)
                f.write("%s\n" % text)

def cal_pn(json_path):
    with open(json_path,'r') as f:
        lines = f.readlines()
    attr_dict = {}
    cnt=0
    for line in lines:
        json_data  = json.loads(line)
        for idx,label in enumerate(json_data['match']):
            if label == -1:
                continue
            if label_list[idx] not in attr_dict.keys():
                attr_dict[label_list[idx]]=[0,0]
            attr_dict[label_list[idx]][label]+=1

    for key,val in attr_dict.items():
        print(key,'0=',str(round(100.0*val[0]/(val[0]+val[1]),2))+'%',',1=',str(round(100.0*val[1]/(val[0]+val[1]),2))+'%', val)

def shuffle_title(json_data_list):
    jieba.load_userdict(os.path.join(abs_path,'../../data/hand_data/user_dict.txt'))  #jieba分词设置自定义用户字典
    texts = []
    for i in tqdm(range(len(json_data_list)),desc='shuffle title process'):
        json_data = json_data_list[i]
        word_list = list(jieba.cut(json_data['title']))
        if len(word_list)==1:
            continue
        random_word_index_list = random.sample(range(0, len(word_list)), len(word_list))
        new_word_list = [ word_list[random_word_index] for random_word_index in random_word_index_list]
        json_data['title'] = ''.join(new_word_list)
        texts.append(json_data)
    return texts

from generate_pseudo_samples import load_attr_to_attrvals

def convert_coarse_to_fine(coarse_path):
    jieba.load_userdict(os.path.join(abs_path,'../../data/hand_data/user_dict.txt'))  #jieba分词设置自定义用户字典
    attr_to_attrvals_dict = load_attr_to_attrvals(os.path.join(abs_path,'../../data/contest_data/attr_to_attrvals.json'))
    with open(coarse_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = shuffle(lines) # shuffle
    new_data = []
    for line_id, line in tqdm(enumerate(lines),desc='convert process'):
        json_data = json.loads(line)
        if line_id<500000:
            title = json_data['title']
            word_list = list(jieba.cut(title))
            key_attr = {}
            for word in word_list:
                if word == '系带' or word == '拉链':  # 同时存在于“裤门襟”和“闭合方式”中
                    if '鞋' in title:
                        key_attr['闭合方式'] = word
                    if '靴' in title:
                        key_attr['闭合方式'] = word
                    if '裤' in title:
                        key_attr['裤门襟'] = word
                else:
                    for attr, attrvals_dict in attr_to_attrvals_dict.items():
                        if word in attrvals_dict.keys():
                            key_attr[attr] = word
                            break
            
            if json_data['match']['图文'] == 1:
                json_data['key_attr'] = key_attr
                for attr in key_attr.keys():
                    json_data['match'][attr] = 1
              
        new_data.append(json_data)
    with open(os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json'),'w',encoding='utf-8') as f:
        for data in new_data:
            text = json.dumps(data, ensure_ascii=False)
            f.write("%s\n" % text)

from generate_pseudo_samples import generate_data
from generate_negative_by_change_attrvals import generate_negative_by_change_attrvals
from replace_title import generate_negative_by_replace_title

from format_title import format_title

def generate_train_test_data_for_bert_lstm_mult_count1():
    '''
    生成训练和验证数据。
    '''
    coarse_data_path = os.path.join(abs_path,'../../data/tmp_data/train_coarse.json')
    fine_data_path = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    format_title(coarse_data_path)
    format_title(fine_data_path)

    data_path_1 = os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json')
    data_path_2 = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    data_path_3 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')
    data_path_4 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode2_negative.txt')
    #data_path_5 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode3_positive.txt')
    data_path_6 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_negative.txt')
    # data_path_7 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode4_negative.txt')
    
    ##删除先前已存在的数据#########
    if os.path.exists(data_path_1):
        os.remove(data_path_1)
    if os.path.exists(data_path_3):
        os.remove(data_path_3)
    if os.path.exists(data_path_4):
        os.remove(data_path_4)
    # if os.path.exists(data_path_5):
    #     os.remove(data_path_5)
    if os.path.exists(data_path_6):
        os.remove(data_path_6)
    # if os.path.exists(data_path_7):
    #     os.remove(data_path_7)
    # if os.path.exists(data_path_8):
    #     os.remove(data_path_8)

    convert_coarse_to_fine(os.path.join(abs_path,'../../data/tmp_data/train_coarse.json'))
    generate_params = [
                        [1,'fine'],[1,'coarse'],
                        [2,'fine'],[2,'coarse'],
                
                    ]
    generate_data(generate_params) # 生成伪样本

    generate_params = [
                        [2,'fine']
                    ]
    generate_data(generate_params, dataset_path=os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')) # 生成伪样本
    
    # 处理标签为定长格式
    data_1 = process_data(data_path_1)
    data_2 = process_data(data_path_2)
    data_3 = process_data(data_path_3)
    data_4 = process_data(data_path_4)
    # data_5 = process_data(data_path_5)
    data_6 = process_data(data_path_6)
    # data_7 = process_data(data_path_7)
    # data_8 = process_data(data_path_8)

    texts =  data_1 + data_2 + data_3 + data_4  + data_6 #+ data_7 # + data_8

    seen = set()
    delete_duplicate_texts = []
    for text in texts:
        tmp = text['title']+text['img_name']
        if tmp not in seen:
            seen.add(tmp)
            delete_duplicate_texts.append(text)
    print('before duplication:',len(texts), ', after duplication:',len(delete_duplicate_texts))

    texts = shuffle(delete_duplicate_texts)        
    out_coarse_train_path = os.path.join(abs_path,'../../data/tmp_data/train_fine_sample.json')
    out_coarse_test_path = os.path.join(abs_path,'../../data/tmp_data/test_fine_sample.json')
    split_data(texts, int(len(texts)*0.98), out_coarse_train_path, out_coarse_test_path)
    print('*' * 50)

    print('train_data rate:')
    cal_pn(out_coarse_train_path)
    print('test_data rate:')
    cal_pn(out_coarse_test_path)



def generate_train_test_data_for_bert_base_count1():
    '''
    生成训练和验证数据。
    '''
    coarse_data_path = os.path.join(abs_path,'../../data/tmp_data/train_coarse.json')
    fine_data_path = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    format_title(coarse_data_path)
    format_title(fine_data_path)

    data_path_1 = os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json')
    data_path_2 = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    data_path_3 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')
    data_path_4 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode2_negative.txt')
    #data_path_5 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode3_positive.txt')
    data_path_6 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_negative.txt')
    data_path_7 = os.path.join(abs_path,'../../data/tmp_data/replace_title.json')
    
    ##删除先前已存在的数据#########
    if os.path.exists(data_path_1):
        os.remove(data_path_1)
    if os.path.exists(data_path_3):
        os.remove(data_path_3)
    if os.path.exists(data_path_4):
        os.remove(data_path_4)
    # if os.path.exists(data_path_5):
    #     os.remove(data_path_5)
    if os.path.exists(data_path_6):
        os.remove(data_path_6)
    if os.path.exists(data_path_7):
        os.remove(data_path_7)

    convert_coarse_to_fine(os.path.join(abs_path,'../../data/tmp_data/train_coarse.json'))
    generate_params = [
                        [1,'fine'],[1,'coarse'],
                        [2,'fine'],[2,'coarse'],
                    ]
    generate_data(generate_params) # 生成伪样本

    generate_params = [
                        [2,'fine']
                    ]
    generate_data(generate_params, dataset_path=os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')) # 生成伪样本
    generate_negative_by_replace_title()
    # 处理标签为定长格式
    data_1 = process_data(data_path_1)
    data_2 = process_data(data_path_2)
    data_3 = process_data(data_path_3)
    data_4 = process_data(data_path_4)
    # data_5 = process_data(data_path_5)
    data_6 = process_data(data_path_6)
    data_7 = process_data(data_path_7)

    texts =  data_1 + data_2 + data_3 + data_4  + data_6 + data_7

    seen = set()
    delete_duplicate_texts = []
    for text in texts:
        tmp = text['title']+text['img_name']
        if tmp not in seen:
            seen.add(tmp)
            delete_duplicate_texts.append(text)
    print('before duplication:',len(texts), ', after duplication:',len(delete_duplicate_texts))

    texts = shuffle(delete_duplicate_texts)        
    out_coarse_train_path = os.path.join(abs_path,'../../data/tmp_data/train_fine_sample.json')
    out_coarse_test_path = os.path.join(abs_path,'../../data/tmp_data/test_fine_sample.json')
    split_data(texts, int(len(texts)*0.98), out_coarse_train_path, out_coarse_test_path)
    print('*' * 50)

    print('train_data rate:')
    cal_pn(out_coarse_train_path)
    print('test_data rate:')
    cal_pn(out_coarse_test_path)

def generate_train_test_data_for_bert_base_count2():
    '''
    生成训练和验证数据。
    '''
    coarse_data_path = os.path.join(abs_path,'../../data/tmp_data/train_coarse.json')
    fine_data_path = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    format_title(coarse_data_path)
    format_title(fine_data_path)

    data_path_1 = os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json')
    data_path_2 = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    data_path_3 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')
    data_path_4 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode2_negative.txt')
    #data_path_5 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode3_positive.txt')
    data_path_6 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_negative.txt')
    data_path_7 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode4_negative.txt')
    
    ##删除先前已存在的数据#########
    if os.path.exists(data_path_1):
        os.remove(data_path_1)
    if os.path.exists(data_path_3):
        os.remove(data_path_3)
    if os.path.exists(data_path_4):
        os.remove(data_path_4)
    # if os.path.exists(data_path_5):
    #     os.remove(data_path_5)
    if os.path.exists(data_path_6):
        os.remove(data_path_6)
    if os.path.exists(data_path_7):
        os.remove(data_path_7)
    # if os.path.exists(data_path_8):
    #     os.remove(data_path_8)

    convert_coarse_to_fine(os.path.join(abs_path,'../../data/tmp_data/train_coarse.json'))
    generate_params = [
                        [1,'fine'],[1,'coarse'],
                        [2,'fine'],[2,'coarse'],
                        [4,'fine'],[4,'coarse'],
                    ]
    generate_data(generate_params) # 生成伪样本

    generate_params = [
                        [2,'fine']
                    ]
    generate_data(generate_params, dataset_path=os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')) # 生成伪样本
    
    # 处理标签为定长格式
    data_1 = process_data(data_path_1)
    data_2 = process_data(data_path_2)
    data_3 = process_data(data_path_3)
    data_4 = process_data(data_path_4)
    # data_5 = process_data(data_path_5)
    data_6 = process_data(data_path_6)
    data_7 = process_data(data_path_7)
    # data_8 = process_data(data_path_8)

    texts =  data_1 + data_2 + data_3 + data_4  + data_6 + data_7 # + data_8

    seen = set()
    delete_duplicate_texts = []
    for text in texts:
        tmp = text['title']+text['img_name']
        if tmp not in seen:
            seen.add(tmp)
            delete_duplicate_texts.append(text)
    print('before duplication:',len(texts), ', after duplication:',len(delete_duplicate_texts))

    texts = shuffle(delete_duplicate_texts)        
    out_coarse_train_path = os.path.join(abs_path,'../../data/tmp_data/train_fine_sample.json')
    out_coarse_test_path = os.path.join(abs_path,'../../data/tmp_data/test_fine_sample.json')
    split_data(texts, int(len(texts)*0.98), out_coarse_train_path, out_coarse_test_path)
    print('*' * 50)

    print('train_data rate:')
    cal_pn(out_coarse_train_path)
    print('test_data rate:')
    cal_pn(out_coarse_test_path)

def generate_train_test_data():
    '''
    生成训练和验证数据。
    '''
    coarse_data_path = os.path.join(abs_path,'../../data/tmp_data/train_coarse.json')
    fine_data_path = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    format_title(coarse_data_path)
    format_title(fine_data_path)

    data_path_1 = os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json')
    data_path_2 = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    data_path_3 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')
    data_path_4 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode2_negative.txt')
    # data_path_5 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode3_positive.txt')
    data_path_6 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_negative.txt')
    data_path_7 = os.path.join(abs_path,'../../data/tmp_data/change_hidden_attrvals.json')
    # data_path_8 = os.path.join(abs_path,'../../data/tmp_data/train_negative2.json')
    
    ##删除先前已存在的数据#########
    if os.path.exists(data_path_1):
        os.remove(data_path_1)
    if os.path.exists(data_path_3):
        os.remove(data_path_3)
    if os.path.exists(data_path_4):
        os.remove(data_path_4)
    # if os.path.exists(data_path_5):
    #     os.remove(data_path_5)
    if os.path.exists(data_path_6):
        os.remove(data_path_6)
    if os.path.exists(data_path_7):
        os.remove(data_path_7)
    # if os.path.exists(data_path_8):
    #     os.remove(data_path_8)

    convert_coarse_to_fine(os.path.join(abs_path,'../../data/tmp_data/train_coarse.json'))
    generate_params = [
                        [1,'fine'],[1,'coarse'],
                        [2,'fine'],[2,'coarse'],
                    ]
    generate_data(generate_params) # 生成伪样本

    generate_params = [
                        [2,'fine']
                    ]
    generate_data(generate_params, dataset_path=os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')) # 生成伪样本
    
    generate_negative_by_change_attrvals()

    # 处理标签为定长格式
    data_1 = process_data(data_path_1)
    data_2 = process_data(data_path_2)
    data_3 = process_data(data_path_3)
    data_4 = process_data(data_path_4)
    # data_5 = process_data(data_path_5)
    data_6 = process_data(data_path_6)
    data_7 = process_data(data_path_7)
    # data_8 = process_data(data_path_8)

    texts =  data_1 + data_2 + data_3 + data_4 + data_6 + data_7 # + data_8

    seen = set()
    delete_duplicate_texts = []
    for text in texts:
        tmp = text['title']+text['img_name']
        if tmp not in seen:
            seen.add(tmp)
            delete_duplicate_texts.append(text)
    print('before duplication:',len(texts), ', after duplication:',len(delete_duplicate_texts))

    texts = shuffle(delete_duplicate_texts)        
    out_coarse_train_path = os.path.join(abs_path,'../../data/tmp_data/train_fine_sample.json')
    out_coarse_test_path = os.path.join(abs_path,'../../data/tmp_data/test_fine_sample.json')
    split_data(texts, int(len(texts)*0.98), out_coarse_train_path, out_coarse_test_path)
    print('*' * 50)

    print('train_data rate:')
    cal_pn(out_coarse_train_path)
    print('test_data rate:')
    cal_pn(out_coarse_test_path)

if __name__=='__main__':
    #convert_coarse_to_fine(os.path.join(abs_path,'../raw_data/train/train_coarse.json'))
    generate_train_test_data_for_bert_base_count1()
    

# # coding:utf-8
# import jieba
# import json
# import random
# from sklearn.utils import shuffle
# from tqdm import tqdm
# import os
# import copy
# abs_path = os.path.abspath(os.path.dirname(__file__))

# ############给关键属性包括图文一个id，方便制作标签####################
# def get_key_attrs(json_path):
#     '''params:
#            json_path: 关键词表的路径

#     return: 关键属性列表
#     '''
#     with open(json_path, 'r', encoding='utf-8') as f:
#         attrvals = f.read()
#     attrvals = json.loads(attrvals)
#     label_list = list(attrvals.keys())
#     return label_list

# label_list = get_key_attrs(os.path.join(abs_path,'../../data/contest_data/attr_to_attrvals.json')) + ['图文']

# label2id = {l: i for i, l in enumerate(label_list)}
# '''
# {'裤门襟': 0, '类别': 1, '领型': 2, '裤长': 3, '裤型': 4, 
#  '裙长': 5, '穿着方式': 6, '闭合方式': 7, '衣长': 8, 
#  '袖长': 9, '版型': 10, '鞋帮高度': 11, '图文': 12}
# '''

# label_list_save_path = os.path.join(abs_path,'../../data/tmp_data/label_list.txt')
# with open(label_list_save_path, 'w', encoding='utf-8') as f:
#     f.write(' '.join(label_list))

# ############将txt数据转换成json文件#####################
# def process_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         texts = []
#         lines = f.readlines()
#         for line_id, line in enumerate(lines):
#             data = json.loads(line)
#             """
#             'img_name', 'title', 'key_attr', 'match', 'feature'
#             """
#             img_name = data['img_name']
#             title = data['title']
#             key_attr = data['key_attr']
#             match = data['match']
#             if isinstance(match, dict):
#                 label = [-1] * len(label2id)
#                 for key in match.keys():
#                     label[label2id[key]] = match[key]
#                 label[label2id['图文']] = match['图文']
#             else:
#                 label = match
#             sample = {}
#             sample['img_name'] = img_name
#             sample['title'] = title
#             sample['key_attr'] = key_attr
#             sample['match'] = label            
#             texts.append(sample)
#     return texts

# ####################根据行号将数据划分成训练集和测试集###############
# def split_data(texts, split_id, out_train_path, out_test_path):
#     with open(out_train_path, 'w', encoding='utf-8') as f:
#         for line_id, text in enumerate(texts):
#             if line_id < split_id:
#                 text = json.dumps(text, ensure_ascii=False)
#                 f.write("%s\n" % text)

#     with open(out_test_path, 'w', encoding='utf-8') as f:
#         for line_id, text in enumerate(texts):
#             if line_id >= split_id:
#                 text = json.dumps(text, ensure_ascii=False)
#                 f.write("%s\n" % text)

# def cal_pn(json_path):
#     with open(json_path,'r') as f:
#         lines = f.readlines()
#     attr_dict = {}
#     cnt=0
#     for line in lines:
#         json_data  = json.loads(line)
#         for idx,label in enumerate(json_data['match']):
#             if label == -1:
#                 continue
#             if label_list[idx] not in attr_dict.keys():
#                 attr_dict[label_list[idx]]=[0,0]
#             attr_dict[label_list[idx]][label]+=1

#     for key,val in attr_dict.items():
#         print(key,'0=',str(round(100.0*val[0]/(val[0]+val[1]),2))+'%',',1=',str(round(100.0*val[1]/(val[0]+val[1]),2))+'%', val)

# def shuffle_title(json_data_list):
#     jieba.load_userdict(os.path.join(abs_path,'../../data/hand_data/user_dict.txt'))  #jieba分词设置自定义用户字典
#     texts = []
#     for i in tqdm(range(len(json_data_list)),desc='shuffle title process'):
#         json_data = json_data_list[i]
#         word_list = list(jieba.cut(json_data['title']))
#         if len(word_list)==1:
#             continue
#         random_word_index_list = random.sample(range(0, len(word_list)), len(word_list))
#         new_word_list = [ word_list[random_word_index] for random_word_index in random_word_index_list]
#         json_data['title'] = ''.join(new_word_list)
#         texts.append(json_data)
#     return texts

# from generate_pseudo_samples import load_attr_to_attrvals

# def convert_coarse_to_fine(coarse_path):
#     jieba.load_userdict(os.path.join(abs_path,'../../data/hand_data/user_dict.txt'))  #jieba分词设置自定义用户字典
#     attr_to_attrvals_dict = load_attr_to_attrvals(os.path.join(abs_path,'../../data/contest_data/attr_to_attrvals.json'))
#     with open(coarse_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     lines = shuffle(lines) # shuffle
#     new_data = []
#     for line_id, line in tqdm(enumerate(lines),desc='convert process'):
#         json_data = json.loads(line)
#         if line_id<500000:
#             title = json_data['title']
#             word_list = list(jieba.cut(title))
#             key_attr = {}
#             for word in word_list:
#                 if word == '系带' or word == '拉链':  # 同时存在于“裤门襟”和“闭合方式”中
#                     if '鞋' in title:
#                         key_attr['闭合方式'] = word
#                     if '靴' in title:
#                         key_attr['闭合方式'] = word
#                     if '裤' in title:
#                         key_attr['裤门襟'] = word
#                 else:
#                     for attr, attrvals_dict in attr_to_attrvals_dict.items():
#                         if word in attrvals_dict.keys():
#                             key_attr[attr] = word
#                             break
            
#             if json_data['match']['图文'] == 1:
#                 json_data['key_attr'] = key_attr
#                 for attr in key_attr.keys():
#                     json_data['match'][attr] = 1
              
#         new_data.append(json_data)
#     with open(os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json'),'w',encoding='utf-8') as f:
#         for data in new_data:
#             text = json.dumps(data, ensure_ascii=False)
#             f.write("%s\n" % text)

# from generate_pseudo_samples import generate_data
# from generate_negative_by_change_attrvals import generate_negative_by_change_attrvals
# from replace_title import generate_negative_by_replace_title

# from format_title import format_title

# def generate_train_test_data():
#     '''
#     生成训练和验证数据。
#     '''
#     coarse_data_path = os.path.join(abs_path,'../../data/tmp_data/train_coarse.json')
#     fine_data_path = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
#     format_title(coarse_data_path)
#     format_title(fine_data_path)

#     data_path_1 = os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json')
#     data_path_2 = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
#     data_path_3 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')
#     data_path_4 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode2_negative.txt')
#     data_path_5 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode3_positive.txt')
#     data_path_6 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_negative.txt')
#     data_path_7 = os.path.join(abs_path,'../../data/tmp_data/train_negative1.json')
#     # data_path_8 = os.path.join(abs_path,'../../data/tmp_data/train_negative2.json')
    
#     ##删除先前已存在的数据#########
#     if os.path.exists(data_path_1):
#         os.remove(data_path_1)
#     if os.path.exists(data_path_3):
#         os.remove(data_path_3)
#     if os.path.exists(data_path_4):
#         os.remove(data_path_4)
#     if os.path.exists(data_path_5):
#         os.remove(data_path_5)
#     if os.path.exists(data_path_6):
#         os.remove(data_path_6)
#     if os.path.exists(data_path_7):
#         os.remove(data_path_7)
#     # if os.path.exists(data_path_8):
#     #     os.remove(data_path_8)

#     convert_coarse_to_fine(os.path.join(abs_path,'../../data/tmp_data/train_coarse.json'))
#     generate_params = [
#                         [1,'fine'],[1,'coarse'],
#                         [2,'fine'],[2,'coarse'],
#                         [3,'fine'],[3,'coarse'],
#                     ]
#     generate_data(generate_params) # 生成伪样本

#     generate_params = [
#                         [2,'fine']
#                     ]
#     generate_data(generate_params, dataset_path=os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')) # 生成伪样本
    
#     generate_negative_by_change_attrvals()

#     # 处理标签为定长格式
#     data_1 = process_data(data_path_1)
#     data_2 = process_data(data_path_2)
#     data_3 = process_data(data_path_3)
#     data_4 = process_data(data_path_4)
#     data_5 = process_data(data_path_5)
#     data_6 = process_data(data_path_6)
#     data_7 = process_data(data_path_7)
#     # data_8 = process_data(data_path_8)

#     texts =  data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7 # + data_8

#     seen = set()
#     delete_duplicate_texts = []
#     for text in texts:
#         tmp = text['title']+text['img_name']
#         if tmp not in seen:
#             seen.add(tmp)
#             delete_duplicate_texts.append(text)
#     print('before duplication:',len(texts), ', after duplication:',len(delete_duplicate_texts))

#     texts = shuffle(delete_duplicate_texts)        
#     out_coarse_train_path = os.path.join(abs_path,'../../data/tmp_data/train_fine_sample.json')
#     out_coarse_test_path = os.path.join(abs_path,'../../data/tmp_data/test_fine_sample.json')
#     split_data(texts, int(len(texts)*0.98), out_coarse_train_path, out_coarse_test_path)
#     print('*' * 50)

#     print('train_data rate:')
#     cal_pn(out_coarse_train_path)
#     print('test_data rate:')
#     cal_pn(out_coarse_test_path)

# # def generate_train_test_data():
# #     '''
# #     生成训练和验证数据。
# #     '''
# #     coarse_data_path = os.path.join(abs_path,'../../data/tmp_data/train_coarse.json')
# #     fine_data_path = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
# #     format_title(coarse_data_path)
# #     format_title(fine_data_path)

# #     data_path_1 = os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json')
# #     data_path_2 = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
# #     data_path_3 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_positive.txt')
# #     data_path_4 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode2_negative.txt')
# #     data_path_5 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode3_positive.txt')
# #     data_path_6 = os.path.join(abs_path,'../../data/tmp_data/pseudo_sample_mode1_negative.txt')
# #     data_path_7 = os.path.join(abs_path,'../../data/tmp_data/train_negative1.json')
# #     # data_path_8 = os.path.join(abs_path,'../../data/tmp_data/train_negative2.json')
    
# #     ##删除先前已存在的数据#########
# #     if os.path.exists(data_path_1):
# #         os.remove(data_path_1)
# #     if os.path.exists(data_path_3):
# #         os.remove(data_path_3)
# #     if os.path.exists(data_path_4):
# #         os.remove(data_path_4)
# #     if os.path.exists(data_path_5):
# #         os.remove(data_path_5)
# #     if os.path.exists(data_path_6):
# #         os.remove(data_path_6)
# #     if os.path.exists(data_path_7):
# #         os.remove(data_path_7)
# #     # if os.path.exists(data_path_8):
# #     #     os.remove(data_path_8)

# #     convert_coarse_to_fine(os.path.join(abs_path,'../../data/tmp_data/train_coarse.json'))
# #     generate_params = [
# #                         [1,'fine'],[1,'coarse'],
# #                         [2,'fine'],[2,'coarse'],
# #                         [3,'fine'],[3,'coarse'],
# #                         # [4,'fine'],[4,'coarse']
# #                     ]
# #     generate_data(generate_params) # 生成伪样本
# #     generate_negative_by_change_attrvals()
# #     # generate_negative_by_replace_title()

# #     # 处理标签为定长格式
# #     data_1 = process_data(data_path_1)
# #     data_2 = process_data(data_path_2)
# #     data_3 = process_data(data_path_3)
# #     data_4 = process_data(data_path_4)
# #     data_5 = process_data(data_path_5)
# #     data_6 = process_data(data_path_6)
# #     data_7 = process_data(data_path_7)
# #     # data_8 = process_data(data_path_8)

# #     texts =  data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7 # + data_8

# #     seen = set()
# #     delete_duplicate_texts = []
# #     for text in texts:
# #         tmp = text['title']+text['img_name']
# #         if tmp not in seen:
# #             seen.add(tmp)
# #             delete_duplicate_texts.append(text)
# #     print('before duplication:',len(texts), ', after duplication:',len(delete_duplicate_texts))

# #     texts = shuffle(delete_duplicate_texts)        
# #     out_coarse_train_path = os.path.join(abs_path,'../../data/tmp_data/train_fine_sample.json')
# #     out_coarse_test_path = os.path.join(abs_path,'../../data/tmp_data/test_fine_sample.json')
# #     split_data(texts, int(len(texts)*0.98), out_coarse_train_path, out_coarse_test_path)
# #     print('*' * 50)


# if __name__=='__main__':
#     #convert_coarse_to_fine(os.path.join(abs_path,'../raw_data/train/train_coarse.json'))
#     generate_train_test_data()
    