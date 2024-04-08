import os
import json
from tqdm import tqdm
import random
import copy
import jieba
from generate_pseudo_samples import load_attr_to_attrvals

abs_path = os.path.abspath(os.path.dirname(__file__))
jieba.load_userdict(os.path.join(abs_path,'../../data/hand_data/user_dict.txt'))  #jieba分词设置自定义用户字典

def get_title_attr_to_attrvals_dict(title, attr_to_attrvals_dict, hidden_attr_to_attrvals_dict):
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

    for attr, attrval in title_attr_to_attrvals_dict.items():
        title = title.replace(attrval, '')
    
    word_list = list(jieba.cut(title))
    new_word_list = []
    for word in word_list:
        if '男' in word:
            new_word_list.extend(['男',word.replace('男','')])
        elif '女' in word:
            new_word_list.extend(['女',word.replace('女','')])
        else:
            new_word_list.append(word)
    word_list = new_word_list

    title_hidden_attr_to_attrvals_dict = {}
    for word in word_list:
        for hidden_attr, hidden_attrvals_dict in hidden_attr_to_attrvals_dict.items():
            if word in hidden_attrvals_dict.keys():
                title_hidden_attr_to_attrvals_dict[hidden_attr] = word
                break_prob = random.random()
                if break_prob > 0.5:
                    break
    return title_attr_to_attrvals_dict, title_hidden_attr_to_attrvals_dict

def generate_negative_by_change_attrvals():
    data_path_1 = os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json')
    data_path_2 = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    attr_to_attrvals_dict = load_attr_to_attrvals(os.path.join(abs_path,'../../data/contest_data/attr_to_attrvals.json'))
    hidden_attr_to_attrvals_dict = load_attr_to_attrvals(os.path.join(abs_path,'../../data/hand_data/hidden_attr_to_attrvals.json'))
    # print(attr_to_attrvals_dict)
    # print(hidden_attr_to_attrvals_dict)
    print('*'*50)
    json_data_list = []
    for line in open(data_path_1, 'r', encoding='utf-8'):
        json_data_list.append(json.loads(line))
    for line in open(data_path_2, 'r', encoding='utf-8'):
        json_data_list.append(json.loads(line))
    
    negative_pseudo_samples = []
    for i in tqdm(range(len(json_data_list)),desc='generate process'):
        json_data = json_data_list[i]
        if json_data['match']['图文'] == 1:  # 只对正样本操作
            title_attr_to_attrvals_dict, title_hidden_attr_to_attrvals_dict = get_title_attr_to_attrvals_dict(json_data['title'], attr_to_attrvals_dict, hidden_attr_to_attrvals_dict)
            
            change_key_attr_prob = random.random()
            # print(json_data['title'])
            if change_key_attr_prob > 0.55:
                # 根据modify_num随机筛选关键属性
                title_attr_num = len(list(title_attr_to_attrvals_dict.keys()))  # 得到标题的关键属性数量
                modify_num = random.randint(1,max(1,title_attr_num))
                title_attr_list = []
                if title_attr_num >= 1:
                    random_attr_index_list = random.sample(range(0, title_attr_num), modify_num)
                    title_attr_list = [list(title_attr_to_attrvals_dict.keys())[i] for i in random_attr_index_list]
                    # 得到筛选过后的title_attr_to_attrvals_dict
                    title_attr_to_attrvals_dict = dict([(attr, title_attr_to_attrvals_dict[attr]) for attr in title_attr_list])
                    # print('key_attr_modify_num : ',modify_num)
                    # print(title_attr_list)
                
                    for attr, attrval in title_attr_to_attrvals_dict.items():
                        attrval_id = attr_to_attrvals_dict[attr][attrval]  # 获取此关键属性的id
                        same_type_attrvals_list = [] # 同类型互斥
                        for attr_ in attr_to_attrvals_dict.keys():
                            if attr_==attr:
                                same_type_attrvals_list.extend([attrval_ for attrval_, id in attr_to_attrvals_dict[attr_].items() if id != attrval_id])
                        
                        if len(same_type_attrvals_list) >= 1:  # 当存在互斥属性值，才进行替换
                            while 1:
                                random_num = random.randint(0,len(same_type_attrvals_list) - 1)  # 从等价关键属性列表中，随机取出与原attrval不同的等价关键属性
                                same_type_attrval = same_type_attrvals_list[random_num]
                                if same_type_attrval != attrval:  # 互斥属性与原属性不同时结束循环
                                    break
                            
                            json_data['title'] = json_data['title'].replace(attrval, same_type_attrval)  # 修改title
                            # 如果是fine数据集，则修改key_attr字段为互斥属性，并且修改match字段中对应的关键属性匹配值
                            if attr in json_data['key_attr'].keys():
                                json_data['key_attr'][attr] = same_type_attrval
                                json_data['match'][attr] = 0
                            json_data['match']['图文'] = 0  # 只要替换过一次，就变成负样本
            
            # 根据modify_num随机筛选隐藏属性
            title_hidden_attr_num = len(list(title_hidden_attr_to_attrvals_dict.keys()))  # 得到标题的隐藏属性数量
            modify_num = random.randint(1,max(1, title_hidden_attr_num))
            
            title_hidden_attr_list = []
            if title_hidden_attr_num >= 1:
                random_hidden_attr_index_list = random.sample(range(0, title_hidden_attr_num), modify_num)
                title_hidden_attr_list = [list(title_hidden_attr_to_attrvals_dict.keys())[i] for i in random_hidden_attr_index_list]
                title_hidden_attr_to_attrvals_dict = dict([(hidden_attr, title_hidden_attr_to_attrvals_dict[hidden_attr]) for hidden_attr in title_hidden_attr_list])
                # print('hidden_attr_modify_num : ',modify_num)
                # print(title_hidden_attr_list)
                for hidden_attr, hidden_attrval in title_hidden_attr_to_attrvals_dict.items():
                    hidden_attrval_id = hidden_attr_to_attrvals_dict[hidden_attr][hidden_attrval]  # 获取此关键属性的id
                    same_type_attrvals_list = []
                    for attr_ in hidden_attr_to_attrvals_dict.keys():
                        if attr_==hidden_attr:
                            same_type_attrvals_list.extend([attrval_ for attrval_, id in hidden_attr_to_attrvals_dict[attr_].items() if id != hidden_attrval_id])
                    if len(same_type_attrvals_list) >= 1:  # 当存在互斥属性值，才进行替换
                        while 1:
                            random_num = random.randint(0,len(same_type_attrvals_list) - 1)  # 从等价关键属性列表中，随机取出与原attrval不同的等价关键属性
                            same_type_attrval = same_type_attrvals_list[random_num]
                            if same_type_attrval != hidden_attrval:  # 互斥属性与原属性不同时结束循环
                                break
                            
                    if hidden_attrval in ['单排扣', '双排扣', '牛角扣', '暗扣']:
                        prob = random.random()
                        if prob<0.2:
                            same_type_attrval = '无扣'

                    if hidden_attrval in ['男', '女']:
                        if '男装' in json_data['title'] or '女装' in json_data['title']:
                            prob = random.random()
                            if prob<0.35:
                                hidden_attrval = hidden_attrval+'装'
                                same_type_attrval = '童装'

                    json_data['title'] = json_data['title'].replace(hidden_attrval, same_type_attrval)
                    json_data['match']['图文'] = 0

            # print(json_data['match'])
            # print(json_data['title'])
            # print('*'*50)
            if json_data['match']['图文'] == 0:
                negative_pseudo_samples.append(json_data)
            
    if negative_pseudo_samples != []:
        f = open(os.path.join(abs_path,'../../data/tmp_data/'+'change_hidden_attrvals.json'), 'w', encoding='utf-8')
        for negative_pseudo_samples in negative_pseudo_samples:
            json_obj = json.dumps(negative_pseudo_samples, ensure_ascii=False)
            f.write(json_obj)
            f.write('\n')
        f.close()

if __name__ == '__main__':
    generate_negative_by_change_attrvals()

# import os
# import json
# from tqdm import tqdm
# import random
# import copy
# import jieba
# from generate_pseudo_samples import load_attr_to_attrvals

# abs_path = os.path.abspath(os.path.dirname(__file__))
# jieba.load_userdict(os.path.join(abs_path,'../../data/hand_data/user_dict.txt'))  #jieba分词设置自定义用户字典

# def get_title_attr_to_attrvals_dict(title, attr_to_attrvals_dict, hidden_attr_to_attrvals_dict):
#     word_list = list(jieba.cut(title))
#     title_attr_to_attrvals_dict = {}
#     for word in word_list:
#         if word == '系带' or word == '拉链':  # 同时存在于“裤门襟”和“闭合方式”中
#             if '鞋' in title:
#                 title_attr_to_attrvals_dict['闭合方式'] = word
#             if '靴' in title:
#                 title_attr_to_attrvals_dict['闭合方式'] = word
#             if '裤' in title:
#                 title_attr_to_attrvals_dict['裤门襟'] = word
#         else:
#             for attr, attrvals_dict in attr_to_attrvals_dict.items():
#                 if word in attrvals_dict.keys():
#                     title_attr_to_attrvals_dict[attr] = word
#                     break

#     for attr, attrval in title_attr_to_attrvals_dict.items():
#         title = title.replace(attrval, '')
    
#     word_list = list(jieba.cut(title))
#     new_word_list = []
#     for word in word_list:
#         if '男' in word:
#             new_word_list.extend(['男',word.replace('男','')])
#         elif '女' in word:
#             new_word_list.extend(['女',word.replace('女','')])
#         else:
#             new_word_list.append(word)
#     word_list = new_word_list

#     title_hidden_attr_to_attrvals_dict = {}
#     for word in word_list:
#         for hidden_attr, hidden_attrvals_dict in hidden_attr_to_attrvals_dict.items():
#             if word in hidden_attrvals_dict.keys():
#                 title_hidden_attr_to_attrvals_dict[hidden_attr] = word
#                 break_prob = random.random()
#                 if break_prob > 0.5:
#                     break
#     return title_attr_to_attrvals_dict, title_hidden_attr_to_attrvals_dict

# def generate_negative_by_change_attrvals():
#     data_path_1 = os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json')
#     data_path_2 = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
#     attr_to_attrvals_dict = load_attr_to_attrvals(os.path.join(abs_path,'../../data/contest_data/attr_to_attrvals.json'))
#     hidden_attr_to_attrvals_dict = load_attr_to_attrvals(os.path.join(abs_path,'../../data/hand_data/hidden_attr_to_attrvals.json'))
#     # print(attr_to_attrvals_dict)
#     # print(hidden_attr_to_attrvals_dict)
#     print('*'*50)
#     json_data_list = []
#     for line in open(data_path_1, 'r', encoding='utf-8'):
#         json_data_list.append(json.loads(line))
#     for line in open(data_path_2, 'r', encoding='utf-8'):
#         json_data_list.append(json.loads(line))
    
#     negative_pseudo_samples = []
#     for i in tqdm(range(len(json_data_list)),desc='generate process'):
#         json_data = json_data_list[i]
#         if json_data['match']['图文'] == 1:  # 只对正样本操作
#             title_attr_to_attrvals_dict, title_hidden_attr_to_attrvals_dict = get_title_attr_to_attrvals_dict(json_data['title'], attr_to_attrvals_dict, hidden_attr_to_attrvals_dict)
            
#             if title_hidden_attr_to_attrvals_dict=={}:
#                 prob_thr = 0.
#             else:
#                 prob_thr = 0.5
#             change_key_attr_prob = random.random()
#             # print(json_data['title'])
#             if change_key_attr_prob > prob_thr:
#                 # 根据modify_num随机筛选关键属性
#                 title_attr_num = len(list(title_attr_to_attrvals_dict.keys()))  # 得到标题的关键属性数量
#                 modify_num = random.randint(1,max(1,title_attr_num))
#                 title_attr_list = []
#                 if title_attr_num >= 1:
#                     random_attr_index_list = random.sample(range(0, title_attr_num), modify_num)
#                     title_attr_list = [list(title_attr_to_attrvals_dict.keys())[i] for i in random_attr_index_list]
#                     # 得到筛选过后的title_attr_to_attrvals_dict
#                     title_attr_to_attrvals_dict = dict([(attr, title_attr_to_attrvals_dict[attr]) for attr in title_attr_list])
#                     # print('key_attr_modify_num : ',modify_num)
#                     # print(title_attr_list)
                
#                     for attr, attrval in title_attr_to_attrvals_dict.items():
#                         attrval_id = attr_to_attrvals_dict[attr][attrval]  # 获取此关键属性的id
#                         same_type_attrvals_list = [] # 同类型互斥
#                         for attr_ in attr_to_attrvals_dict.keys():
#                             if attr_==attr:
#                                 same_type_attrvals_list.extend([attrval_ for attrval_, id in attr_to_attrvals_dict[attr_].items() if id != attrval_id])
                        
#                         if len(same_type_attrvals_list) >= 1:  # 当存在互斥属性值，才进行替换
#                             while 1:
#                                 random_num = random.randint(0,len(same_type_attrvals_list) - 1)  # 从等价关键属性列表中，随机取出与原attrval不同的等价关键属性
#                                 same_type_attrval = same_type_attrvals_list[random_num]
#                                 if same_type_attrval != attrval:  # 互斥属性与原属性不同时结束循环
#                                     break
                            
#                             json_data['title'] = json_data['title'].replace(attrval, same_type_attrval)  # 修改title
#                             # 如果是fine数据集，则修改key_attr字段为互斥属性，并且修改match字段中对应的关键属性匹配值
#                             if attr in json_data['key_attr'].keys():
#                                 json_data['key_attr'][attr] = same_type_attrval
#                                 json_data['match'][attr] = 0
#                             json_data['match']['图文'] = 0  # 只要替换过一次，就变成负样本
            
#             # 根据modify_num随机筛选隐藏属性
#             title_hidden_attr_num = len(list(title_hidden_attr_to_attrvals_dict.keys()))  # 得到标题的隐藏属性数量
#             modify_num = random.randint(1,max(1,title_hidden_attr_num))
#             title_hidden_attr_list = []
#             if title_hidden_attr_num >= 1:
#                 random_hidden_attr_index_list = random.sample(range(0, title_hidden_attr_num), modify_num)
#                 title_hidden_attr_list = [list(title_hidden_attr_to_attrvals_dict.keys())[i] for i in random_hidden_attr_index_list]
#                 title_hidden_attr_to_attrvals_dict = dict([(hidden_attr, title_hidden_attr_to_attrvals_dict[hidden_attr]) for hidden_attr in title_hidden_attr_list])
#                 # print('hidden_attr_modify_num : ',modify_num)
#                 # print(title_hidden_attr_list)
#                 for hidden_attr, hidden_attrval in title_hidden_attr_to_attrvals_dict.items():
#                     hidden_attrval_id = hidden_attr_to_attrvals_dict[hidden_attr][hidden_attrval]  # 获取此关键属性的id
#                     same_type_attrvals_list = []
#                     for attr_ in hidden_attr_to_attrvals_dict.keys():
#                         if attr_==hidden_attr:
#                             same_type_attrvals_list.extend([attrval_ for attrval_, id in hidden_attr_to_attrvals_dict[attr_].items() if id != hidden_attrval_id])
#                     if len(same_type_attrvals_list) >= 1:  # 当存在互斥属性值，才进行替换
#                         while 1:
#                             random_num = random.randint(0,len(same_type_attrvals_list) - 1)  # 从等价关键属性列表中，随机取出与原attrval不同的等价关键属性
#                             same_type_attrval = same_type_attrvals_list[random_num]
#                             if same_type_attrval != hidden_attrval:  # 互斥属性与原属性不同时结束循环
#                                 break
                    
#                     if hidden_attrval in ['单排扣','双排扣','牛角扣','暗扣']:
#                         prob = random.random()
#                         if prob<0.2:
#                             same_type_attrval = '无扣'
                    
#                     if hidden_attrval in ['男', '女']:
#                         if '男装' in json_data['title'] or '女装' in json_data['title']:
#                             prob = random.random()
#                             if prob<0.35:
#                                 hidden_attrval = hidden_attrval+'装'
#                                 same_type_attrval = '童装'

#                     json_data['title'] = json_data['title'].replace(hidden_attrval, same_type_attrval)         
#                     json_data['match']['图文'] = 0
#             # print(json_data['match'])
#             # print(json_data['title'])
#             # print('*'*50)

#             if json_data['match']['图文'] == 0:
#                 negative_pseudo_samples.append(json_data)
            
    

#     if negative_pseudo_samples != []:
#         f = open(os.path.join(abs_path,'../../data/tmp_data/'+'train_negative1.json'), 'w', encoding='utf-8')
#         for negative_pseudo_samples in negative_pseudo_samples:
#             json_obj = json.dumps(negative_pseudo_samples, ensure_ascii=False)
#             f.write(json_obj)
#             f.write('\n')
#         f.close()

# if __name__ == '__main__':
#     generate_negative_by_change_attrvals()