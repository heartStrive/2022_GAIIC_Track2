import os
import json
from tqdm import tqdm
import random
import copy
from generate_pseudo_samples import load_attr_to_attrvals
abs_path = os.path.abspath(os.path.dirname(__file__))
# f1 = open('./hello.txt','w')
def generate_negative_by_replace_title():
    data_path_1 = os.path.join(abs_path,'../../data/tmp_data/new_train_coarse.json')
    data_path_2 = os.path.join(abs_path,'../../data/tmp_data/train_fine.json')
    attr_to_attrvals_dict = load_attr_to_attrvals(os.path.join(abs_path,'../../data/contest_data/attr_to_attrvals.json'))

    json_data_list = []
    for line in open(data_path_1, 'r', encoding='utf-8'):
        json_data_list.append(json.loads(line))
    for line in open(data_path_2, 'r', encoding='utf-8'):
        json_data_list.append(json.loads(line))

    negative_pseudo_samples = []
    used_pair_set = set()
    for i in tqdm(range(len(json_data_list)),desc='replace process'):
        json_data = json_data_list[i]
        
        if json_data['match']['图文'] == 1 and json_data['key_attr']!={}:  # 只对正样本操作
            # print('*' * 50)
            title_attr_to_attrvals_dict = json_data['key_attr']
            prob = random.random()

            if prob>0.6:
                s1 = set(title_attr_to_attrvals_dict.keys())
                while 1:
                    random_num = random.randint(0, len(json_data_list) - 1)
                    now_title_attr_to_attrvals_dict = json_data_list[random_num]['key_attr']
                    s2 = set(now_title_attr_to_attrvals_dict.keys())
                    res = s1&s2
                    if i != random_num and json_data['title']!=json_data_list[random_num]['title'] and json_data_list[random_num]['key_attr']!={} and len(res)==0:
                        break
                
                if random_num>i:
                    pair = (i,random)
                else:
                    pair = (random_num,i)
                if pair in used_pair_set:
                    continue
                else:
                    used_pair_set.add(pair)

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

                if json_data['title'] != now_data['title']:
                    now_data['match']['图文']=0

            else:
                s1 = set(title_attr_to_attrvals_dict.keys())
                while 1:
                    random_num = random.randint(0, len(json_data_list) - 1)
                    now_title_attr_to_attrvals_dict = json_data_list[random_num]['key_attr']
                    s2 = set(now_title_attr_to_attrvals_dict.keys())
                    res = s1&s2
                    if i != random_num and json_data['title']!=json_data_list[random_num]['title'] and len(res)>0:
                        break

                
                if random_num>i:
                    pair = (i,random)
                else:
                    pair = (random_num,i)
                if pair in used_pair_set:
                    continue
                else:
                    used_pair_set.add(pair)

                now_key_attr = json_data_list[random_num]['key_attr']
                intersections = set(title_attr_to_attrvals_dict.keys())&set(now_key_attr.keys())

                now_data = copy.deepcopy(json_data_list[random_num])
                now_data['img_name']=json_data['img_name'] # 原来数据的特征拿过来
                now_data['match'] = {}
                tmp_key_attr = now_data['key_attr']
                now_data['key_attr'] = {}

                for now_key_attr in intersections: # match先全改成0
                    now_data['match'][now_key_attr] = 0
                    now_data['key_attr'][now_key_attr] = tmp_key_attr[now_key_attr]

                for now_key_attr in now_data['match'].keys():
                    if now_key_attr=='图文':
                        continue
                    # now_data 领型:高领，json_data：领型:半高领
                    if now_key_attr in json_data['match'].keys():
                        attrval_id = attr_to_attrvals_dict[now_key_attr][json_data['key_attr'][now_key_attr]]  # 获取此关键属性的id
                        equal_attrvals_list = [attrval for attrval,id in attr_to_attrvals_dict[now_key_attr].items() if id == attrval_id]  # 找到等价属性值列表
                        if now_data['key_attr'][now_key_attr] in equal_attrvals_list:
                            now_data['match'][now_key_attr] = 1

                if json_data['title'] != now_data['title']:
                    now_data['match']['图文']=0   
            if now_data['match']['图文']==0:
                negative_pseudo_samples.append(now_data)

            # print('*'*100, file=f1)
            # print('1:', json_data, file=f1)
            # print('2:', json_data_list[random_num], file=f1)
            # print('3:', now_data, file=f1)

    if negative_pseudo_samples != []:
        f = open(os.path.join(abs_path,'../../data/tmp_data/'+'replace_title.json'), 'w', encoding='utf-8')
        for negative_pseudo_samples in negative_pseudo_samples:
            json_obj = json.dumps(negative_pseudo_samples, ensure_ascii=False)
            f.write(json_obj)
            f.write('\n')
        f.close()

if __name__ == '__main__':
    generate_negative_by_replace_title()