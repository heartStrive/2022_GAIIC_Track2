import jieba
import os
import json
from generate_pseudo_samples import load_attr_to_attrvals
abs_path = os.path.abspath(os.path.dirname(__file__))

# jieba分词设置自定义用户字典
user_dict_path = os.path.join(abs_path, '../../data/hand_data/user_dict.txt')
jieba.load_userdict(user_dict_path)  
attr_to_attrvals_dict = load_attr_to_attrvals(os.path.join(abs_path,'../../data/contest_data/attr_to_attrvals.json'))


def get_title_key_attrs(title):
    word_list = list(jieba.cut(title))
    key_attr_list = []
    for word in word_list:
        if word == '系带' or word == '拉链':  # 同时存在于“裤门襟”和“闭合方式”中
            if '鞋' in title:
                key_attr_list.append(('闭合方式', word))
            if '靴' in title:
                key_attr_list.append(('闭合方式', word))
            if '裤' in title:
                key_attr_list.append(('裤门襟', word))
        else:
            for attr, attrvals_dict in attr_to_attrvals_dict.items():
                if word in attrvals_dict.keys():
                    key_attr_list.append((attr, word))
                    break
    return key_attr_list



def format_title(data_path):
    user_dict_path = os.path.join(abs_path, '../../data/hand_data/user_dict.txt')
    stop_words_path = os.path.join(abs_path, '../../data/hand_data/stop_words.txt')
    hidden_attr_to_attrvals_path = os.path.join(abs_path, '../../data/hand_data/hidden_attr_to_attrvals.json')
    # # jieba分词设置自定义用户字典
    # jieba.load_userdict(user_dict_path)  

    # 取出颜色属性值
    with open(hidden_attr_to_attrvals_path, 'r', encoding='utf-8') as f:
        hidden_attr_to_attrvals = json.load(f)
    color_list = hidden_attr_to_attrvals['颜色']
    color_dict = {}
    # 不需要额外处理的颜色
    color_list2 = ["驼色", "粉色", "花色", "咖色", "杏色"]
    for color in color_list:
        if color in color_list2:
            continue
        color2 = color.replace('色','')
        color_dict[color2] = color

    # 等价属性
    equal_attrs = {
        '咖啡色':'咖色','毛呢大衣':'呢大衣','儿童':'童装','毛衣':'针织衫',
        # '超短款':'短款','超长款':'长款','中长款':'中款',
        # '标准型':'修身型','超短裙':'短裙','中长裙':'中裙','半高领':'高领',
        # '微喇裤':'喇叭裤',
        # '常规厚度':'普厚','厚度常规':'普厚'
    }

    
    for attr in equal_attrs.keys():
        print(attr,equal_attrs[attr])
    # 取出停用词
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    stop_words = []
    for line in lines:
        stop_words.append(line.strip())

    json_data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        json_data = json.loads(line)
        
        # title转换成大写
        json_data['title'] = json_data['title'].upper()
        
        # print(json_data['title'])

        # 处理等价隐藏属性
        for attrval in equal_attrs.keys():
            if attrval in json_data['title']:
                # if attr=='厚度常规' and '厚度常规款' in json_data['title']:
                #     continue
                json_data['title'] = json_data['title'].replace(attrval,equal_attrs[attrval])
                
                # if 'key_attr' in json_data.keys():
                #     for key_attr,key_attrval in json_data['key_attr'].items():
                #         if key_attrval==attrval:
                #             json_data['key_attr'][key_attr] = equal_attrs[attrval]
                
        
        # 删除停用词
        for word in stop_words:
            if word in json_data['title']:
                json_data['title'] = json_data['title'].replace(word,'')
                
        # step 处理颜色
        for key,val in color_dict.items():
            # key->黑, val->黑色
            if key in json_data['title'] and val not in json_data['title']:
                json_data['title'] = json_data['title'].replace(key, val)

        json_data_list.append(json_data)
        # print(json_data['title'])
        # print('*'*50)
    if data_path.endswith('.txt'):
        data_path = data_path[:-4]+'.json'
    

    f = open(data_path, 'w', encoding='utf-8')
    for json_data in json_data_list:
        json_obj = json.dumps(json_data, ensure_ascii=False)
        f.write(json_obj)
        f.write('\n')
    f.close()

if __name__=='__main__':
    data_path = '../../data/contest_data/semi_testA.txt'
    format_title(data_path)

# 89 89 89 92.47
# 4 12 92.2
# 92.60 