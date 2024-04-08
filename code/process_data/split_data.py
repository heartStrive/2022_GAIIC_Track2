import json
import os
from tqdm import tqdm

def storage_features(data_path, save_path):
    '''
    将图像特征以img_name.txt的方式存在save_path, 其余属性保存至json文件。
    '''
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line_id, line in tqdm(enumerate(lines)):
        data = json.loads(line)
        img_name = data['img_name']
        
        feature = data['feature']
        with open(save_path+img_name+'.txt', 'w', encoding='utf-8') as f:
            feature = json.dumps(feature, ensure_ascii=False)
            f.write("%s\n" % feature)
            
        data.pop('feature') # 删除feature
        texts.append(data)

    new_data_path = data_path[:-4]+'.json'
    new_data_path = new_data_path.replace('contest_data','tmp_data')
    with open(new_data_path, 'w', encoding='utf-8') as f:
        for line_id, text in enumerate(texts):
            text = json.dumps(text, ensure_ascii=False)
            f.write("%s\n" % text)

if __name__ == '__main__':
    train_coarse_path = '../../data/contest_data/train_coarse.txt'
    train_fine_path = '../../data/contest_data/train_fine.txt'
    # out_train_coarse_imgs_path = '../raw_data/coarse_imgs/'
    # out_train_fine_imgs_path = '../raw_data/fine_imgs/'
    # os.makedirs(out_train_coarse_imgs_path, exist_ok=True)
    # os.makedirs(out_train_fine_imgs_path, exist_ok=True)

    out_imgs_path = '../../data/tmp_data/imgs/'
    os.makedirs(out_imgs_path, exist_ok=True)
    storage_features(train_coarse_path, out_imgs_path)
    storage_features(train_fine_path, out_imgs_path)