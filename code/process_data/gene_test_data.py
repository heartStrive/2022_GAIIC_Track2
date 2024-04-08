import json
test_data_path = '../raw_data/test_fine_sample.json'
out_path = '../raw_data/preliminary_testA.json'
imgs_dir = '../raw_data/imgs/'
def loadData(path, mode='coarse'):
    '''
    生成和测试数据key一样的数据，方便本地验证。
    '''
    allData=[]
    with open(path, 'r', encoding='utf-8') as f:
        texts = []
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            data = json.loads(line)
            """
            'img_name', 'title', 'key_attr', 'match', 'feature'
            """
            img_name = data['img_name']
            title = data['title']
            query = ['图文']
            for key in data['key_attr']:
                query.append(key)
            label = data['match']

            sample_data = {}
            sample_data['img_name'] = img_name
            sample_data['title'] = title
            sample_data['query'] = query

            with open(imgs_dir+data['img_name']+'.txt', 'r', encoding='utf-8') as f:
                feature = json.loads(f.readlines()[0])
                sample_data['feature']=feature
                
            sample_data['match'] = label
            allData.append(sample_data)
    return allData

data = loadData(test_data_path)

def write_data(texts, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for line_id, text in enumerate(texts):
            text = json.dumps(text)
            f.write("%s\n" % text)

write_data(data, out_path)