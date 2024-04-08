import json
import logging
import os
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter
from transformers import BertTokenizer, AdamW, BertModel,BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig, ElectraModel, ElectraConfig, ElectraTokenizer, \
    RobertaTokenizer, RobertaModel, RobertaConfig
from NEZHA.modeling_nezha import NeZhaModel
from NEZHA.configuration_nezha import NeZhaConfig

MODELS = {
    'Bert': BertModel,
    'XLNet': XLNetModel,
    'Electra': ElectraModel,
    'NEZHA': NeZhaModel
    }

TOKENIZERS = {
    'Bert': BertTokenizer,
    'XLNet': XLNetTokenizer,
    'Electra': ElectraTokenizer,
    'NEZHA': BertTokenizer
    }

CONFIGS = {
    'Bert':BertConfig,
    'XLNet': XLNetConfig,
    'Electra': ElectraConfig,
    'NEZHA': NeZhaConfig
    }

class Config:
    def __init__(self):
        # 预训练模型路径
        self.model_path = '../pretrained_bert/'
        self.MAX_LEN = 32
        self.model = "Bert"
        self.vocab = 'vocab.txt'

logging.basicConfig()
logger = logging.getLogger('convert_data')
logger.setLevel(logging.INFO)

# 将数据写入文件
def write_to_json(results, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for line_id, res in enumerate(results):
            text = json.dumps(res)
            f.write("%s\n" % text)

def process_oov_record(config, text, normal_vocab,
                       idmap,
                       min_frequence=5,
                       min_oov_word_idx=35000):

    tokens = config.tokenizer.tokenize(text)
    new_tokens = []
    oov_word_map = {}
    cur_oov_idx = min_oov_word_idx
    for tokens in tokens:
        for i in range(len(tokens)):
            if normal_vocab.get(tokens[i], 0) < min_frequence:
                if tokens[i] not in oov_word_map:
                    oov_word_map[tokens[i]] = str(cur_oov_idx)
                    cur_oov_idx += 1
                new_tokens.append(oov_word_map[tokens[i]])
            else:
                new_tokens.append(idmap[tokens[i]])
    return " ".join(new_tokens)


def construct_normal_vocab(config, input_file,
                           output_file,
                           output_idmap_file):
    input_json = open(input_file, 'r', encoding='utf-8')
    word_ct = Counter()
    normal_voc = {}
    for row in tqdm(input_json):
        data = json.loads(row)
        text = data['title']
        text = config.tokenizer.tokenize(text)
        word_ct.update(text)
    idmap = {}
    for idx, (word, num) in enumerate(word_ct.most_common()):
        normal_voc[word] = num
        idmap[word] = str(idx)
    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(normal_voc, fout)
    with open(output_idmap_file, 'w', encoding='utf-8') as fout:
        json.dump(idmap, fout, ensure_ascii=False, indent=2)

    return normal_voc, idmap

def process_oov_json(config, input_json, normal_vocab, idmap, min_frequence=3):
    all_data = []
    for row in tqdm(input_json):
        data = json.loads(row)
        text = data['title']
        text = process_oov_record(config, text, normal_vocab,
                                            idmap,
                                            min_frequence=min_frequence)
        data['title'] = text
        all_data.append(data)
    return all_data

def process_oov_file(config, input_file,
                     output_file,
                     normal_vocab,
                     idmap,
                     mode='train'):
    input_json = open(input_file, 'r', encoding='utf-8')
    output_json = process_oov_json(config, input_json, normal_vocab, idmap, min_frequence=3)
    write_to_json(output_json, output_file)

def process_oov_words(config, train_file='../raw_data/train_coarse_sample.json',
                      test_file='../raw_data/train_coarse_sample.json',
                      output_dir='../raw_data'):
    os.makedirs(output_dir, exist_ok=True)
    logger.info('construct normal vocabulary...')
    normal_vocab, idmap = construct_normal_vocab(config, train_file,
                                          os.path.join(output_dir, 'normal_vocab.json'),
                                          os.path.join(output_dir, 'idmap.json'))
    logger.info('process oov file...')
    process_oov_file(config, train_file, os.path.join(output_dir, 'train.json'), normal_vocab, idmap,
                     mode='train')
    process_oov_file(config, test_file, os.path.join(output_dir, 'test.json'), normal_vocab, idmap,
                     mode='test')

    ct = 0
    tail = []
    for k, v in normal_vocab.items():
        if v >= ct:  # 统计词频大于ct的词
            tail.append(k)
    tail.sort()
    pre = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', ]  # 加入一些必须的符号
    vocab = pre + tail
    with open('../raw_data/vocab.txt', "w", encoding="utf-8") as f:
        for i in vocab:
            f.write(str(i) + '\n')

if __name__ == '__main__':
    config = Config()
    print(config.model_path + config.vocab)
    config.tokenizer = TOKENIZERS[config.model].from_pretrained(config.model_path + config.vocab)

    process_oov_words(config)