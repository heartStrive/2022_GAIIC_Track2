import sys
sys.path.append('../../process_data')
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import logging
import torch
import random
import os
from torch import nn, optim
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold
from torch.optim.swa_utils import AveragedModel, SWALR
abs_path = os.path.abspath(os.path.dirname(__file__))


class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 6
        self.model = "VisualBertLastCls"
        self.Stratification = False
        self.model_path = os.path.join(abs_path,'../../../data/pretrain_model/pretrained_visual_bert/')
        self.pretrained_bert_path = os.path.join(abs_path,'../../../data/pretrain_model/pretrained_roberta/')
        self.imgs_path = os.path.join(abs_path,'../../../data/tmp_data/imgs/')

        self.num_class = 2
        self.dropout = 0.1
        self.MAX_LEN = 32
        self.epoch = 10
        self.learn_rate = 4e-5
        self.normal_lr = 2e-4
        self.batch_size = 256
        self.k_fold = 10
        self.seed = 2022
        
        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')
        # 优化器选择
        self.use_swa = True
        self.swa_lr = 2e-5
        self.swa_start_step = 22500
        self.swa_steps = 300
        self.swa_model_valid = False

        # 损失函数选择
        self.focalloss = False
        # 对抗训练策略
        self.pgd = False
        self.fgm = True

if __name__=='__main__':


    from model import VisualBertLastCls
    from utils import *
    from evaluation import eval_model
    import time
    import logging
    from utils import loadData, data_cnt
    logging.basicConfig(level=logging.DEBUG, filename="train.log",filemode='a')

    from NEZHA.modeling_nezha import *
    from process_data import generate_train_test_data

    MODEL_CLASSES = {
        'VisualBertLastCls': VisualBertLastCls,
    }

    config = Config()
    os.environ['PYTHONHASHSEED']='0'#消除hash算法的随机性
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    file_path = './log/'
    # 创建一个logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    generate_train_test_data()
    train_data = loadData('../../../data/tmp_data/train_fine_sample.json')
    valid_data = loadData('../../../data/tmp_data/test_fine_sample.json')
    data_cnt('../../../data/tmp_data/train_fine_sample.json')
    data_cnt('../../../data/tmp_data/test_fine_sample.json')
    train_D = data_generator(train_data, config, mode='train', shuffle=True)
    val_D = data_generator(valid_data, config, mode='test')

    visual_bert_model = MODEL_CLASSES[config.model](config).to(config.device) # visual_bert

    # 加载中文预训练bert权重
    pretrained_bert_config = BertConfig.from_pretrained(config.pretrained_bert_path + 'config.json', output_hidden_states=True)
    pretrained_weights = BertModel.from_pretrained(config.pretrained_bert_path, config=pretrained_bert_config).state_dict()
    missing_keys, unexpected_keys=visual_bert_model.bert_model.load_state_dict(pretrained_weights, strict=False)
    print('*'*50)
    print('missing_keys:')
    print(missing_keys)
    print('*'*50)


    if config.use_swa:
        swa_visual_bert_model = AveragedModel(visual_bert_model)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        visual_bert_model = torch.nn.DataParallel(visual_bert_model)

    if config.pgd:
        pgd = PGD(visual_bert_model)
        K = 3

    elif config.fgm:
        fgm = FGM(visual_bert_model)

    if config.focalloss:
        loss_fn = FocalLoss(config.num_class)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    num_train_steps = int(len(train_data) / config.batch_size * config.epoch)
    param_optimizer = list(visual_bert_model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if config.Stratification:
        bert_params = [x for x in param_optimizer if 'bert' in x[0]]
        normal_params = [p for n, p in param_optimizer if 'bert' not in n]
        optimizer_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': normal_params, 'lr': config.normal_lr},
        ]
    else:
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]


    optimizer = AdamW(optimizer_parameters, lr=config.learn_rate) # lr为全局学习率
    if config.use_swa:
        swa_scheduler = SWALR(optimizer,swa_lr=config.swa_lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_data) / config.batch_size / 2),
        num_training_steps=num_train_steps
    )

    global_step = 0
    best_score = 0
    PATH = '../../../data/best_model_{}.pth'.format(str(config.modelId))
    save_model_path = '../../../data/model_data/visual_roberta_count1/'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    for e in range(config.epoch):
        if e>0:
            generate_train_test_data()
            train_data = loadData('../../../data/tmp_data/train_fine_sample.json')
            valid_data = loadData('../../../data/tmp_data/test_fine_sample.json')
            train_D = data_generator(train_data, config, mode='train', shuffle=True)
            val_D = data_generator(valid_data, config, mode='test')

        print('\n------------epoch:{}------------'.format(e))
        visual_bert_model.train()
        acc = 0
        train_len = 0
        loss_num = 0
        tq = tqdm(train_D,ncols=70,disable=False)
        last=time.time()
        for input_ids, input_masks, segment_ids, targets in tq:
            if input_ids.shape[0]<config.batch_size:
                continue
            label_t = torch.tensor(targets['label'], dtype=torch.long).to(config.device)
            features = torch.tensor(targets['feature'], dtype=torch.float).to(config.device)
            y_pred = visual_bert_model(input_ids, input_masks, segment_ids, features)
            
            loss = loss_fn(y_pred[0].view(-1,2), label_t[:,0])
            for i in range(1,13):
                tmp_loss = loss_fn(y_pred[i].view(-1,2), label_t[:,i])
                loss += tmp_loss
            loss.backward()

            if config.pgd:
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K - 1:
                        visual_bert_model.zero_grad()
                    else:
                        pgd.restore_grad()
                    y_pred = visual_bert_model(input_ids, input_masks, segment_ids, features)

                    loss = loss_fn(y_pred[0].view(-1,2), label_t[:,0])
                    for i in range(1,13):
                        tmp_loss = loss_fn(y_pred[i].view(-1,2), label_t[:,i])
                        loss += tmp_loss
                    loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数

            elif config.fgm:
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                y_pred = visual_bert_model(input_ids, input_masks, segment_ids, features)
                loss = loss_fn(y_pred[0].view(-1,2), label_t[:,0])
                for i in range(1,13):
                    tmp_loss = loss_fn(y_pred[i].view(-1,2), label_t[:,i])
                    loss += tmp_loss
                loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数

            # 梯度下降，更新参数
            optimizer.step()
            if config.use_swa and (global_step + 1) > config.swa_start_step:
                if (global_step + 1) % config.swa_steps == 0:
                    config.swa_model_valid = True
                    swa_visual_bert_model.update_parameters(visual_bert_model)
                swa_scheduler.step()
            else:
                scheduler.step()  # Update learning rate schedule
            visual_bert_model.zero_grad()

            global_step += 1
            loss_num += loss.item()
            train_len += 3 # label_t.shape[0]
            tq.set_postfix(epoch=e, step=global_step, loss=loss_num / train_len)
        # print(f"微调第{e}轮耗时：{time.time()-last}")

        if config.swa_model_valid:
            cur_score = eval_model(config, swa_visual_bert_model, val_D)
        else:
            cur_score = eval_model(config, visual_bert_model, val_D)

        print("best_score:{:.4f}  cur_score:{:.4f} \n".format(best_score, cur_score))
        if cur_score >= best_score:
            best_score = cur_score
            if config.swa_model_valid:
                torch.save(swa_visual_bert_model.state_dict(), PATH)
            else:
                torch.save(visual_bert_model.state_dict(), PATH)

        if config.swa_model_valid:
            torch.save(swa_visual_bert_model.state_dict(), save_model_path+'epoch_{}.pth'.format(e))
        else:
            torch.save(visual_bert_model.state_dict(), save_model_path+'epoch_{}.pth'.format(e))
    optimizer.zero_grad()

    del visual_bert_model
    if config.use_swa:
        del swa_visual_bert_model
    torch.cuda.empty_cache()
