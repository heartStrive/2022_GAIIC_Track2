# 代码说明

## 环境配置

+ 项目使用的镜像：gaiic-semi

+ 项目使用的依赖的版本：python 3.7、pytorch 1.8、tensorflow 2.6、CUDA 11.3

+ 使用了自定义镜像，基础镜像为：GPU CUDA11.3 Python3.7 Tensorflow2.6 PyTorch1.8（腾讯云专用）

+ 额外安装包的名称和版本：transformers 4.11.0、jieba 0.42.1、gensim 4.1.2

## 数据

+ 没有使用额外的公开数据

## 预训练模型

+ 使用了bert-base预训练模型，可以通过URL：https://huggingface.co/ckiplab/bert-base-chinese ，下载获得模型权重，
  对code/bert_base_count1/finetuning/model.py、code/bert_base_lstm_mult_count1/finetuning/model.py、
  code/visual_bert_count1/finetuning/model.py、code/visual_bert_count2/finetuning/model.py中的网络进行初始化

+ 使用了roberta预训练模型，可以通过URL：https://huggingface.co/hfl/chinese-roberta-wwm-ext ，下载获得模型权重，
  对code/visual_roberta_count1/finetuning/model.py中的网络进行初始化

+ 使用了nezha预训练模型，可以通过URL：https://huggingface.co/peterchou/nezha-chinese-base/tree/main ，下载获
  得模型权重，对code/visual_nezha_count2/finetuning/model.py中的网络进行初始化


## 算法

### 整体思路介绍

1. 使用bert提取文本特征，和图像特征融合，匹配/不匹配二分类。

2. 使用bert提取文本特征，和图像特征融合，匹配/不匹配多分类。

3. 模型融合。


### 方法的创新点

1. 基于nezha提出nezha的多模态版本，将文本特征和视觉特征交互融合。

2. 挖掘商品标题中的隐藏属性，构建4种商品类别的8种隐藏属性( "衣门襟", "适用性别", "图案", "上装", "颜色", "鞋款", "裤子类别", "包材质")，涵盖76个细分类别。

3. 基于mask的关键属性匹配。

4. 随机权重平均。

5. 对抗训练FGM。

6. 分层学习率。


### 网络结构

1. 第一种模型，使用bert和nezha的多模态模型，对embedding后的图像特征和文本特征进行融合，最后使用Linear层实现二分类；

2. 第二种模型，文本特征提取使用bert-base，使用LSTM对embedding后的文本特征和图像特征进行特征融合，最后使用Linear层实现多分类；

3. 第三种模型，文本特征提取使用bert-base，使用Concat对embedding后的文本特征和图像特征进行特征融合，最后使用Linear层实现二分类；

### 损失函数

+ 损失函数选用CrossEntropy和BCEWithLogits损失。

### 数据扩增

1. 正样本构造方法
   + 修改关键属性为等价属性
   + 删除关键属性
   + 适当打乱标题
2. 负样本构造方法
   + 修改关键属性为互斥属性
   + 替换互斥的隐藏属性
   + 随机替换商品标题以及同类商品间替换标题
   + 适当打乱标题
3. 特征增强
   + 中心翻转特征
   + 随机擦除特征
   + 打乱部分特征

### 模型集成

1. Voting方法
2. Averaging方法


### 算法的其他细节

1. 优化器使用AdamW

2. 使用了Warmup策略

3. 使用对抗训练FGM
   
## 训练流程

sh train.sh

## 测试流程

sh test.sh data/contest_data/preliminary_testB.txt

## 其他注意事项

1. data/hand_data 中存放是我们根据任务手动构建的文件：
   + user_dict.txt存放的是用于jieba分词的自定义用户词表;
   + color.txt存放的是经过筛选的颜色种类;
   + hidden_attr_to_attrvals.json存放的是手动构建的隐藏属性词表;
   + stop_word.txt存放的是停用词;
2. 训练集样本数：验证集样本数 = 98 : 2

