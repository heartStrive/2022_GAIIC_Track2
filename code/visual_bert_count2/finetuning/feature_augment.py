
import random
import torch

class ContinuousRandomErasing(object):
    def __init__(self, probability=0.5, left=0.02, right=0.33, mean=0.):
        self.probability = probability # 随机擦除概率
        self.mean = mean # 随机擦除填充的value
        self.range = (left, right) # 擦除比率范围，随机生成该区间内的随机数当作是擦除比率

    def __call__(self, feature):
        if random.uniform(0, 1) >= self.probability:
            return feature
        ratio = random.uniform(self.range[0],self.range[1])
        left = random.uniform(0, 1-ratio) # 1-ratio是保证擦除比率
        right = left + ratio # 不能超过1.0

        l = int(round(feature.shape[0]*left))
        r = int(round(feature.shape[0]*right))
        feature[l:r] = self.mean
        return feature

class DiscontinuousRandomErasing(object):
    def __init__(self, probability=0.5, left=0.02, right=0.33, mean=0.):
        self.probability = probability # 随机擦除概率
        self.mean = mean # 随机擦除填充的value
        self.range = (left, right) # 擦除比率范围，随机生成该区间内的随机数当作是擦除比率

    def __call__(self, feature):
        if random.uniform(0, 1) >= self.probability:
            return feature
        ratio = random.uniform(self.range[0],self.range[1])
        erase_idx = random.sample(range(0,feature.shape[0]),int(feature.shape[0]*ratio)) # 生成一定数量索引
        feature[erase_idx] = self.mean
        return feature

class CenterFlip(object):
    def __init__(self, probability=0.5):
        self.probability = probability
    def __call__(self, feature):
        if random.uniform(0, 1) >= self.probability:
            return feature
        feature = torch.flipud(feature)
        return feature

if __name__=='__main__':
    transform = ContinuousRandomErasing()
    feature = torch.rand(20)
    print(feature)
    print(transform(feature))

    transform = DiscontinuousRandomErasing()
    feature = torch.rand(20)
    print(feature)
    print(transform(feature))

    transform = CenterFlip()
    feature = torch.tensor(torch.rand(20))
    print(feature)
    print(transform(feature))