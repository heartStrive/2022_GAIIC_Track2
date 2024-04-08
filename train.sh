{
   cd code/process_data # 进入处理数据的路径
   python split_data.py # 将图像特征单独取出存储
   python generate_pseudo_samples.py # 生成用户自定义词表
   cd ../.. # 返回项目初始路径
}
wait # 等待子进程结束

{
    # 进入特定模型路径，指定显卡并消除hash算法的随机性训练模型
    (cd ./code/bert_base_count1/finetuning/ && PYTHONUNBUFFERED=1 python train_classifier.py)
}
wait # 等待子进程结束

{
    # 进入特定模型路径，指定显卡并消除hash算法的随机性训练模型
    (cd ./code/bert_base_lstm_mult_count1/finetuning/ && PYTHONUNBUFFERED=1 python train_classifier.py)
}
wait # 等待子进程结束

{
    # 进入特定模型路径，指定显卡并消除hash算法的随机性训练模型
    (cd ./code/visual_bert_count1/finetuning/ && PYTHONUNBUFFERED=1 python train_classifier.py)
}
wait # 等待子进程结束

{
    # 进入特定模型路径，指定显卡并消除hash算法的随机性训练模型
    (cd ./code/visual_bert_count2/finetuning/ && PYTHONUNBUFFERED=1 python train_classifier.py)
}
wait # 等待子进程结束

{
    # 进入特定模型路径，指定显卡并消除hash算法的随机性训练模型
    (cd ./code/visual_roberta_count1/finetuning/ && PYTHONUNBUFFERED=1 python train_classifier.py)
}
wait # 等待子进程结束

{
    # 进入特定模型路径，指定显卡并消除hash算法的随机性训练模型
    (cd ./code/visual_nezha_count2/finetuning/ && PYTHONUNBUFFERED=1 python train_classifier.py)
}
wait # 等待子进程结束
