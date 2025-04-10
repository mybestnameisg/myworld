# 数据集配置
DATASET = {
    'name': 'CIFAR10',
    'num_classes': 10,
    'image_size': 32,
    'train_size': 50000,
    'test_size': 10000
}

# 模型配置
MODEL = {
    'name': 'CustomCNN',
    'initial_channels': 64,
    'num_blocks': 3,
    'dropout_rate': 0.5
}

# 训练配置
TRAIN = {
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 0.003,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'scheduler': {
        'type': 'cosine',
        'T_max': 50
    }
}

# 路径配置
PATH = {
    'data': './data',
    'weight': './weight/save',
    'model_save': './weight/save/custom_cnn.pth'
}

# 类别标签
CLASSES = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
] 