# 增强型ResNet图像分类系统

这是一个基于PyTorch实现的图像分类系统，使用增强型ResNet架构对CIFAR-10数据集进行训练和预测。

## 项目结构

```
EnhancedResNet/         #自用模型
├── model.py          # 模型定义
├── train.py          # 训练脚本
├── config.py         # 配置文件
├── web_app.py        # Web应用服务器
├── templates/        # Web界面模板
│   └── index.html    # 主页面
├── data/             # 数据集目录
└── weight/           # 模型权重保存目录
    └── save/         # 训练好的模型保存位置
以上文件夹请自主创建
```

## 模型特点

- 使用增强型ResNet架构
- 包含残差连接
- 使用通道注意力机制
- 采用Dropout正则化
- 使用Kaiming初始化

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- torchvision
- Flask
- Pillow
- CUDA (可选，用于GPU加速)

## 使用方法

1. 安装依赖：
```bash
pip install torch torchvision flask pillow
```

2. 训练模型：
```bash
python train.py
```

3. 运行Web应用：
```bash
python web_app.py
```
然后在浏览器中访问 http://localhost:5000

4. 模型配置：
可以在 `config.py` 中修改训练参数，包括：
- 批次大小
- 学习率
- 训练轮数
- 模型结构参数

## 数据集

使用CIFAR-10数据集，包含以下10个类别：
- 飞机
- 汽车
- 鸟
- 猫
- 鹿
- 狗
- 青蛙
- 马
- 船
- 卡车

## 训练参数

- 批次大小：64
- 学习率：0.003
- 动量：0.9
- 权重衰减：1e-4
- 训练轮数：50
- 学习率调度：余弦退火

## 模型保存

训练过程中会自动保存验证准确率最高的模型到 `weight/save/Enhanced_ResNet.pth`。 
