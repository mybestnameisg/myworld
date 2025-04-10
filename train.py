import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import EnhancedResNet
import os
import multiprocessing
from tqdm import tqdm

print("开始导入必要的库...")

# 设置随机种子
torch.manual_seed(42)
print("设置随机种子完成")

# 数据预处理
print("开始设置数据预处理...")
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
print("数据预处理设置完成")

# 加载CIFAR-10数据集
print("开始加载CIFAR-10数据集...")
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
print("训练集加载完成")
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
print("训练数据加载器创建完成")

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
print("测试集加载完成")
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
print("测试数据加载器创建完成")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建模型
print("开始创建模型...")
model = EnhancedResNet().to(device)

# 加载已保存的模型
model_path = './weight/save/enhanced_resnet.pth'
if os.path.exists(model_path):
    print(f"加载已保存的模型: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("模型加载完成")
else:
    print("未找到已保存的模型，将从头开始训练")

print("模型创建完成")

# 定义损失函数和优化器
print("设置损失函数和优化器...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
print("损失函数和优化器设置完成")

# 训练函数
def train(epoch):
    print(f"\n开始训练第 {epoch + 1} 轮...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 添加进度条
    pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}')
    for i, data in enumerate(pbar):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条信息
        pbar.set_postfix({
            'loss': f'{running_loss / (i + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
        
        if i % 100 == 99:
            print(f'\n[Epoch {epoch + 1}, Step {i + 1}] Loss: {running_loss / 100:.4f}, Accuracy: {100. * correct / total:.2f}%')
            running_loss = 0.0

# 验证函数
def validate():
    print("\n开始验证...")
    model.eval()
    correct = 0
    total = 0
    
    # 添加验证进度条
    pbar = tqdm(testloader, desc='Validating')
    with torch.no_grad():
        for data in pbar:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条信息
            pbar.set_postfix({
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    accuracy = 100. * correct / total
    print(f'\n验证准确率: {accuracy:.2f}%')
    return accuracy

def main():
    print("\n开始主程序...")
    # 创建保存目录
    os.makedirs('./weight/save', exist_ok=True)
    print("创建保存目录完成")

    # 训练循环
    best_acc = 0
    for epoch in range(50):
        train(epoch)
        acc = validate()
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), './weight/save/enhanced_resnet.pth')
            print(f"保存新的最佳模型，准确率: {acc:.2f}%")
        
        scheduler.step()

    print('训练完成！')

if __name__ == '__main__':
    print("程序开始执行...")
    main() 