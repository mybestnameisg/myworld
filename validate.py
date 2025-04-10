import torch
from torchvision import datasets, transforms
from model import CustomCNN
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载测试集
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

# 加载模型
model = CustomCNN().to(device)
model.load_state_dict(torch.load('./weight/save/custom_cnn.pth', map_location=device))
model.eval()

# 验证
correct = 0
total = 0
with torch.no_grad():
    for data in tqdm(testloader, desc='验证中'):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

accuracy = 100. * correct / total
print(f'\n模型在测试集上的准确率: {accuracy:.2f}%') 