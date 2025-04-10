from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from model import EnhancedResNet
import os
from config import CLASSES

app = Flask(__name__)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedResNet(num_classes=10).to(device)
# 使用训练好的模型
model.load_state_dict(torch.load('../../weight/save/Enhanced_ResNet.pth', map_location=device))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10图像大小为32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取上传的图片
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # 预处理图片
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # 获取所有类别的置信率
            all_probs = []
            for i, prob in enumerate(probabilities):
                all_probs.append({
                    'class': CLASSES[i],
                    'confidence': float(prob)
                })
            
            # 按置信率排序
            all_probs.sort(key=lambda x: x['confidence'], reverse=True)
            
        return jsonify({
            'success': True,
            'results': all_probs
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 