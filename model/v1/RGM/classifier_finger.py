import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
# 模型配置
batch_size = 256
# 批次的大小
input_data_pth=r'model\v1\input\FLOOR3_v2.csv'
lr = 1e-3
# 优化器的学习率
valid_size = 0.2
test_size=0.1
num_epochs = 300
new_path = r'd:\Desktop\PHD\reasearch\biyework\maml'
model_path_train=r'model\v1\output\classifier_ori.pth'
from tqdm import tqdm
# 3. 构建模型
class LocationClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LocationClassifier, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, num_classes)
        # )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)
        
if "__main__"==__name__:
    df=pd.read_csv(input_data_pth)
    features = ['rssi', 'average_rssi', 'rssi_variance', "average_snr",'snr','sf', 'tp']
    X = df[features].values
    y = df['location_id'].values
    # 2. 数据预处理
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 将标签转换为整数索引
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=42)
    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True)

    # 初始化模型
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    model = LocationClassifier(input_dim, num_classes)
    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 应用学习率下降策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.1, verbose=True)

    # 5. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for X_batch, y_batch in pbar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # 前向传播
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # 更新进度条描述
                pbar.set_postfix(loss=total_loss / len(train_loader))
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch,y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f},Validation Loss: {valid_loss:.4f}")
        # 更新学习率
        scheduler.step(valid_loss)
        # 如果验证损失没有改善，则保存当前模型
        torch.save(model.state_dict(), model_path_train)
    # 加载模型
    model.load_state_dict(torch.load(model_path_train))
    # 6. 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

