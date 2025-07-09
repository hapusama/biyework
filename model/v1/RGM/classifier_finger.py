import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import generate_three_loader_v3
from sklearn.model_selection import train_test_split
# 模型配置
batch_size = 128
input_dim=6
# mode='generate'
mode='test'
# mode='original'
num_classes=21
# 批次的大小
input_data_pth=r'model\v1\output\floor3_v3.pth'
lr = 1e-2
# 优化器的学习率
valid_size = 0.2
test_size=0.1
num_epochs = 250
# num_epochs=100
new_path = r'd:\Desktop\PHD\reasearch\biyework\maml'
model_path_train=r'model\v1\output\classifier_ori.pth'
from tqdm import tqdm
# 3. 构建模型
class LocationClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LocationClassifier, self).__init__()
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
    complex_dataset_generated_real=torch.load(input_data_pth)
    train_loader,valid_loader,test_loader=generate_three_loader_v3(complex_dataset_generated_real, 
                                                                   batch_size, 
                                                                   valid_size, 
                                                                   test_size)
    print(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}, test_loader: {len(test_loader)}")
    # 初始化模型
    model = LocationClassifier(input_dim, num_classes)
    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 应用学习率下降策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.1, verbose=True)
    # 设置随机数种子

    # 5. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if mode=='generate':
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_idx,(data_batch_fake,_,_,label_int_batch) in enumerate(tqdm(train_loader)):
                data_batch_fake, label_int_batch = data_batch_fake.to(device), label_int_batch.to(device)
                data_batch_fake = data_batch_fake.squeeze(1)
                # 在训练循环中添加噪声
                data_batch_fake += torch.randn_like(data_batch_fake) * 0.01  # 添加高斯噪声
                # 随机丢弃部分特征
                dropout_mask = torch.rand_like(data_batch_fake) > 0.1  # 90% 的概率保留特征
                data_batch_fake *= dropout_mask
                optimizer.zero_grad()
                outputs = model(data_batch_fake)
                loss = criterion(outputs, label_int_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch_idx,(data_batch_fake,data_batch_real,_,label_int_batch) in enumerate(valid_loader):
                    data_batch_fake, label_int_batch = data_batch_fake.to(device), label_int_batch.to(device)
                    data_batch_fake = data_batch_fake.squeeze(1)
                    # 在训练循环中添加噪声
                    data_batch_fake += torch.randn_like(data_batch_fake) * 0.01  # 添加高斯噪声
                    # 随机丢弃部分特征
                    dropout_mask = torch.rand_like(data_batch_fake) > 0.1  # 90% 的概率保留特征
                    data_batch_fake *= dropout_mask
                    # data_batch_fake = data_batch_fake.view(-1, input_dim)
                    outputs = model(data_batch_fake)
                    loss = criterion(outputs, label_int_batch)
                    valid_loss += loss.item()
                    
            valid_loss /= len(valid_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f},Validation Loss: {valid_loss:.4f}")
            # 更新学习率
            scheduler.step(valid_loss)
            # 如果验证损失没有改善，则保存当前模型
            torch.save(model.state_dict(), model_path_train)
    
    if mode=='original':
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_idx,(data_batch_fake,data_batch_real,_,label_int_batch) in enumerate(tqdm(train_loader)):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                data_batch_real = data_batch_real.squeeze(1)
                optimizer.zero_grad()
                outputs = model(data_batch_real)
                loss = criterion(outputs, label_int_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch_idx,(data_batch_fake,data_batch_real,_,label_int_batch) in enumerate(valid_loader):
                    data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                    data_batch_real = data_batch_real.squeeze(1)
                    # data_batch_fake = data_batch_fake.view(-1, input_dim)
                    outputs = model(data_batch_real)
                    loss = criterion(outputs, label_int_batch)
                    valid_loss += loss.item()
                    
            valid_loss /= len(valid_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f},Validation Loss: {valid_loss:.4f}")
            # 更新学习率
            scheduler.step(valid_loss)
            # 如果验证损失没有改善，则保存当前模型
            torch.save(model.state_dict(), model_path_train)
    
    if mode=='test':
        # 加载模型
        model.load_state_dict(torch.load(model_path_train))
        # 6. 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx,(_,data_batch_real,_,label_int_batch) in enumerate(test_loader):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                data_batch_real = data_batch_real.squeeze(1)
                # data_batch_real = data_batch_real.view(-1, input_dim)
                outputs = model(data_batch_real)
                _, predicted = torch.max(outputs.data, 1)
                total += label_int_batch.size(0)
                # 把匹配成功的点打印出来
                matched_labels = label_int_batch[predicted == label_int_batch]
                print(f"Matched Labels: {matched_labels.cpu().numpy()}")
                
                correct += (predicted == label_int_batch).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

