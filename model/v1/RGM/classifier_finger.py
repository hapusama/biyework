import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.dataset import generate_three_loader_v4
from src.dataset import RealorFakeDataset

# 模型配置
batch_size = 256
input_dim=6
mode='generate'
# mode='test'
# mode='original'
num_classes=21
# 批次的大小
fake_data_pth=r'model\v1\output\floor3_sf_11_fake.pth'
real_data_pth=r'model\v1\input\finger_sf_11_floor3_dataset.pth'
lr = 1e-3
# 优化器的学习率
valid_size = 0.05
test_size=0.2
num_epochs = 100
new_path = r'd:\Desktop\PHD\reasearch\biyework\maml'
model_path_train=r'model\v1\output\classifier_ori.pth'
location_vector_path = r'model\v1\output\location_vector_v3.csv'
# todo 最后重新设置一个总共的yml，尽量一到两个，把参数全都统一写入yml中
from tqdm import tqdm
# 3. 构建模型
class LocationClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LocationClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)
        
if "__main__"==__name__:

    real_data=torch.load(real_data_pth)
    # 加载数据集
    real_dataset = RealorFakeDataset(real_data['features'], real_data['label'])
    real_train_loader, real_valid_loader, real_test_loader = generate_three_loader_v4(real_dataset,
                                                                batch_size,
                                                                valid_size,
                                                                test_size)
    # 初始化模型
    model = LocationClassifier(input_dim, num_classes)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 应用学习率下降策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1, verbose=True)
    # 设置随机数种子
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if mode=='generate':
        fake_dataset=torch.load(fake_data_pth)
        train_loader,valid_loader,test_loader=generate_three_loader_v4(fake_dataset,
                                                                batch_size, 
                                                                valid_size, 
                                                                test_size)
        print(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}, test_loader: {len(test_loader)}")        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_idx,(data_batch_fake,label_int_batch) in enumerate(tqdm(train_loader)):
                data_batch_fake, label_int_batch = data_batch_fake.to(device), label_int_batch.to(device)
                # 在训练循环中添加噪声
                # data_batch_fake += torch.randn_like(data_batch_fake) * 0.01  # 添加高斯噪声
                optimizer.zero_grad()
                outputs = model(data_batch_fake)
                loss = criterion(outputs, label_int_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            # 用真实数据集测试模型
            valid_loss = 0
            with torch.no_grad():
                for batch_idx,(data_batch_fake,label_int_batch) in enumerate(real_valid_loader):
                    data_batch_fake, label_int_batch = data_batch_fake.to(device), label_int_batch.to(device)
                    # 在训练循环中添加噪声
                    # data_batch_fake += torch.randn_like(data_batch_fake) * 0.01  # 添加高斯噪声
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
             # 加载模型
        model.load_state_dict(torch.load(model_path_train))
        # 6. 测试模型
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx,(data_batch_real,label_int_batch) in enumerate(real_test_loader):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                # data_batch_real = data_batch_real.view(-1, input_dim)
                outputs = model(data_batch_real)
                _, predicted = torch.max(outputs.data, 1)
                total += label_int_batch.size(0)
                # 把匹配成功的点打印出来
                matched_labels = label_int_batch[predicted == label_int_batch]
                print(f"Matched Labels: {matched_labels.cpu().numpy()}")
                # 把匹配失败的点打印出来，并且将其对应的预测值也打印出来
                mismatched_labels = label_int_batch[predicted != label_int_batch]
                mismatched_predictions = predicted[predicted != label_int_batch]
                print(f"Mismatched Labels: {mismatched_labels.cpu().numpy()}, Predictions: {mismatched_predictions.cpu().numpy()}")
                
                correct += (predicted == label_int_batch).sum().item()
                # 并且打印每一个label预测正确的次数
                for i in range(num_classes):
                    correct_count = (predicted[label_int_batch == i] == i).sum().item()
                    print(f"Label {i} Correct Count: {correct_count}")
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        loc_df=pd.read_csv(location_vector_path)
        # 计算每一个点预测的召回率
        for i in range(num_classes):
            # 计算每一个点预测的召回率
            true_positive = (predicted[label_int_batch == i] == i).sum().item()
            false_negative = (predicted[label_int_batch != i] == i).sum().item()
            # 分母为label_int_batch中i的个数
            # 计算召回率
            nums=(label_int_batch==i).sum().item()
            recall = true_positive / nums if nums > 0 else 0
            # recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            loc_label=loc_df[loc_df['idx']==i]['location_id'].values[0]
            print(f"Label {loc_label} Recall: {recall:.4f}")
            
        # # 计算每一个点预测的精确率
        # for i in range(num_classes):
        #     # 计算每一个点预测的精确率
        #     true_positive = (predicted[label_int_batch == i] == i).sum().item()
        #     false_positive = (predicted[label_int_batch != i] == i).sum().item()
        #     precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        #     loc_label=loc_df[loc_df['idx']==i]['location_id'].values[0]
        #     print(f"Label {loc_label} Precision: {precision:.4f}")
    if mode=='original':
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_idx,(data_batch_real,label_int_batch) in enumerate(tqdm(real_train_loader)):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                optimizer.zero_grad()
                outputs = model(data_batch_real)
                loss = criterion(outputs, label_int_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch_idx,(data_batch_real,label_int_batch) in enumerate(real_valid_loader):
                    data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                    # data_batch_fake = data_batch_fake.view(-1, input_dim)
                    outputs = model(data_batch_real)
                    loss = criterion(outputs, label_int_batch)
                    valid_loss += loss.item()
                    
            valid_loss /= len(real_valid_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(real_train_loader):.4f},Validation Loss: {valid_loss:.4f}")
            # 更新学习率
            scheduler.step(valid_loss)
            # 如果验证损失没有改善，则保存当前模型
            torch.save(model.state_dict(), model_path_train)
            model.load_state_dict(torch.load(model_path_train))
        # 6. 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx,(data_batch_real,label_int_batch) in enumerate(real_test_loader):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                # data_batch_real = data_batch_real.view(-1, input_dim)
                outputs = model(data_batch_real)
                _, predicted = torch.max(outputs.data, 1)
                total += label_int_batch.size(0)
                # 把匹配成功的点打印出来
                matched_labels = label_int_batch[predicted == label_int_batch]
                print(f"Matched Labels: {matched_labels.cpu().numpy()}")
                # 把匹配失败的点打印出来，并且将其对应的预测值也打印出来
                mismatched_labels = label_int_batch[predicted != label_int_batch]
                mismatched_predictions = predicted[predicted != label_int_batch]
                print(f"Mismatched Labels: {mismatched_labels.cpu().numpy()}, Predictions: {mismatched_predictions.cpu().numpy()}")
                
                correct += (predicted == label_int_batch).sum().item()
                # 并且打印每一个label预测正确的次数
                for i in range(num_classes):
                    correct_count = (predicted[label_int_batch == i] == i).sum().item()
                    print(f"Label {i} Correct Count: {correct_count}")
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        # 计算每一个点预测的召回率
        for i in range(num_classes):
            # 计算每一个点预测的召回率
            true_positive = (predicted[label_int_batch == i] == i).sum().item()
            false_negative = (predicted[label_int_batch != i] == i).sum().item()
            nums=(label_int_batch==i).sum().item()
            recall = true_positive / nums if nums > 0 else 0
            loc_df=pd.read_csv(location_vector_path)
            loc_label=loc_df[loc_df['idx']==i]['location_id'].values[0]
            # recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            print(f"Label {loc_label} Recall: {recall:.4f}")
    if mode=='test':
        # 加载模型
        model.load_state_dict(torch.load(model_path_train))
        # 6. 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx,(data_batch_real,label_int_batch) in enumerate(real_test_loader):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                data_batch_real = data_batch_real.squeeze(1)
                # data_batch_real = data_batch_real.view(-1, input_dim)
                outputs = model(data_batch_real)
                _, predicted = torch.max(outputs.data, 1)
                total += label_int_batch.size(0)
                # 把匹配成功的点打印出来
                matched_labels = label_int_batch[predicted == label_int_batch]
                print(f"Matched Labels: {matched_labels.cpu().numpy()}")
                # 把匹配失败的点打印出来，并且将其对应的预测值也打印出来
                mismatched_labels = label_int_batch[predicted != label_int_batch]
                mismatched_predictions = predicted[predicted != label_int_batch]
                print(f"Mismatched Labels: {mismatched_labels.cpu().numpy()}, Predictions: {mismatched_predictions.cpu().numpy()}")
                
                correct += (predicted == label_int_batch).sum().item()
                # 并且打印每一个label预测正确的次数
                for i in range(num_classes):
                    correct_count = (predicted[label_int_batch == i] == i).sum().item()
                    print(f"Label {i} Correct Count: {correct_count}")
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        
   