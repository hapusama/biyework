import numpy as np
from sklearn.base import defaultdict
import torch
import os
import pandas as pd
import torch.nn as nn
from src.parameter_paser import parse_args_finetune
import torch.optim as optim
from src.dataset import generate_three_loader_v3,generate_three_loader_v2
from src.dataset import generate_three_dataset_v2, ComplexDatasetLocs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
# 模型配置
batch_size = 128
input_dim=8
mode='generate'
# mode='test'
# mode='original'
# mode="ori"
num_classes=21
# 批次的大小
lr = 1e-2
# 优化器的学习率
valid_size = 0.05
test_size=0.25
num_epochs = 150
# num_epochs=250
new_path = r'd:\Desktop\PHD\research\biyework\maml'
ori_pth = r"model\v1\input\floor3_sf_11_pretrain_dataset.pth"
location_vector_path = r"model\v1\output\location_vector_v2.csv"
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
    base_dir = os.path.dirname(os.path.realpath(__file__))
    args=parse_args_finetune()

    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"
    input_data_pth=os.path.join(output_dir,args.data_name_fake)
    model_path_train=os.path.join(output_dir,args.save_model_name_fake)
    print(f"input_data_pth: {input_data_pth}")
    
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
            for batch_idx, (data_batch_fake, _, _, label_int_batch) in enumerate(tqdm(train_loader)):
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
                    data_batch_fake,data_batch_real, label_int_batch = data_batch_fake.to(device), data_batch_real.to(device), label_int_batch.to(device)
                    data_batch_real = data_batch_real.squeeze(1)
                    data_batch_fake = data_batch_fake.squeeze(1)
                    # 在训练循环中添加噪声
                    data_batch_fake += torch.randn_like(data_batch_fake) * 0.01  # 添加高斯噪声
                    # 随机丢弃部分特征
                    dropout_mask = torch.rand_like(data_batch_fake) > 0.1  # 90% 的概率保留特征
                    data_batch_fake *= dropout_mask
                    # 在验证循环中添加噪声
                    data_batch_real = data_batch_real.squeeze(1)
                    data_batch_real += torch.randn_like(data_batch_real) * 0.01  # 添加高斯噪声
                    # 随机丢弃部分特征
                    dropout_mask = torch.rand_like(data_batch_real) > 0.1  # 90% 的概率保留特征
                    data_batch_real *= dropout_mask
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
        location_vector_df = pd.read_csv(location_vector_path)
        coords = location_vector_df.set_index('idx')[['true_x', 'true_y']].to_dict('index')
        adjacent_list = list(coords.keys())
        with torch.no_grad():
            label_correct = defaultdict(int)
            label_total = defaultdict(int)
            total_distance_error = 0.0
            unmatched_count = 0
            for batch_idx, (_, data_batch_real, _, label_int_batch) in enumerate(test_loader):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                data_batch_real = data_batch_real.squeeze(1)
                outputs = model(data_batch_real)
                _, predicted = torch.max(outputs.data, 1)
                for true_label, pred_label in zip(label_int_batch.cpu().numpy(), predicted.cpu().numpy()):
                    label_total[true_label] += 1
                    if true_label == pred_label:
                        label_correct[true_label] += 1
                    else:
                        # 计算未匹配到的定位误差
                        if true_label in coords and pred_label in coords:
                            x1, y1 = coords[true_label]['true_x'], coords[true_label]['true_y']
                            x2, y2 = coords[pred_label]['true_x'], coords[pred_label]['true_y']
                            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                            total_distance_error += dist
                            unmatched_count += 1
                total += label_int_batch.size(0)
                correct += (predicted == label_int_batch).sum().item()
            if unmatched_count > 0:
                avg_distance_error = total_distance_error / total
                print(f"Average localization error for unmatched labels: {avg_distance_error:.4f}")
            else:
                print("All labels matched, no localization error for unmatched labels.")
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

        labels = list(label_total.keys())
        accuracies = [label_correct[l] / label_total[l] if label_total[l] > 0 else 0 for l in labels]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, accuracies, color='skyblue')
        plt.xlabel('Label')
        plt.ylabel('Accuracy')
        plt.title('Per-label Accuracy')
        plt.xticks(labels)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

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
    if mode=="ori" :
        loaded_data = torch.load(ori_pth)
        rssi = loaded_data['rssi']  # shape [24576,7]
        snr = loaded_data['snr']
        label = loaded_data['label']
        
        complex_dataset = ComplexDatasetLocs(rssi, 
                                            snr, 
                                            label, 
                                            location_vector_path
                                            )
        train_set,valid_set,test_set=generate_three_dataset_v2(complex_dataset, 
                                                                    valid_size, 
                                                                    test_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        for epoch in range(num_epochs):
            total_loss = 0
            for (input_vec,_,label) in tqdm(train_loader):
                input_vec = input_vec.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                outputs = model(input_vec)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch_idx,(input_vec,_,label) in enumerate(valid_loader):
                    input_vec = input_vec.to(device)
                    label = label.to(device)
                    outputs = model(input_vec)
                    loss = criterion(outputs, label)
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
            for batch_idx,(input_vec,_,label) in enumerate(test_loader):
                input_vec, label = input_vec.to(device), label.to(device)
                # input_vec = input_vec.view(-1, input_dim)
                outputs = model(input_vec)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                # 把匹配成功的点打印出来
                matched_labels = label[predicted == label]
                print(f"Matched Labels: {matched_labels.cpu().numpy()}")

                correct += (predicted == label).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")