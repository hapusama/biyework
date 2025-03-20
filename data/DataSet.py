from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class rssiDataset(Dataset):
    #之后要是csv文件数据量太大了，尝试将数据集分成多个文件，然后在这里读取多个文件
    def __init__(self, csv_file_path, mode="train", val_ratio=0.2, test_ratio=0.1):
        self.data = pd.read_csv(csv_file_path)
        self.mode = mode
        # 分割数据集为训练集和验证集，还有测试集
         # 分割数据集为训练集、验证集和测试集
        indices = np.arange(len(self.data))
        train_indices, temp_indices = train_test_split(indices, test_size=(val_ratio + test_ratio), random_state=42, shuffle=True)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_ratio/(val_ratio + test_ratio), random_state=42, shuffle=True)

        if self.mode == 'train':
            self.indices = train_indices
        elif self.mode == 'val':
            self.indices = val_indices
        elif self.mode == 'test':
            self.indices = test_indices
                    


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = {
            'location_id': self.data["location_id"].iloc[actual_idx],
            'rssi': self.data["rssi"].iloc[actual_idx],
            'sf': self.data["sf"].iloc[actual_idx],
            'snr': self.data["snr"].iloc[actual_idx],
            'tp': self.data["tp"].iloc[actual_idx]
        }
        return sample

if __name__ == "__main__":
    path=r"data\processedData\all_data_new.csv"
    train_dataset = rssiDataset(path, mode='train')
    val_dataset = rssiDataset(path, mode='val')
    test_dataset = rssiDataset(path, mode='test')
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 